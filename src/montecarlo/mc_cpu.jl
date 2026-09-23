# Monte Carlo — mc_cpu.jl
#
# CPU reference drivers: the full generate -> track -> record event loops
# (sphere and GRMHD), threaded over superphotons; the GPU port is validated
# against them. Normalisation: the weight table fixes w(nu) = V int_jnu_zone /
# (h Ns_param) and the generated frequency density is prop. to int_jnu_zone/w
# = const (uniform in ln nu), so generating N_target = Ns_param * band packets
# makes nu L_nu(bin) = (m_e c^2 / dln nu) Sum w_j E_j an unbiased estimate of
# nu V int_jnu_zone(nu) with no free constant (Ns_param = Ns/band -> N_target ~ Ns).

"""
    mc_spectrum(model, Ns; nthreads, seed, nu_min, nu_max, n_esamp, n_ebins, n_thbins)
        -> Spectrum

Run the sphere Monte Carlo: build the one-zone weight table, generate ~`Ns`
superphotons (uniform-random emission points in the ball), track each to
escape and bin it into the returned `Spectrum`. Photon `i` owns the Philox
stream `mc_primary_rng(seed, i)`; the `nthreads` chunks accumulate exactly
(`ExactSpectrum`) and merge by integer addition, so the result is bitwise
independent of `nthreads` and of scheduling.
`nu_min`/`nu_max` set both the generation band and the spectrum grid.
`tabulated_gen = true` opts into the tabulated frequency rejection
(much cheaper generation, statistically identical but not guaranteed bitwise).
"""
function mc_spectrum(model::SphereModel, Ns::Real;
                     nthreads::Integer = Threads.nthreads(),
                     seed::Integer = 0x5eed5eed,
                     nu_min::Real = _MC_NU_MIN_DEFAULT,
                     nu_max::Real = _MC_NU_MAX_DEFAULT,
                     n_esamp::Integer = _MC_N_ESAMP_DEFAULT,
                     n_ebins::Integer = 200, n_thbins::Integer = 6,
                     tabulated_gen::Bool = false)
    band = log(nu_max) - log(nu_min)
    Ns_param = Float64(Ns) / band                       # so N_target ~ Ns
    zone = sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0))
    wgt = build_weight_table([zone], Ns_param;
                             n_esamp = n_esamp, nu_min = nu_min, nu_max = nu_max)
    N_target, dnmax = zone_nsuperphotons(zone, wgt)
    N_gen = max(1, round(Int, N_target))
    gt = tabulated_gen ? build_int_jnu_table(zone, wgt) : nothing
    tabulated_gen && (dnmax = tabulated_dnmax(gt, wgt))  # envelope of the interpolant

    nchunks = max(1, Int(nthreads))
    specs = [ExactSpectrum(; nu_min = nu_min, nu_max = nu_max,
                           n_ebins = n_ebins, n_thbins = n_thbins) for _ in 1:nchunks]
    _mc_sphere_loop!(specs, model, wgt, dnmax, N_gen, Int(seed), gt)

    total = ExactSpectrum(; nu_min = nu_min, nu_max = nu_max,
                          n_ebins = n_ebins, n_thbins = n_thbins)
    for s in specs
        merge_spectra!(total, s)
    end
    return Spectrum(total)
end

# =========================================================================
# SCATTERING SPHERE (idealized Comptonization sphere)
# =========================================================================
#
# Multi-scatter event loop: generate synchrotron seed superphotons, track each
# with the biased scatter decision (track_superphoton_scatter!), and
# iteratively drain the scattered children through a per-thread work queue
# (layered, not recursive). Escaped photons are binned into per-scattering-order
# spectra so the successive Compton up-scattering bumps (and the unscattered
# nscatt=0 direct emission) can be validated separately.

"""
    ScatterResult

Output of `mc_spectrum_scatter`. `orders[n+1]` is the escaped-`Spectrum` of photons
with scattering order `n` (`orders[end]` lumps `nscatt >= max_scatt`); `total` sums
all orders. Diagnostics: primary/child counts, the created primary and child
weights, the first-order child weight (spawned by primaries) and `scatter_loss`,
the total incident weight lost to scattering (the unbiasedness conservation law
checks `w_children ≈ scatter_loss`).
"""
struct ScatterResult
    orders::Vector{Spectrum}
    total::Spectrum
    max_scatt::Int
    n_primary::Int
    n_children::Int
    w_primary::Float64
    w_children::Float64
    w_first_order::Float64
    scatter_loss::Float64
end

"""
    mc_spectrum_scatter(model, table, Ns; kwargs...) -> ScatterResult

Run the scattering-sphere Monte Carlo: build the synchrotron weight table over
the generation band `[gen_nu_min, gen_nu_max]`, generate ~`Ns` seed
superphotons, and track+scatter each into per-order spectra on the wider
spectrum grid `[spec_nu_min, spec_nu_max]` (which must reach the up-scattered
bumps). `table` is the hot-cross lookup; `max_scatt` caps the scattering
generations. Threaded over primaries with private exact per-order
accumulators; a child's stream is keyed by its genealogy (`mc_child_rng`), so
the children, the spectra and the diagnostics are bitwise independent of
`nthreads` and of scheduling.
"""
function mc_spectrum_scatter(model::SphereModel, table::HotcrossTable, Ns::Real;
                             nthreads::Integer = Threads.nthreads(),
                             seed::Integer = 0x5ca77e5,
                             gen_nu_min::Real = 1.0e8, gen_nu_max::Real = 1.0e13,
                             spec_nu_min::Real = 1.0e8, spec_nu_max::Real = 1.0e17,
                             n_esamp::Integer = _MC_N_ESAMP_DEFAULT,
                             n_ebins::Integer = 140, n_thbins::Integer = 6,
                             max_scatt::Integer = 3, tabulated_gen::Bool = false)
    band = log(gen_nu_max) - log(gen_nu_min)
    Ns_param = Float64(Ns) / band
    zone = sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0))
    wgt = build_weight_table([zone], Ns_param;
                             n_esamp = n_esamp, nu_min = gen_nu_min, nu_max = gen_nu_max)
    N_target, dnmax = zone_nsuperphotons(zone, wgt)
    N_gen = max(1, round(Int, N_target))
    gt = tabulated_gen ? build_int_jnu_table(zone, wgt) : nothing
    tabulated_gen && (dnmax = tabulated_dnmax(gt, wgt))  # envelope of the interpolant
    nchunks = max(1, Int(nthreads))
    K = Int(max_scatt)

    layout = MCStreamLayout(N_gen, _MC_TRACK_MAX_STEPS, K)

    mkorders() = [ExactSpectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                                n_ebins = n_ebins, n_thbins = n_thbins) for _ in 0:K]
    orders_by_chunk = [mkorders() for _ in 1:nchunks]
    # per-chunk exact diagnostics: 1 primary weight, 2 scatter loss
    diag_by_chunk = [ExactScalars(2) for _ in 1:nchunks]
    ws_by_chunk = [ScatterWorkspace(layout) for _ in 1:nchunks]

    _mc_scatter_loop!(orders_by_chunk, diag_by_chunk, ws_by_chunk, model, table, wgt,
                      dnmax, gt, N_gen, Int(seed), K)

    orders = mkorders()
    diag = ExactScalars(2)
    wdiag = ExactScalars(2)
    for c in 1:nchunks
        for n in 1:(K + 1)
            merge_spectra!(orders[n], orders_by_chunk[c][n])
        end
        merge_scalars!(diag, diag_by_chunk[c])
        merge_scalars!(wdiag, ws_by_chunk[c].diag)
    end
    return _scatter_result(orders, K, N_gen, sum(ws.n_children for ws in ws_by_chunk),
                           xsum_round(diag, 1), xsum_round(wdiag, 1),
                           xsum_round(wdiag, 2), xsum_round(diag, 2))
end

# Round exact per-order spectra into a `ScatterResult`; the total is the exact
# sum over orders, rounded once.
function _scatter_result(xorders::Vector{ExactSpectrum}, K::Int, n_primary::Int,
                         n_children::Int, w_primary, w_children, w_first_order,
                         scatter_loss)
    x1 = xorders[1]
    xtotal = ExactSpectrum(; nu_min = x1.nu_min, nu_max = x1.nu_max,
                           n_ebins = x1.n_ebins, n_thbins = x1.n_thbins)
    for o in xorders
        merge_spectra!(xtotal, o)
    end
    return ScatterResult([Spectrum(o) for o in xorders], Spectrum(xtotal), K, n_primary,
                         n_children, w_primary, w_children, w_first_order, scatter_loss)
end

"""
    _mc_scatter_loop!(orders_by_chunk, diag_by_chunk, ws_by_chunk, model, table, wgt,
                      dnmax, gt, N_gen, seed, max_scatt) -> orders_by_chunk

Threaded generate/track/scatter/record kernel over contiguous chunks of the
primaries `1:N_gen`. Each chunk owns exact per-order accumulators, exact
diagnostics (primary weight, scatter loss) and a `ScatterWorkspace`; primary
`i` draws from `mc_primary_rng(seed, i)` and every child from its genealogy
stream, so nothing a photon does depends on its chunk. Each primary is pushed,
then the work queue is drained: pop, track+scatter, record into the
scattering-order bin.
"""
function _mc_scatter_loop!(orders_by_chunk, diag_by_chunk, ws_by_chunk,
                           model::SphereModel, table::HotcrossTable,
                           wgt::WeightTable, dnmax::Float64,
                           gt::Union{Nothing,IntJnuTable}, N_gen::Int, seed::Int,
                           max_scatt::Int)
    nchunks = length(orders_by_chunk)
    Threads.@threads for c in 1:nchunks
        orders = orders_by_chunk[c]
        diag = diag_by_chunk[c]
        ws = ws_by_chunk[c]
        lo = div((c - 1) * N_gen, nchunks) + 1
        hi = div(c * N_gen, nchunks)
        for i in lo:hi
            rng = mc_primary_rng(seed, i)
            sp, rng = sample_sphere_photon(model, wgt, dnmax, rng, gt)
            xsum_add!(diag, 1, sp.w)
            push_photon!(ws, sp, rng)
            while !isempty(ws.sp)
                cur, crng = pop_photon!(ws)
                esc, recorded, loss, crng =
                    track_superphoton_scatter!(ws, cur, model, table, crng; max_scatt = max_scatt)
                xsum_add!(diag, 2, loss)
                if recorded
                    ord = min(Int(esc.nscatt), max_scatt) + 1
                    record!(orders[ord], esc)
                end
            end
        end
    end
    return orders_by_chunk
end

"""
    _mc_sphere_loop!(specs, model, wgt, dnmax, N_gen, seed, gt) -> specs

Threaded generate/track/record kernel: the `1:N_gen` packets are split into
contiguous chunks, each processed into its own private `ExactSpectrum`, so the
result is race-free regardless of task migration. Superphoton `i` owns the
Philox stream `mc_primary_rng(seed, i)`.
"""
function _mc_sphere_loop!(specs::Vector{ExactSpectrum}, model::SphereModel,
                          wgt::WeightTable, dnmax::Float64, N_gen::Int, seed::Int,
                          gt::Union{Nothing,IntJnuTable} = nothing)
    nchunks = length(specs)
    Threads.@threads for c in 1:nchunks
        spec = specs[c]
        lo = div((c - 1) * N_gen, nchunks) + 1
        hi = div(c * N_gen, nchunks)
        for i in lo:hi
            rng = mc_primary_rng(seed, i)
            sp, rng = sample_sphere_photon(model, wgt, dnmax, rng, gt)
            sp, recorded = track_superphoton(sp, model, rng)
            recorded && record!(spec, sp)
        end
    end
    return specs
end

# =========================================================================
# GRMHD MONTE CARLO
# =========================================================================
#
# Emergent-spectrum driver for an AthenaK GRMHD snapshot: generate seed
# thermal-synchrotron superphotons over the emitting grid (weighted by each
# zone's emission integral, grmonty `init_weight_table`/`init_zone`), track
# each along a curved Kerr-Schild null geodesic (RK4, step h = (r - r_h)/C,
# sampling the fluid every step for absorption + the biased Compton scatter
# trigger), and bin the escaped photons by the polar angle of their asymptotic
# escape direction into viewing-inclination bands x log-nu -> nu L_nu. The
# transport core reuses the sphere estimators unchanged.

# Horizon capture buffer (matches trace_rays_sampled's r_h + 1e-3 stop).
const _GRMHD_HORIZON_EPS = 1.0e-3
# Step cap per superphoton: a physical photon escapes/plunges in O(C ln(R/r_emit))
# ~ 10^2-10^3 steps; a rare packet lingering on a near-horizon unstable orbit
# (h = (r-r_h)/C -> 0 there) is dropped at the cap. Such photons are deeply
# redshifted, so the spectral impact is negligible (cf. grmonty MAXNSTEP).
const _GRMHD_MAX_STEPS = 20_000

"""
    _grmhd_nu_alpha(pl, p) -> Float64

Invariant thermal-synchrotron absorption rate `nu * alpha_nu` [cm^-1 s^-1] at a
fluid sample `pl` (from `plasma_at`) for the covariant wavevector `p`. The
fluid-frame frequency is `nu = -(p . u) m_e c^2 / h`; the pitch cosine to B is
`cos = (p . b) / (|b| (-(p . u)))` (grmonty `get_bk_angle`). Zero for vacuum /
non-emitting / past-directed samples.
"""
@inline function _grmhd_nu_alpha(pl, p::SVector{4})
    pl.inside || return 0.0
    kdotu = dot(p, pl.ucon)                       # p_mu u^mu (< 0 for E > 0)
    kdotu >= 0.0 && return 0.0
    nu = -kdotu * ME * CL * CL / HPL
    b_sq = dot(pl.bcon, pl.gcov * pl.bcon)
    b_sq <= 0.0 && return 0.0
    cth = dot(p, pl.bcon) / (sqrt(b_sq) * (-kdotu))
    sth = sqrt(max(0.0, 1.0 - cth * cth))
    return nu * alpha_nu(pl.n_e, pl.theta_e, pl.B, nu, sth)
end

"""
    _grmhd_nu_alpha_scatt(table, pl, p) -> Float64

Invariant electron-scattering rate `nu * alpha_s = nu n_e sigma_h(eps, Theta_e)`
[cm^-1 s^-1] at fluid sample `pl` for wavevector `p` (hot-cross `table`). Zero for
vacuum / past-directed samples.
"""
@inline function _grmhd_nu_alpha_scatt(table::HotcrossTable, pl, p::SVector{4})
    pl.inside || return 0.0
    kdotu = dot(p, pl.ucon)
    kdotu >= 0.0 && return 0.0
    nu = -kdotu * ME * CL * CL / HPL
    return alpha_inv_scatt(table, nu, pl.theta_e, pl.n_e)
end

"""
    grmhd_bias(bias0, theta_e, nscatt) -> Float64

Importance bias for the exponential-transform scatter decision on the GRMHD
grid: `b = max(1, bias0 (n_sc + 1)^2)`. `bias0 > 1` boosts the rare scattering
rate so the Compton tail is sampled at modest `Ns`; the `(n_sc+1)^2` growth
keeps higher orders populated. Unbiased for any `b >= 1` (the child carries the
reweighted `w(1-e^{-dtau_s})/(1-e^{-b dtau_s})`, see `_mc_scatter_event`).
`theta_e` accepted for signature parity with the sphere `bias_func`.
"""
@inline grmhd_bias(bias0::Real, theta_e::Real, nscatt::Integer) =
    max(1.0, Float64(bias0) * (Float64(nscatt) + 1.0)^2)

# Escaped-photon polar angle [rad] in [0, pi] about the spin (z) axis, from the
# ASYMPTOTIC contravariant direction k^mu = g^{mu nu} p_nu at the escape point
# (KS -> flat there, so k^i is the travel direction). NOT folded (i = 17 and 163 deg
# are distinct viewing inclinations).
@inline function _grmhd_escape_theta(st, x::SVector{4}, p::SVector{4})
    kcon = inverse_metric(st, x) * p
    ksp = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    ksp <= 0.0 && return -1.0
    return acos(clamp(kcon[4] / ksp, -1.0, 1.0))
end

# Affine step so the COORDINATE displacement is ~ (r - r_h)/C: the physical
# wavevector magnitude is ~ h nu / m_e c^2 ~ 1e-11, so a bare affine step would
# freeze the photon; dividing by the spatial |k| (grmonty `stepsize`) makes the
# step a true geometric advance while keeping p physical (invariant optical
# depth and conserved E = -p_t stay exact).
@inline function _grmhd_dlam(st, x::SVector{4}, p::SVector{4},
                             r::Float64, r_h::Float64, C::Float64)
    kcon = inverse_metric(st, x) * p
    ks = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    return (r - r_h) / (C * (ks + 1.0e-300))
end

"""
    track_superphoton_grmhd(sph, model, C, R_record) -> (Superphoton, recorded::Bool)

Transport `sph` along the curved KS null geodesic (fixed-step RK4, affine step
from `_grmhd_dlam`), decaying its weight by the deterministic absorption depth
`dtau_a = 1/2[(nu alpha_nu)_n + (nu alpha_nu)_{n+1}] (L_unit h / m_e c^2) dl`
(grmonty `dtauK`), sampling the fluid each step. Returns the escaped packet
(`recorded = true`) on reaching `R_record` or streaming outbound past `Rem`;
`recorded = false` on horizon capture or the step cap. Absorption-only.
"""
function track_superphoton_grmhd(sph::Superphoton, model::GRMHDModel,
                                 C::Float64, R_record::Float64, Rem::Float64)
    st = model.spacetime
    r_h = model.r_h
    x = sph.x
    p = sph.p
    w = sph.w
    tau = sph.tau_abs
    y = SVector(x[1], x[2], x[3], x[4], p[1], p[2], p[3], p[4])
    Cgr = model.L_unit * HPL / (ME * CL * CL)
    pl = plasma_at(model, x)
    na0 = _grmhd_nu_alpha(pl, p)
    r_prev = ks_radius(st, x)
    recorded = false
    @inbounds for _ in 1:_GRMHD_MAX_STEPS
        r = ks_radius(st, SVector(y[1], y[2], y[3], y[4]))
        r < r_h + _GRMHD_HORIZON_EPS && break                     # captured
        if r > R_record || (r > Rem && r > r_prev)                # escaped
            recorded = true
            break
        end
        r_prev = r
        h = _grmhd_dlam(st, x, p, r, r_h, C)
        y = rk4_step_fixed(st, y, h)
        xn = SVector(y[1], y[2], y[3], y[4])
        pn = SVector(y[5], y[6], y[7], y[8])
        pln = plasma_at(model, xn)
        na1 = _grmhd_nu_alpha(pln, pn)
        dtau = 0.5 * (na0 + na1) * Cgr * h
        tau += dtau
        w *= exp(-dtau)
        x = xn; p = pn; na0 = na1
    end
    sp = Superphoton(x, p, w; tau_abs = tau, tau_scatt = sph.tau_scatt,
                     nscatt = sph.nscatt, x1i = sph.x1i, x2i = sph.x2i,
                     ne0 = sph.ne0, thetae0 = sph.thetae0, b0 = sph.b0)
    return sp, recorded
end

"""
    track_superphoton_grmhd_scatter!(ws, sph, model, table, C, R_record, bias0,
                                     max_scatt, rng)
        -> (escaped::Superphoton, recorded::Bool, rng::MCRng)

Curved-geodesic tracker with the biased exponential-transform Compton scatter
decision (grmonty `track_super_photon`): every step accumulates absorption and
scattering optical depth, depletes the incident beam by the physical
`exp(-(dtau_a + dtau_s))`, and on a trigger spawns a Compton-scattered child at
a sub-step-interpolated event point (new energy/direction from
`scatter_superphoton`, `nscatt+1`) onto `ws`. Escape/capture as in
`track_superphoton_grmhd`.
"""
function track_superphoton_grmhd_scatter!(ws::ScatterWorkspace, sph::Superphoton,
                                          model::GRMHDModel, table::HotcrossTable,
                                          C::Float64, R_record::Float64, Rem::Float64,
                                          bias0::Float64, max_scatt::Int, rng::MCRng)
    st = model.spacetime
    r_h = model.r_h
    x = sph.x
    p = sph.p
    w = sph.w
    tau_a = sph.tau_abs
    tau_s = sph.tau_scatt
    nscatt = sph.nscatt
    y = SVector(x[1], x[2], x[3], x[4], p[1], p[2], p[3], p[4])
    Cgr = model.L_unit * HPL / (ME * CL * CL)
    pl = plasma_at(model, x)
    na0 = _grmhd_nu_alpha(pl, p)
    ns0 = _grmhd_nu_alpha_scatt(table, pl, p)
    r_prev = ks_radius(st, x)
    recorded = false
    rng0 = rng                                    # this photon's stream identity
    nspawn = 0
    child_w = 0.0
    @inbounds for _ in 1:_GRMHD_MAX_STEPS
        r = ks_radius(st, SVector(y[1], y[2], y[3], y[4]))
        r < r_h + _GRMHD_HORIZON_EPS && break
        if r > R_record || (r > Rem && r > r_prev)
            recorded = true
            break
        end
        r_prev = r
        h = _grmhd_dlam(st, x, p, r, r_h, C)
        y2 = rk4_step_fixed(st, y, h)
        xn = SVector(y2[1], y2[2], y2[3], y2[4])
        pn = SVector(y2[5], y2[6], y2[7], y2[8])
        pln = plasma_at(model, xn)
        na1 = _grmhd_nu_alpha(pln, pn)
        ns1 = _grmhd_nu_alpha_scatt(table, pln, pn)
        dtau_a = 0.5 * (na0 + na1) * Cgr * h
        dtau_s = 0.5 * (ns0 + ns1) * Cgr * h

        # ---- biased scatter decision (shared exponential transform) ----------
        if dtau_s > 0.0 && pln.inside && Int(nscatt) < max_scatt
            b = grmhd_bias(bias0, pln.theta_e, nscatt)
            spawn, weight_scat, frac, rng =
                _mc_scatter_event(w, b, dtau_s, _MC_WEIGHT_MIN, rng)
            if spawn
                event = nspawn
                nspawn += 1
                yev = rk4_step_fixed(st, y, h * frac)             # event point in-step
                xev = SVector(yev[1], yev[2], yev[3], yev[4])
                pev = SVector(yev[5], yev[6], yev[7], yev[8])
                plev = plasma_at(model, xev)
                if plev.inside
                    p_new, _, rng = scatter_superphoton(pev, plev.ucon, plev.ucov,
                                                        plev.bcon, plev.theta_e,
                                                        plev.gcov, plev.gcon, rng)
                    w_child = weight_scat * exp(-dtau_a * frac)   # absorbed up to event
                    child = Superphoton(xev, p_new, w_child;
                                        tau_abs = tau_a, tau_scatt = tau_s,
                                        nscatt = nscatt + Int32(1),
                                        x1i = sph.x1i, x2i = sph.x2i,
                                        ne0 = sph.ne0, thetae0 = sph.thetae0, b0 = sph.b0)
                    _spawn_child!(ws, child, mc_child_rng(rng0, event, ws.layout))
                    child_w += w_child
                end
            end
        end

        # ---- physical depletion of the incident beam ------------------------
        w *= exp(-(dtau_a + dtau_s))
        tau_a += dtau_a
        tau_s += dtau_s
        y = y2; x = xn; p = pn; na0 = na1; ns0 = ns1
    end
    _ws_child_weight!(ws, child_w, sph.nscatt)
    sp = Superphoton(x, p, w; tau_abs = tau_a, tau_scatt = tau_s, nscatt = nscatt,
                     x1i = sph.x1i, x2i = sph.x2i,
                     ne0 = sph.ne0, thetae0 = sph.thetae0, b0 = sph.b0)
    return sp, recorded, rng
end

# ------------------------------------------------------------ result container

"""
    GRMHDSpectrumResult

Output of `mc_spectrum_grmhd`. `spectra[k]` is the single-row escaped `Spectrum` of
photons whose asymptotic escape polar angle falls in inclination band
`incl_deg[k] +- half_width_deg`; `dOmega[k]` is that band's solid angle
`2 pi (cos th_lo - cos th_hi)` (use `nuLnu_solid(spectra[k], dOmega[k])` for the
isotropic-equivalent nu L_nu). `total` is the angle-summed escaped spectrum.
Diagnostics: `emitted_power` (Sum w E m_e c^2 over generated seeds, erg/s),
`escaped_power` (over recorded photons), and the zone/photon counts.
"""
struct GRMHDSpectrumResult
    incl_deg::Vector{Float64}
    half_width_deg::Float64
    dOmega::Vector{Float64}
    spectra::Vector{Spectrum}
    total::Spectrum
    emitted_power::Float64
    escaped_power::Float64
    n_zones::Int
    n_gen::Int
    n_recorded::Int
    scattering::Bool
end

"""
    incl_nuLnu(res::GRMHDSpectrumResult) -> (nu, L)

Per-inclination isotropic-equivalent `nu L_nu` [erg s^-1]: `nu` the shared bin
centres, `L[k, :]` the spectrum for `incl_deg[k]` (via `nuLnu_solid`).
"""
function incl_nuLnu(res::GRMHDSpectrumResult)
    nu = nu_centers(res.total)
    L = zeros(Float64, length(res.spectra), length(nu))
    for k in eachindex(res.spectra)
        _, Lk, _ = nuLnu_solid(res.spectra[k], res.dOmega[k])
        @inbounds L[k, :] .= Lk
    end
    return nu, L
end

# ------------------------------------------------------------ driver

"""
    mc_spectrum_grmhd(model::GRMHDModel, Ns; scattering=true, nthreads, seed,
                      incl_deg=(17.0,50.0,90.0), half_width_deg=10.0,
                      gen_nu_min=1e8, gen_nu_max=1e16, spec_nu_min=1e8, spec_nu_max=1e24,
                      n_esamp, n_ebins=160, C=80.0, R_record=1.0e3, max_scatt=3,
                      bias0=1.0, stride=1, r_emit_max=Inf, max_zones=typemax(Int))
        -> GRMHDSpectrumResult

Run the GRMHD Monte Carlo: build the emitting-zone list and grid weight table,
distribute ~`Ns` seed superphotons over the zones (per-zone count ~ N_s,
stochastic rounding), track each to escape/capture (curved geodesic, absorption
always; Compton scattering when `scattering=true`), and bin the escaped photons
by asymptotic inclination x log-nu. Threaded with one private exact
accumulator set per chunk; each seed owns the Philox stream of its global id
and each child the stream of its genealogy (`mc_child_rng`), so the result is
bitwise independent of `nthreads`. `stride`/`max_zones` subsample the grid for fast
tests; `r_emit_max` caps the emitting region to the luminous torus. `bias0`
boosts the rare scattering rate (unbiased for any `bias0 >= 1`);
`tabulated_gen` opts into the tabulated frequency rejection (see `mc_spectrum`).
"""
function mc_spectrum_grmhd(model::GRMHDModel, Ns::Real;
                           scattering::Bool = true,
                           nthreads::Integer = Threads.nthreads(),
                           seed::Integer = 0x51EED5,
                           incl_deg = (17.0, 50.0, 90.0),
                           half_width_deg::Real = 10.0,
                           gen_nu_min::Real = 1.0e8, gen_nu_max::Real = 1.0e16,
                           spec_nu_min::Real = 1.0e8, spec_nu_max::Real = 1.0e24,
                           n_esamp::Integer = _MC_N_ESAMP_DEFAULT,
                           n_ebins::Integer = 160,
                           C::Real = 80.0, R_record::Real = 1.0e3,
                           max_scatt::Integer = 3, bias0::Real = 1.0,
                           stride::Integer = 1, max_zones::Integer = typemax(Int),
                           r_emit_max::Real = Inf,
                           table::Union{Nothing,HotcrossTable} = nothing,
                           tabulated_gen::Bool = false)
    zones = emission_zones(model; max_zones = max_zones, stride = stride,
                           r_emit_max = r_emit_max)
    nz = length(zones)
    nz == 0 && error("mc_spectrum_grmhd: no emitting zones (check cuts / snapshot)")

    band = log(gen_nu_max) - log(gen_nu_min)
    Ns_param = Float64(Ns) / band
    # Compute the per-(zone,node) pitch-angle emission integral once (threaded
    # over zones), then build both the weight table and the per-zone counts from
    # the cached table (see emission_tables.jl).
    jt = build_zone_emission_table(zones;
                                   n_esamp = n_esamp, nu_min = gen_nu_min, nu_max = gen_nu_max)
    wgt = build_weight_table(jt, Ns_param; nthreads = nthreads)
    Nsz_all, dnmax = zone_counts(jt, wgt; nthreads = nthreads)
    # opt-in tabulated generation: log the same cached per-(zone,node) integrals
    # for the accept numerator, with the per-zone envelope over that interpolant
    # replacing the exact-integrand dnmax (acceptance <= 1, emission_tables.jl).
    lint = nothing
    if tabulated_gen
        lint, dnmax = zone_int_jnu_tables(jt, wgt)
    end

    # per-zone target count (stochastic rounding, dedicated stream); serial over
    # zones so the single `rr` Philox stream is consumed in a fixed order and the
    # generated superphoton layout is reproducible.
    ngen = Vector{Int}(undef, nz)
    rr = MCRng(Int(seed), 0)
    @inbounds for z in 1:nz
        Nsz = Nsz_all[z]
        fl = floor(Int, Nsz)
        u, rr = mc_uniform(rr)
        ngen[z] = fl + (u < (Nsz - fl) ? 1 : 0)
    end
    offset = Vector{Int}(undef, nz + 1)
    offset[1] = 0
    @inbounds for z in 1:nz
        offset[z + 1] = offset[z] + ngen[z]
    end
    Ntot = offset[nz + 1]

    # inclination bands (radians) + solid angles
    cen = collect(Float64.(incl_deg))
    nb = length(cen)
    hw = deg2rad(Float64(half_width_deg))
    lo = [clamp(deg2rad(cen[b]) - hw, 0.0, π) for b in 1:nb]
    hi = [clamp(deg2rad(cen[b]) + hw, 0.0, π) for b in 1:nb]
    dOmega = [2.0 * π * (cos(lo[b]) - cos(hi[b])) for b in 1:nb]

    # Only the SCATTER path reads the hot-cross table; an absorption-only run
    # (scattering=false) would otherwise pay the full ~221x81 quadrature build for a
    # table that is never consulted, so hand the loop a trivial placeholder instead.
    tbl = table !== nothing ? table :
          scattering ? build_hotcross_table() : HotcrossTable(zeros(Float64, 1, 1))

    # emission-confinement radius: a photon streaming outbound past this never
    # re-enters emitting plasma, so its escape direction is frozen (early-out). Take
    # the outermost emitting zone (+20%) so the confinement is exact for THIS grid,
    # but at least the model's `R_em` to bound vacuum stepping.
    Rem = model.R_em
    @inbounds for z in 1:nz
        Rem = max(Rem, 1.2 * ks_radius(model.spacetime, zones[z].x))
    end

    nchunks = max(1, Int(nthreads))
    mkbands() = [ExactSpectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                               n_ebins = n_ebins, n_thbins = 1) for _ in 1:nb]
    mktotal() = ExactSpectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                              n_ebins = n_ebins, n_thbins = 1)
    bands_by_chunk = [mkbands() for _ in 1:nchunks]
    total_by_chunk = [mktotal() for _ in 1:nchunks]
    # exact per-chunk powers: 1 emitted, 2 escaped (Sum w E)
    pow_by_chunk = [ExactScalars(2) for _ in 1:nchunks]
    nrec = zeros(Int, nchunks)
    layout = MCStreamLayout(max(1, Ntot), _GRMHD_MAX_STEPS, scattering ? Int(max_scatt) : 0)

    _mc_grmhd_loop!(bands_by_chunk, total_by_chunk, pow_by_chunk, nrec,
                    model, zones, wgt, dnmax, lint, offset, Ntot, tbl,
                    Float64(C), Float64(R_record), Rem, Float64(bias0), Int(max_scatt),
                    scattering, Int(seed), lo, hi, layout)

    bands = mkbands()
    total = mktotal()
    pow = ExactScalars(2)
    for c in 1:nchunks
        merge_spectra!(total, total_by_chunk[c])
        for b in 1:nb
            merge_spectra!(bands[b], bands_by_chunk[c][b])
        end
        merge_scalars!(pow, pow_by_chunk[c])
    end
    mec2 = ME * CL * CL
    return GRMHDSpectrumResult(cen, Float64(half_width_deg), dOmega,
                               [Spectrum(b) for b in bands], Spectrum(total),
                               mec2 * xsum_round(pow, 1), mec2 * xsum_round(pow, 2), nz, Ntot,
                               sum(nrec), scattering)
end

# Record an escaped packet into the total + every matching inclination band.
@inline function _grmhd_record!(bands, total, sp::Superphoton, st, lo, hi)
    th = _grmhd_escape_theta(st, sp.x, sp.p)
    th < 0.0 && return
    record_energy!(total, sp)
    @inbounds for b in eachindex(bands)
        (th >= lo[b] && th <= hi[b]) && record_energy!(bands[b], sp)
    end
    return
end

# Per-zone generation table for the tabulated opt-in: `nothing` stays the exact
# path; the matrix hands out a column view on the weight table's log-nu grid.
@inline _zone_int_table(::Nothing, wgt::WeightTable, z::Int) = nothing
@inline _zone_int_table(lint::Matrix{Float64}, wgt::WeightTable, z::Int) =
    IntJnuTable(wgt.lnu_min, wgt.dlnu, view(lint, :, z))

"""
    _mc_grmhd_loop!(...) -> nothing

Threaded generate/track/record kernel over the global photon range `1:Ntot`,
split into contiguous equal-size sub-ranges (a single bright zone's photons are
spread across chunks). Each chunk owns private exact accumulators, a
`ScatterWorkspace` and its diagnostic slots — race-free regardless of task
migration. Photon `gid` maps to its zone by binary search on the cumulative
`offset` and is seeded `mc_primary_rng(seed, gid)`; children are keyed by
genealogy under `layout`. `lint` is the optional tabulated-generation matrix
(`nothing` = exact default).
"""
function _mc_grmhd_loop!(bands_by_chunk, total_by_chunk, pow_by_chunk, nrec,
                         model::GRMHDModel, zones, wgt::WeightTable,
                         dnmax::Vector{Float64}, lint::Union{Nothing,Matrix{Float64}},
                         offset::Vector{Int}, Ntot::Int,
                         table::HotcrossTable,
                         C::Float64, R_record::Float64, Rem::Float64, bias0::Float64,
                         max_scatt::Int, scattering::Bool, seed::Int, lo, hi,
                         layout::MCStreamLayout)
    st = model.spacetime
    nchunks = length(total_by_chunk)
    Threads.@threads for c in 1:nchunks
        bands = bands_by_chunk[c]
        total = total_by_chunk[c]
        pow = pow_by_chunk[c]
        ws = ScatterWorkspace(layout)
        nr = 0
        glo = div((c - 1) * Ntot, nchunks) + 1
        ghi = div(c * Ntot, nchunks)
        @inbounds for gid in glo:ghi
            z = searchsortedfirst(offset, gid) - 1        # offset[z] < gid <= offset[z+1]
            zone = zones[z]
            rng = mc_primary_rng(seed, gid)
            sp, rng = sample_zone_photon(zone, wgt, dnmax[z], rng,
                                         _zone_int_table(lint, wgt, z))
            xsum_add!(pow, 1, sp.w * sp.E)
            if scattering
                push_photon!(ws, sp, rng)
                while !isempty(ws.sp)
                    cur, crng = pop_photon!(ws)
                    e, recorded, crng = track_superphoton_grmhd_scatter!(
                        ws, cur, model, table, C, R_record, Rem, bias0, max_scatt, crng)
                    if recorded
                        xsum_add!(pow, 2, e.w * e.E)
                        nr += 1
                        _grmhd_record!(bands, total, e, st, lo, hi)
                    end
                end
            else
                e, recorded = track_superphoton_grmhd(sp, model, C, R_record, Rem)
                if recorded
                    xsum_add!(pow, 2, e.w * e.E)
                    nr += 1
                    _grmhd_record!(bands, total, e, st, lo, hi)
                end
            end
        end
        nrec[c] = nr
    end
    return nothing
end
