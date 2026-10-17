# Oracle: BLP.Transfer against BlackLightPlus.Radiation, bit for bit: the nonthermal fit
# constants, the synchrotron fits, the tetrads, the coefficient functions on decoded
# samples (five population mixes and cut sets, Float64 and Dual units, both tetrad
# normalisations), the segment operators, and both marches on a torus over Kerr (fp64,
# fp32 geodesic lane, Dual mass scale, two-set ladder), a superposed binary and stored
# trajectories; plus the hot-path contract of the marches.
#
# The script runs twice. In the calling process (native CPU target) every comparison is
# bitwise except those marked `contracted`: the polarized marches, the ladder lanes and
# the coefficient functions on Dual states. There LLVM's choice of which `a*b + c` pairs
# of the `muladd`-based StaticArrays products and dot products it fuses into FMA
# instructions depends on the surrounding code (the inlined decode, the strided reads of
# Dual fields), and the results agree to round-off. The script then reruns itself on
# `-C generic`, a CPU target without FMA, where nothing is contracted and every
# comparison, `contracted` ones included, must be bitwise: the operations and their
# order are the old ones.
Base.include(@__MODULE__, joinpath(@__DIR__, "helpers.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "decks.jl"))

using BLP.Transfer: StokesI, IQUV, PowerLaw, Kappa, TransferParams, output_type, coefficients,
                    segment, couple, stokes_admissible, fluid_tetrad_checked, normal_observer,
                    qu_rotation, rotate_qu, thermal_synchrotron, thermal_synchrotron_polarized,
                    power_law_synchrotron, power_law_synchrotron_polarized, kappa_synchrotron,
                    kappa_synchrotron_polarized, SpacetimeMarch, StoredPath, march
using BLP.Matter: GridProvider, PlasmaPrimitives, ElectronModel, PlasmaCuts, legacy_bound, decode,
                  pgas_from_eint, primitive_four_vectors
import BLP.Matter: primitives, domain_radius

const OS = BlackLightPlus.Simulation
const OI = BlackLightPlus.Inputs
const CC = BlackLightPlus.Constants
const bits = Decks.bits

"True in the child process on a CPU target without FMA."
const NOFMA = get(ENV, "BLP_ORACLE_NOFMA", "") == "1"

const DEV = scratch_module(:OldTransferDevice)
Core.eval(DEV, :(using BlackLightPlus.Simulation: units))
load_old!(DEV, "gpu_image.jl", [:_unit_fields])

@noinline callf(f::F, args...) where {F} = f(args...)
@noinline callk(f::F, args...; kw...) where {F} = f(args...; kw...)

struct OracleTag end
dual(x) = ForwardDiff.Dual{OracleTag}(x, 1.0)

# ---------------------------------------------------------------- parameter sets
const M_BH = 6.2e9 * 1.989e33
const L_U = CC.GNEWT * M_BH / CC.CL^2

plasma(; kw...) = OI.PlasmaConfig(; kw...)
const MIXES = (thermal = plasma(),
               powerlaw = plasma(thermal_frac = 0.6, power_frac = 0.4),
               kappa = plasma(thermal_frac = 0.5, kappa_frac = 0.5, kappa = 3.8, w = 0.6),
               all = plasma(thermal_frac = 0.34, power_frac = 0.33, kappa_frac = 0.33,
                            kappa = 4.6, w = 0.4, power_p = 2.8),
               cuts = plasma(thermal_frac = 0.7, power_frac = 0.3, sigma_cut = 0.12,
                             theta_e_min = 2.0, use_p = false))

runconfig(p; ms = 3.0e-18 * L_U^3, ν = 230.0e9, a = 0.9) =
    OI.RunConfig(bh_a = a, m_bh_cgs = M_BH, mass_scale_cgs = ms, frequency_hz = ν, plasma = p)

# The old prepared configuration and its march parameter set (as the fused drivers build it).
function old_params(cfg)
    u = OS.units(cfg.m_bh_cgs, cfg.mass_scale_cgs)
    scc = BlackLightPlus._prepared_scc(cfg, u)
    gp = merge(OR.coefficient_params(scc), (nu_obs = cfg.frequency_hz, L_unit = u.L_unit))
    return scc, gp, u
end
old_pol(gp, inc) = (i = deg2rad(inc); merge(gp, (cu1 = 0.0, cu2 = -cos(i), cu3 = 0.0, cu4 = sin(i))))

electrons(p) = ElectronModel(mu = p.mu, ne_ni = p.ne_ni, use_p = p.use_p, rat_low = p.rat_low,
                             rat_high = p.rat_high, gamma = p.gamma, gamma_i = p.gamma_i,
                             gamma_e = p.gamma_e, thermal_frac = p.thermal_frac)
cuts(p) = PlasmaCuts(sigma = legacy_bound(Float64, -1.0, p.sigma_cut),
                     theta_e = legacy_bound(Float64, p.theta_e_min, -1.0))
function new_params(cfg; mass_scale = cfg.mass_scale_cgs)
    p = cfg.plasma
    pl = p.power_frac != 0.0 ? PowerLaw(p.power_frac, p.power_p, p.gamma_min, p.gamma_max) : nothing
    kp = p.kappa_frac != 0.0 ? Kappa(p.kappa_frac, p.kappa, p.w) : nothing
    return TransferParams(NN.Units(cfg.m_bh_cgs, mass_scale), electrons(p), cuts(p),
                          cfg.frequency_hz; powerlaw = pl, kappa = kp)
end

# ---------------------------------------------------------------- torus providers
# The sampler closure of the old marches over the same torus as `Decks.TorusDeck`.
old_sampler(y) = SVector{8,Float64}(Decks.torus_primitives(Float64(y[2]), Float64(y[3]), Float64(y[4])))

"The torus with the internal energy the stored-path drivers recover from the gas pressure."
struct RoundTripTorus <: GridProvider
    gamma::Float64
end
function primitives(p::RoundTripTorus, x::SVector{4})
    t = Decks.torus_primitives(x[2], x[3], x[4])
    eint = pgas_from_eint(Float64(t[5]), p.gamma) / (p.gamma - 1)
    return PlasmaPrimitives{Float64}(Float64(t[1]), SVector{3,Float64}(t[2], t[3], t[4]), eint,
                                     SVector{3,Float64}(t[6], t[7], t[8]), t[1] > 0)
end
domain_radius(::RoundTripTorus) = 11.0
old_nt_sampler(γ) = x -> begin
    t = Decks.torus_primitives(x[2], x[3], x[4])
    (rho = Float64(t[1]), pgas = OS.pgas_from_eint(Float64(t[5]), γ),
     uu = SVector{3,Float64}(t[2], t[3], t[4]), bb = SVector{3,Float64}(t[6], t[7], t[8]),
     inside = t[1] > 0)
end

const Prog = Decks.MarchProgram

# ---------------------------------------------------------------- random samples
function random_sample(rng, sto)
    while true
        x = SVector(0.0, randpoint(rng, 1.6, 30.0)...)
        OG.ks_radius(sto, x) > 1.5 || continue
        gcov, gcon = OG.metric_and_inverse(sto, x)
        k, ok = try
            OG.null_covariant(sto, x, randn(rng, SVector{3,Float64}))
        catch err
            err isa DomainError || rethrow()
            (zero(SVector{4,Float64}), false)
        end
        ok || continue
        kcov = k / (-k[1]) .* (1.0 .+ 1e-9 .* randn(rng, SVector{4,Float64}))
        kcon = gcon * kcov
        rho = exp(1.5 * randn(rng))
        eint = rand(rng) < 0.03 ? -0.1 * rand(rng) : rho * exp(randn(rng) - 2.0)
        uu = 0.6 * randn(rng, SVector{3,Float64})
        bb = rand(rng) < 0.03 ? zero(SVector{3,Float64}) :
             sqrt(rho) * exp(randn(rng) - 1.0) * randn(rng, SVector{3,Float64})
        return (; x, gcov, gcon, kcov, kcon, rho, eint, uu, bb)
    end
end

# Values and first partials of a number, vector or matrix, as Float64 vectors.
vals(c) = [Float64(ForwardDiff.value(v)) for v in c]
pars(c) = [v isa ForwardDiff.Dual ? Float64(ForwardDiff.partials(v, 1)) : 0.0 for v in c]
near(a::Tuple, b::Tuple) = length(a) == length(b) && all(near(x, y) for (x, y) in zip(a, b))
near(a, b) = isapprox(vals(a), vals(b); rtol = COEFF_RTOL) && isapprox(pars(a), pars(b); rtol = COEFF_RTOL)

"Bitwise equality of two results, or, for a `contracted` comparison natively, `near`."
agree(a, b; contracted = false) =
    (NOFMA || !contracted) ? bits(a) == bits(b) && typeof(a) === typeof(b) : near(a, b)

# The coefficient functions on decoded samples. Both sides see the same decoded
# kinematics: the old functions take the four-vectors of the Float64 decode, which carry
# no unit dependence; the four-vectors themselves are checked against the old ones.
function check_coefficients(rng, cfg, n; dual_units = false)
    scc, gp, u = old_params(cfg)
    a = cfg.bh_a
    sto = OG.KerrSchildCartesian(a); stn = NS.KerrSchildCartesian(a)
    p64 = new_params(cfg)
    if dual_units
        old = (merge(gp, DEV._unit_fields(cfg.m_bh_cgs, dual(cfg.mass_scale_cgs))),)
        pn = new_params(cfg; mass_scale = dual(cfg.mass_scale_cgs))
    else
        old = (scc, gp)
        pn = p64
    end
    ν = cfg.frequency_hz
    nbad = 0; nvalid = 0; nfv = 0
    for _ in 1:n
        s = random_sample(rng, sto)
        g = NS.metric_and_inverse(stn, s.x)
        prim = PlasmaPrimitives{Float64}(s.rho, s.uu, s.eint, s.bb, true)
        state = callf(decode, prim, g, pn.units, pn.electrons, pn.cuts)
        s64 = callf(decode, prim, g, p64.units, p64.electrons, p64.cuts)
        nvalid += state.valid
        pgas = OS.pgas_from_eint(s.eint, cfg.plasma.gamma)
        fvo = callf(OS.primitive_four_vectors, s.uu..., s.bb..., s.gcov, s.gcon)
        fvn = callf(primitive_four_vectors, s.uu, s.bb, g...)
        nfv += bits(fvn) != bits(fvo)
        fv = s64.valid ? (ucon = s64.ucon, ucov = s64.ucov, bcon = s64.bcon, bcov = s64.bcov, b_sq = s64.b_sq) : fvo
        new_i = callf(coefficients, StokesI(), pn, state, s.kcov, ν)
        for c in old
            old_i = callf(OR.unpolarized_coefficients, c, s.rho, pgas, fv, s.kcov, ν)
            nbad += !agree(new_i, old_i; contracted = dual_units)
            for renorm in (false, true)
                new_p = callf(coefficients, IQUV(), pn, state, s.kcov, s.kcon, g, ν, renorm)
                old_p = callf(OR.eight_coefficients, c, s.rho, pgas, fv, s.gcov, s.gcon, s.kcov,
                              s.kcon, ν, renorm)
                nbad += !agree(new_p, old_p; contracted = dual_units)
            end
        end
    end
    return (; nbad, nvalid, nfv)
end

# ---------------------------------------------------------------- segments
function random_coefficients(rng)
    kind = rand(rng, 1:8)
    j = SVector(exp(randn(rng)), 0.5randn(rng), 0.0, 0.3randn(rng))
    kind == 1 && (j = SVector(j[1], 2.0 * j[1], 0.0, j[1]))          # |j_P| > j_I
    α = SVector(exp(randn(rng)), 0.4randn(rng), 0.0, 0.2randn(rng))
    ρ = SVector(0.0, randn(rng), 0.0, randn(rng))
    kind == 2 && (α = zero(α); ρ = zero(ρ))                            # emission only
    kind == 3 && (α = SVector(α[1], 0.0, 0.0, 0.0); ρ = zero(ρ))       # total absorption
    kind == 4 && (α = zero(α))                                         # pure rotation
    kind == 5 && (α = SVector(α[1], 1.5α[1], 0.0, 0.0))                # α_P > α_I
    kind == 6 && (ρ = SVector(0.0, α[2], 0.0, α[4]))                   # degenerate
    kind == 7 && (ρ = zero(ρ))                                         # ρ_P = 0
    ds = exp(4.0 * randn(rng))                                         # thin and saturated
    return j, α, ρ, ds
end

function check_segments(rng, n)
    nbad = 0
    for _ in 1:n
        j, α, ρ, ds = random_coefficients(rng)
        S = SVector(exp(randn(rng)), randn(rng), randn(rng), randn(rng))
        ok = bits(callf(segment, IQUV(), j, α, ρ, ds)) == bits(callf(OR.stokes_segment, j, α, ρ, ds)) &&
             bits(callk(segment, IQUV(), j, α, ρ, ds; dtau_max = 3.0)) ==
             bits(callk(OR.stokes_segment, j, α, ρ, ds; dtau_max = 3.0)) &&
             bits(callf(couple, IQUV(), S, j, α, ρ, ds)) == bits(callf(OR.stokes_couple, S, j, α, ρ, ds)) &&
             bits(callf(stokes_admissible, S)) == bits(callf(OR.stokes_admissible, S))
        ji, ai = j[1] * (rand(rng) < 0.1 ? 0.0 : 1.0), α[1] * (rand(rng) < 0.2 ? -1.0 : 1.0)
        ok &= bits(callf(segment, StokesI(), ji, ai, ds)) == bits(callf(OR.unpolarized_segment, ji, ai, ds)) &&
              bits(callk(segment, StokesI(), ji, ai, ds; dtau_max = 3.0)) ==
              bits(callk(OR.unpolarized_segment, ji, ai, ds; dtau_max = 3.0)) &&
              bits(callf(couple, StokesI(), S[1], ji, ai, ds)) ==
              bits(callf(OR.couple_unpolarized, S[1], ji, ai, ds))
        jd, ad = dual(ji), ai * dual(1.0)
        ok &= bits(callf(segment, StokesI(), jd, ad, ds)) == bits(callf(OR.unpolarized_segment, jd, ad, ds))
        nbad += !ok
    end
    return nbad
end

# ---------------------------------------------------------------- marches
const A_SPIN = 0.9
const C_STEP = 80.0
const DIST = 1000.0
const ESCAPE = 1.1 * DIST
const R_EM = 12.0
const NMAX = 2500
const CAM = NS.PlaneParallelCamera(inc_deg = 60.0, d = DIST, fov = 24.0, npix = 32)
# Natively contracted comparisons: march outputs within this fraction of the largest old
# intensity of the ray set (Faraday rotation amplifies the polarized round-off)
const MARCH_RTOL = 1e-7
# and coefficient outputs within this relative norm distance, values and partials apart
const COEFF_RTOL = 1e-9

pick(rays, n) = rays[round.(Int, range(1, length(rays); length = n))]

"""
Rays whose new and old results differ: in bits, or, for a `contracted` comparison in the
native process, by more than `MARCH_RTOL` of the largest old intensity.
"""
function compare_rays(newf, oldf, rays; contracted = false)
    outs = [(newf(y0), oldf(y0)) for y0 in rays]
    (NOFMA || !contracted) && return count(o -> bits(o[1]) != bits(o[2]), outs)
    scale = maximum(o -> abs(first(o[2])), outs)
    return count(o -> maximum(abs.(o[1] .- o[2])) > MARCH_RTOL * scale, outs)
end

# Bytes allocated by one march, after a warm-up call compiles it.
allocs(mode, y0, prog) = (march(mode, y0, prog); @allocated march(mode, y0, prog))

@testset "oracle: Transfer$(NOFMA ? " (no FMA)" : "")" begin
    rng = Xoshiro(20261016)

    @testset "fit constants" begin
        nbad = 0
        for _ in 1:300
            p = rand(rng) < 0.1 ? 2.0 : 1.5 + 3.0 * rand(rng)
            gmin = exp(4.0 * rand(rng)); gmax = gmin * exp(5.0 + 10.0 * rand(rng))
            c = OR.SimCoeffConfig(plasma_p = p, plasma_power_frac = 0.3)
            OR.precompute_power_law!(c, gmin, gmax)
            n = PowerLaw(0.3, p, gmin, gmax)
            nbad += bits((n.jj, n.aa, n.jj_q, n.jj_v, n.aa_q, n.aa_v, n.rho, n.rho_q, n.rho_v, n.gamma_min)) !=
                    bits((c.power_jj, c.power_aa, c.power_jj_q, c.power_jj_v, c.power_aa_q, c.power_aa_v,
                          c.power_rho, c.power_rho_q, c.power_rho_v, c.plasma_gamma_min))
            k = 3.5 + 1.5 * rand(rng); w = 0.2 + 1.5 * rand(rng)
            c = OR.SimCoeffConfig(plasma_kappa = k, plasma_w = w, plasma_kappa_frac = 0.3)
            OR.precompute_kappa!(c)
            n = Kappa(0.3, k, w)
            nbad += bits(Tuple(getfield(n, f) for f in fieldnames(Kappa)[4:end])) !=
                    bits(Tuple(getfield(c, Symbol(:kappa_, f)) for f in fieldnames(Kappa)[4:end]))
        end
        @test nbad == 0
    end

    @testset "synchrotron fits" begin
        cfg = runconfig(MIXES.all)
        scc, _, _ = old_params(cfg)
        pn = new_params(cfg)
        nbad = 0
        for _ in 1:3000
            n_e = exp(10.0 + 3.0 * randn(rng)); te = exp(2.0 * randn(rng)); bb = exp(3.0 + 2.0 * randn(rng))
            nu = exp(26.0 + 2.0 * randn(rng)); st = rand(rng) * (rand(rng) < 0.05 ? -1.0 : 1.0)
            ct = (rand(rng) < 0.5 ? -1.0 : 1.0) * sqrt(max(0.0, 1.0 - st^2))
            tf = rand(rng) < 0.05 ? 0.0 : rand(rng)
            nbad += bits(callf(thermal_synchrotron, tf, n_e, te, bb, nu, st)) !=
                    bits(callf(OR.thermal_synchrotron, tf, n_e, te, bb, nu, st))
            nbad += bits(callf(thermal_synchrotron_polarized, tf, n_e, te, bb, nu, st, ct)) !=
                    bits(callf(OR.thermal_synchrotron_polarized, tf, n_e, te, bb, nu, st, ct))
            nbad += bits(callf(power_law_synchrotron, pn.powerlaw, n_e, bb, nu, st)) !=
                    bits(callf(OR.power_law_synchrotron, scc, n_e, bb, nu, st))
            nbad += bits(callf(power_law_synchrotron_polarized, pn.powerlaw, n_e, bb, nu, st, ct)) !=
                    bits(callf(OR.power_law_synchrotron_polarized, scc, n_e, bb, nu, st, ct))
            nbad += bits(callf(kappa_synchrotron, pn.kappa, n_e, bb, nu, st)) !=
                    bits(callf(OR.kappa_synchrotron, scc, n_e, bb, nu, st))
            nbad += bits(callf(kappa_synchrotron_polarized, pn.kappa, n_e, bb, nu, st, ct)) !=
                    bits(callf(OR.kappa_synchrotron_polarized, scc, n_e, bb, nu, st, ct))
            tdual = dual(te)
            nbad += bits(callf(thermal_synchrotron, tf, n_e, tdual, bb, nu, st)) !=
                    bits(callf(OR.thermal_synchrotron, tf, n_e, tdual, bb, nu, st))
        end
        @test nbad == 0
    end

    @testset "tetrads" begin
        sto = OG.KerrSchildCartesian(A_SPIN)
        nbad = 0
        for _ in 1:3000
            s = random_sample(rng, sto)
            fv = OS.primitive_four_vectors(s.uu..., s.bb..., s.gcov, s.gcon)
            up = rand(rng) < 0.05 ? fv.ucon + 0.3 * s.kcon : fv.bcon     # up in span(u, k)
            for r in (false, true)
                nbad += bits(callk(fluid_tetrad_checked, fv.ucon, fv.ucov, s.kcon, s.kcov, up, s.gcov,
                                   s.gcon; renorm_e1 = r)) !=
                        bits(callk(OR.fluid_tetrad_checked, fv.ucon, fv.ucov, s.kcon, s.kcov, up, s.gcov,
                                   s.gcon; renorm_e1 = r))
            end
            nbad += bits(callf(normal_observer, s.gcon)) != bits(callf(OR._normal_observer, s.gcon))
            t1 = first(OR.fluid_tetrad_checked(fv.ucon, fv.ucov, s.kcon, s.kcov, fv.bcon, s.gcov, s.gcon))
            n0 = OR._normal_observer(s.gcon)
            t2 = first(OR.fluid_tetrad_checked(n0, s.gcov * n0, s.kcon, s.kcov, randn(rng, SVector{4,Float64}),
                                               s.gcov, s.gcon))
            nbad += bits(callf(qu_rotation, t1, t2, s.gcov)) != bits(callf(OR._qu_rotation, t1, t2, s.gcov))
            c, sn = OR._qu_rotation(t1, t2, s.gcov)
            M = randn(rng, SMatrix{4,4,Float64}); b = randn(rng, SVector{4,Float64})
            nbad += bits(callf(rotate_qu, c, sn, M)) != bits(callf(OR._rotate_qu, c, sn, M))
            nbad += bits(callf(rotate_qu, c, sn, b)) != bits(callf(OR._rotate_qu, c, sn, b))
        end
        @test nbad == 0
    end

    @testset "coefficients: $name" for name in keys(MIXES)
        cfg = runconfig(MIXES[name])
        r = check_coefficients(rng, cfg, 5000)
        @test r.nbad == 0
        @test r.nfv == 0
        @test r.nvalid > (name === :cuts ? 300 : 2500)
        @test check_coefficients(rng, cfg, 5000; dual_units = true).nbad == 0
    end

    @testset "segment operators" begin
        @test check_segments(rng, 20000) == 0
    end

    rays = pick(Decks.camera_rays(NS.KerrSchildCartesian(A_SPIN), CAM), 200)
    frame = NS.CameraFrame(CAM)
    r_h = 1 + sqrt(1 - A_SPIN * A_SPIN)
    budget = NS.MarchBudget(C_STEP, ESCAPE, R_EM, NMAX)
    stk = NS.KerrSchildCartesian(A_SPIN)

    @testset "Kerr marches: $name" for name in (:thermal, :all, :cuts)
        cfg = runconfig(MIXES[name])
        _, gp, _ = old_params(cfg)
        gpp = old_pol(gp, CAM.inc_deg)
        pn = new_params(cfg)
        prog = Prog(stk, SpacetimeMarch(stk, budget), Decks.TorusDeck(), pn, frame)
        km = OR.KerrMarch(A_SPIN, r_h, C_STEP, ESCAPE, R_EM)
        @test output_type(StokesI(), pn) === Float64 && output_type(IQUV(), pn) === SVector{4,Float64}
        @test compare_rays(y -> callf(march, StokesI(), y, prog),
                           y -> callf(OR.reverse_unpolarized_march, y, old_sampler, Float64, gp, km, NMAX),
                           rays) == 0
        @test compare_rays(y -> callf(march, IQUV(), y, prog),
                           y -> callf(OR.reverse_polarized_march, y, old_sampler, gpp, km, NMAX),
                           rays; contracted = true) == 0
        iq = [march(IQUV(), y, prog) for y in rays]
        @test count(s -> s[1] > 0, iq) > 100 && count(s -> abs(s[2]) + abs(s[3]) > 0, iq) > 50

        # fp32 geodesic lane, fp64 transfer
        rays32 = [SVector{8,Float32}(y) for y in rays]
        stk32 = NS.lane(stk, Float32)
        prog32 = Prog(stk, SpacetimeMarch(stk32, NS.lane(budget, Float32)), Decks.TorusDeck(), pn, frame)
        km32 = OR.KerrMarch(Float32(A_SPIN), Float32(r_h), Float32(C_STEP), Float32(ESCAPE), Float32(R_EM))
        @test compare_rays(y -> callf(march, StokesI(), y, prog32),
                           y -> callf(OR.reverse_unpolarized_march, y, old_sampler, Float64, gp, km32, NMAX),
                           rays32) == 0
        @test compare_rays(y -> callf(march, IQUV(), y, prog32),
                           y -> callf(OR.reverse_polarized_march, y, old_sampler, gpp, km32, NMAX),
                           rays32; contracted = true) == 0
    end

    @testset "Dual mass scale and the mass-scale ladder" begin
        cfg = runconfig(MIXES.powerlaw)
        _, gp, _ = old_params(cfg)
        km = OR.KerrMarch(A_SPIN, r_h, C_STEP, ESCAPE, R_EM)
        ms = cfg.mass_scale_cgs
        pd = new_params(cfg; mass_scale = dual(ms))
        gpd = merge(gp, DEV._unit_fields(cfg.m_bh_cgs, dual(ms)))
        DT = promote_type(typeof(gpd.d_unit), typeof(gpd.e_unit), typeof(gpd.b_unit),
                          typeof(gpd.plasma_rat_high), typeof(gpd.plasma_rat_low))
        @test output_type(StokesI(), pd) === DT
        @test bits((pd.units.dens_unit, pd.units.e_unit, pd.units.B_unit)) ==
              bits((gpd.d_unit, gpd.e_unit, gpd.b_unit))
        progd = Prog(stk, SpacetimeMarch(stk, budget), Decks.TorusDeck(), pd, frame)
        @test compare_rays(y -> callf(march, StokesI(), y, progd),
                           y -> callf(OR.reverse_unpolarized_march, y, old_sampler, DT, gpd, km, NMAX),
                           rays) == 0
        out = [march(StokesI(), y, progd) for y in rays]
        @test count(d -> ForwardDiff.partials(d, 1) != 0, out) > 100

        scales = (ms, 2.5 * ms)
        pl = ntuple(k -> new_params(cfg; mass_scale = scales[k]), 2)
        gpl = ntuple(k -> merge(gp, DEV._unit_fields(cfg.m_bh_cgs, scales[k])), 2)
        progl = Prog(stk, SpacetimeMarch(stk, budget), Decks.TorusDeck(), pl, frame)
        @test output_type(StokesI(), pl) === SVector{2,Float64}
        @test compare_rays(y -> callf(march, StokesI(), y, progl),
                           y -> callf(OR.reverse_unpolarized_march, y, old_sampler, SVector{2,Float64},
                                      gpl, km, NMAX),
                           rays) == 0
        # each lane is the march of its set alone
        prog2 = Prog(stk, SpacetimeMarch(stk, budget), Decks.TorusDeck(), pl[2], frame)
        @test compare_rays(y -> callf(march, StokesI(), y, progl)[2], y -> callf(march, StokesI(), y, prog2),
                           rays; contracted = true) == 0
    end

    @testset "superposed binary" begin
        m, s, c = [0.5, 0.5], [0.3, -0.2], [(4.0, 0.0, 0.0), (-4.0, 0.0, 0.0)]
        stn = NS.SuperposedKerrSchild(m, s, c); sto = OG.SuperposedKerrSchild(m, s, c)
        raysb = pick(Decks.camera_rays(stn, CAM), 200)
        cfg = runconfig(MIXES.kappa)
        _, gp, _ = old_params(cfg)
        gpp = old_pol(gp, CAM.inc_deg)
        prog = Prog(stn, SpacetimeMarch(stn, budget), Decks.TorusDeck(), new_params(cfg), frame)
        sm = OR.SpacetimeMarch(sto, C_STEP, ESCAPE, R_EM)
        @test compare_rays(y -> callf(march, StokesI(), y, prog),
                           y -> callf(OR.reverse_unpolarized_march, y, old_sampler, Float64, gp, sm, NMAX),
                           raysb) == 0
        @test compare_rays(y -> callf(march, IQUV(), y, prog),
                           y -> callf(OR.reverse_polarized_march, y, old_sampler, gpp, sm, NMAX),
                           raysb; contracted = true) == 0
        @test sum(march(StokesI(), y, prog) for y in raysb) > 0
    end

    @testset "stored paths" begin
        cfg = runconfig(MIXES.all)
        scc, _, u = old_params(cfg)
        pn = new_params(cfg)
        sto = OG.KerrSchildCartesian(A_SPIN)
        states, dls = Decks.sampled_trace(stk, rays; C = C_STEP, escape = ESCAPE, nmax = NMAX, R_em = R_EM)
        γ = cfg.plasma.gamma
        idx = collect(eachindex(rays))
        prog(i) = Prog(stk, StoredPath(stk, states[i], dls[i]), RoundTripTorus(γ), pn, frame)
        @test compare_rays(i -> callf(march, StokesI(), states[i][end], prog(i)),
                           i -> callf(OR.unpolarized_march, sto, scc, old_nt_sampler(γ), states[i], dls[i],
                                      cfg.frequency_hz, u.L_unit),
                           idx) == 0
        @test compare_rays(i -> callf(march, IQUV(), states[i][end], prog(i)),
                           i -> callk(OR.polarized_march, sto, scc, old_nt_sampler(γ), states[i], dls[i],
                                      cfg.frequency_hz, u.L_unit; cam_up = frame.up),
                           idx; contracted = true) == 0
        @test maximum(length, states) > 100
    end

    @testset "hot path" begin
        cfg = runconfig(MIXES.all)
        pn = new_params(cfg)
        y0 = rays[100]; y32 = SVector{8,Float32}(y0)
        prog = Prog(stk, SpacetimeMarch(stk, budget), Decks.TorusDeck(), pn, frame)
        prog32 = Prog(stk, SpacetimeMarch(NS.lane(stk, Float32), NS.lane(budget, Float32)),
                      Decks.TorusDeck(), pn, frame)
        for (y, pr) in ((y0, prog), (y32, prog32)), mode in (StokesI(), IQUV())
            @test allocs(mode, y, pr) == 0
        end
        @test isbits(prog) && isbits(prog32)
        @test Decks.device_isbits(prog) && Decks.device_isbits(prog32)
    end

    if !NOFMA
        @testset "every comparison bitwise without FMA" begin
            cmd = `$(Base.julia_cmd()) -C generic -t 8 --project=$(Base.active_project())
                   -e "using Test; include($(repr(@__FILE__)))"`
            @test success(pipeline(addenv(cmd, "BLP_ORACLE_NOFMA" => "1"); stdout, stderr))
        end
    end
end
