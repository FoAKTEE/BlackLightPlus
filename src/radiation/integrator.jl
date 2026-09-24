# Radiation pipeline: per-ray fluid sampling -> invariant coefficients -> transfer
# march -> Stokes images. Polarized mode transports the coherency tensor N^{μν}
# (blacklight polarized_transfer.cpp, IntegratePolarizedRadiation), Strang-split:
# half-step parallel transport with midpoint-averaged connection and k^μ, analytic
# plasma coupling in the fluid tetrad (stokes_couple), then a quarter-factor
# predictor transport. Samples run SOURCE -> CAMERA; the camera readout rotates N into the
# camera tetrad and scales invariant -> physical by ν_obs³. Ray states carry p = -k
# (camera.jl), so every sample reads the photon wave covector as k_μ = -p_μ; k_t = -1
# at the camera ⇒ delta_lambda_cgs = dλ_code · L_unit / ν_obs; nonthermal electrons
# (power-law + kappa) enter additively in the 8 coefficients.

"""
    eight_coefficients(cfg, rho, pgas, fv, gcov, gcon, kcov, kcon, ν_obs, renorm_e1=false)
        -> (j::SVector{4}, α::SVector{4}, ρ::SVector{4}, tet, valid::Bool)

Invariant polarized coefficients at one sample: thermal + power-law + kappa
additive mixing (blacklight CalculatePolarizedCoefficients). `cfg` is a
`SimCoeffConfig` or its isbits `coefficient_params` set (the device kernels call this
same function). `kcov`/`kcon` are the
photon wave vector (future-directed, source → camera; `-p` of a ray state), so the
fluid-frame frequency is ν = -k·u ν_obs and cos θ_B is signed along the propagation. `rho`/`pgas` are
code-unit scalars; `fv` is the NamedTuple from `primitive_four_vectors`. Pass
`renorm_e1 = true` off single CKS so the returned fluid tetrad's e_(1) (hence the
Q/U frame) is unit.
"""
@inline function eight_coefficients(cfg, rho, pgas,
                                    fv, gcov, gcon, kcov::SVector{4}, kcon::SVector{4},
                                    ν_obs, renorm_e1::Bool = false)
    (rho <= 0.0 || pgas <= 0.0) && return _ZERO4S, _ZERO4S, _ZERO4S, _ZERO4x4, false
    b_sq = fv.b_sq
    b_sq <= 0.0 && return _ZERO4S, _ZERO4S, _ZERO4S, _ZERO4x4, false   # b² < 0 is round-off

    pd = plasma_decode(cfg, rho, pgas, b_sq)
    pd.omit && return _ZERO4S, _ZERO4S, _ZERO4S, _ZERO4x4, false
    n_e_cgs = pd.n_e_cgs; bb_cgs = pd.bb_cgs; theta_e = pd.theta_e

    # Orthonormal-frame pitch angle with SIGNED cos(theta_B). The fluid tetrad is
    # returned so polarized_march can reuse it for the plasma coupling instead of
    # rebuilding it per sample. renorm_e1 = true off single CKS (|det g| != 1):
    # without it e_(1) carries norm 1/|det g| and the Q/U Stokes frame is wrong;
    # the pitch angle is unaffected (b lies in span{e0,e2,e3}).
    tet = fluid_tetrad(fv.ucon, fv.ucov, kcon, kcov, fv.bcon, gcov, gcon;
                       renorm_e1 = renorm_e1)
    kt1 = kt2 = kt3 = 0.0
    bt1 = bt2 = bt3 = 0.0
    @inbounds for μ in 1:4
        kt1 += tet[2, μ] * kcov[μ]; kt2 += tet[3, μ] * kcov[μ]; kt3 += tet[4, μ] * kcov[μ]
        bt1 += tet[2, μ] * fv.bcov[μ]; bt2 += tet[3, μ] * fv.bcov[μ]; bt3 += tet[4, μ] * fv.bcov[μ]
    end
    k2t = kt1^2 + kt2^2 + kt3^2
    b2t = bt1^2 + bt2^2 + bt3^2
    kbt = kt1 * bt1 + kt2 * bt2 + kt3 * bt3
    cos2 = min(kbt * kbt / (k2t * b2t), 1.0)
    sin_th = sqrt(1.0 - cos2)
    cos_th = sqrt(cos2) * (kbt >= 0.0 ? 1.0 : -1.0)

    minus_ku = 0.0
    @inbounds for μ in 1:4
        minus_ku -= kcov[μ] * fv.ucon[μ]
    end
    nu_cgs = minus_ku * ν_obs                    # momentum_factor = 1 (k_t = -1)

    ji, jq, jv, ai, aq, av, rq, rv =
        thermal_synchrotron_polarized(cfg.plasma_thermal_frac, n_e_cgs, theta_e,
                                      bb_cgs, nu_cgs, sin_th, cos_th)
    if has_power_law(cfg)
        t = power_law_synchrotron_polarized(cfg, n_e_cgs, bb_cgs, nu_cgs, sin_th, cos_th)
        ji += t[1]; jq += t[2]; jv += t[3]; ai += t[4]; aq += t[5]; av += t[6]
        rq += t[7]; rv += t[8]
    end
    if has_kappa(cfg)
        t = kappa_synchrotron_polarized(cfg, n_e_cgs, bb_cgs, nu_cgs, sin_th, cos_th)
        ji += t[1]; jq += t[2]; jv += t[3]; ai += t[4]; aq += t[5]; av += t[6]
        rq += t[7]; rv += t[8]
    end
    return SVector(ji, jq, 0.0, jv), SVector(ai, aq, 0.0, av), SVector(0.0, rq, 0.0, rv), tet, true
end

const _ZERO4S = SVector(0.0, 0.0, 0.0, 0.0)
const _ZERO4x4 = zero(SMatrix{4,4,Float64})

"""
    unpolarized_coefficients(cfg, rho, pgas, fv, kcov, ν_obs) -> (j, α)

Invariant Stokes-I emissivity and absorptivity at one sample: the plasma decode and
cuts of `eight_coefficients`, the pitch angle from the invariant identity
cos²θ_B = (k·b)² / ((k·u)² b²) (b·u = 0, k null: the fluid-frame spatial momentum has
norm -k·u; the traced k's null-constraint drift enters at relative order
k·k/(k·u)², below 1e-7 on the C = 80 march), and the thermal + power-law + kappa
unpolarized fits. No tetrad is built. A vacuum, unmagnetized or cut sample returns zeros, which the
segment operator treats as a no-op. `cfg` is a `SimCoeffConfig` (AD `Dual` configs
included) or its `coefficient_params` set; the zeros take the unit/plasma type so a
`Dual` accumulation is type-stable.
"""
@inline function unpolarized_coefficients(cfg, rho, pgas, fv, kcov::SVector{4}, ν_obs)
    z = zero(promote_type(typeof(cfg.d_unit), typeof(cfg.e_unit), typeof(cfg.b_unit),
                          typeof(cfg.plasma_rat_high), typeof(cfg.plasma_rat_low)))
    (rho <= 0.0 || pgas <= 0.0) && return (z, z)
    b_sq = fv.b_sq
    b_sq <= 0.0 && return (z, z)                        # b² < 0 is round-off
    pd = plasma_decode(cfg, rho, pgas, b_sq)
    pd.omit && return (z, z)
    minus_ku = -dot(kcov, fv.ucon)                      # = ν_local / ν_obs (k_t = -1)
    kb = dot(kcov, fv.bcon)
    cos2 = min(kb * kb / (minus_ku * minus_ku * b_sq), 1.0)
    sin_th = sqrt(1.0 - cos2)
    nu = minus_ku * ν_obs
    j, a = thermal_synchrotron(cfg.plasma_thermal_frac, pd.n_e_cgs, pd.theta_e, pd.bb_cgs,
                               nu, sin_th)
    if has_power_law(cfg)
        jp, ap = power_law_synchrotron(cfg, pd.n_e_cgs, pd.bb_cgs, nu, sin_th)
        j += jp; a += ap
    end
    if has_kappa(cfg)
        jk, ak = kappa_synchrotron(cfg, pd.n_e_cgs, pd.bb_cgs, nu, sin_th)
        j += jk; a += ak
    end
    return (j, a)
end

# Coordinate-frame N^{αβ} -> covariant tetrad-frame N_{ab}. Explicit left-to-right
# sums (no matmul/fma reassociation) keep the accumulation order fixed.
#   tb[ν,α] = Σ_β gcov[ν,β] nn[α,β];  tc[μ,ν] = Σ_α gcov[μ,α] tb[ν,α]
#   td[b,μ] = Σ_ν tet[b,ν] tc[μ,ν];   out[a,b] = Σ_μ tet[a,μ] td[b,μ]
@inline function _coord_to_tetrad_cov(nn::SMatrix{4,4,ComplexF64}, gcov, tet)
    tb = @inbounds @SMatrix ComplexF64[
        gcov[ν,1]*nn[α,1] + gcov[ν,2]*nn[α,2] + gcov[ν,3]*nn[α,3] + gcov[ν,4]*nn[α,4]
        for ν in 1:4, α in 1:4]
    tc = @inbounds @SMatrix ComplexF64[
        gcov[μ,1]*tb[ν,1] + gcov[μ,2]*tb[ν,2] + gcov[μ,3]*tb[ν,3] + gcov[μ,4]*tb[ν,4]
        for μ in 1:4, ν in 1:4]
    td = @inbounds @SMatrix ComplexF64[
        tet[b,1]*tc[μ,1] + tet[b,2]*tc[μ,2] + tet[b,3]*tc[μ,3] + tet[b,4]*tc[μ,4]
        for b in 1:4, μ in 1:4]
    return @inbounds @SMatrix ComplexF64[
        tet[a,1]*td[b,1] + tet[a,2]*td[b,2] + tet[a,3]*td[b,3] + tet[a,4]*td[b,4]
        for a in 1:4, b in 1:4]
end

# ta[μ,β] = Σ_α v[α] Γ[μ,α,β]  (the half-step transport "connection·k" reduction).
@inline _connection_dot(v::SVector{4}, Γ) = @inbounds @SMatrix Float64[
    v[1]*Γ[μ,1,β] + v[2]*Γ[μ,2,β] + v[3]*Γ[μ,3,β] + v[4]*Γ[μ,4,β]
    for μ in 1:4, β in 1:4]

# dNN[μ,ν] = -Σ_β (T[μ,β] N[β,ν] + T[ν,β] N[μ,β])  (parallel-transport increment).
@inline _transport_dNN(T::SMatrix{4,4,Float64}, N::SMatrix{4,4,ComplexF64}) =
    @inbounds @SMatrix ComplexF64[
        -((T[μ,1]*N[1,ν] + T[ν,1]*N[μ,1]) + (T[μ,2]*N[2,ν] + T[ν,2]*N[μ,2]) +
          (T[μ,3]*N[3,ν] + T[ν,3]*N[μ,3]) + (T[μ,4]*N[4,ν] + T[ν,4]*N[μ,4]))
        for μ in 1:4, ν in 1:4]

# Coupled Stokes vector -> coordinate-frame contravariant N^{μν}: rebuild the sparse
# tetrad-frame N (time/parallel rows zero by construction) then rotate back with tet
# (tf[ν,a]=Σ_b tet[b,ν] ntc[a,b], nn[μ,ν]=Σ_a tet[a,μ] tf[ν,a]).
@inline function _tetrad_to_coord(se::SVector{4}, tet)
    z = zero(ComplexF64)
    ntc = @SMatrix ComplexF64[z            z                 z                 z;
                              z  (se[1]+se[2])  (se[3]-im*se[4])                 z;
                              z  (se[3]+im*se[4])  (se[1]-se[2])                 z;
                              z            z                 z                 z]
    tf = @inbounds @SMatrix ComplexF64[
        tet[1,ν]*ntc[a,1] + tet[2,ν]*ntc[a,2] + tet[3,ν]*ntc[a,3] + tet[4,ν]*ntc[a,4]
        for ν in 1:4, a in 1:4]
    return @inbounds @SMatrix ComplexF64[
        tet[1,μ]*tf[ν,1] + tet[2,μ]*tf[ν,2] + tet[3,μ]*tf[ν,3] + tet[4,μ]*tf[ν,4]
        for μ in 1:4, ν in 1:4]
end

@inline _stokes_from_tet(nt) = SVector(
    0.5 * real(nt[2, 2] + nt[3, 3]),
    0.5 * real(nt[2, 2] - nt[3, 3]),
    0.5 * real(nt[2, 3] + nt[3, 2]),
    0.5 * imag(nt[3, 2] - nt[2, 3]))

# Unit "normal"/Eulerian observer n^μ = -α g^{0μ} at the camera (α = lapse):
# n·n = -1 exactly; the spatial part is +g^{0i}/g^{00}/α = -β^i/α (β^i = -g^{0i}/g^{00}).
@inline function _normal_observer(gcon::SMatrix{4,4})
    lapse = 1.0 / sqrt(-gcon[1, 1])
    return SVector(1.0 / lapse, gcon[1, 2] / gcon[1, 1] / lapse,
                   gcon[1, 3] / gcon[1, 1] / lapse, gcon[1, 4] / gcon[1, 1] / lapse)
end

"""
    polarized_march(st, cfg, sampler, ys, dls, ν_obs, L_unit;
                    cam_up, intensity_only=false) -> SVector{4}

March ONE ray's coherency tensor from source to camera. `ys`/`dls` are the
trace_rays outputs for this ray (source→camera order). `sampler(x)` returns the
NamedTuple (rho, pgas, uu, bb, inside) at position x (code units). Returns the
camera-frame Stokes (I, Q, U, V) at ν_obs (physical, scaled by ν_obs³).
"""
function polarized_march(st, cfg::SimCoeffConfig, sampler, ys::Vector{SVector{8,Float64}},
                         dls::Vector{Float64}, ν_obs::Float64, L_unit::Float64;
                         cam_up::SVector{4,Float64}, intensity_only::Bool=false)
    # Immutable coherency-tensor state, rebound each step (StaticArrays keeps these
    # in registers/stack; no per-sample heap allocation).
    nn = zero(SMatrix{4,4,ComplexF64})
    nn_tmp = zero(SMatrix{4,4,ComplexF64})
    conn_old = zero(SArray{Tuple{4,4,4},Float64})
    kcon_old = zero(SVector{4,Float64})
    dλ_old = 0.0
    m = length(ys)
    inv_nu = 1.0 / ν_obs
    # single CKS has |det g| = 1 exactly (e_(1) already unit); every other spacetime
    # (superposed / provider / interpolated) needs the explicit e_(1) renormalization.
    renorm_e1 = !(st isa KerrSchildCartesian)

    @inbounds for n in 1:m
        y = ys[n]
        x = SVector(y[1], y[2], y[3], y[4])
        kcov = -SVector(y[5], y[6], y[7], y[8])  # photon k = -p
        dλ = dls[n]
        dλ_new = n < m ? dls[min(n + 1, m)] : dλ
        # reference pairing: coefficients at sample n couple over the segment BEHIND it
        dλ_cgs = (n == 1 ? 0.0 : dls[n-1]) * L_unit * inv_nu

        gcov, gcon = metric_and_inverse(st, x)
        # reuse the gcon just computed; christoffel would otherwise re-derive it at x
        Γ = christoffel(st, x, gcon, metric_gradient(st, x))
        conn_old = n == 1 ? Γ : 0.5 .* (conn_old .+ Γ)
        kcon = gcon * kcov
        kcon_old = n == 1 ? kcon : 0.5 .* (kcon_old .+ kcon)

        # First half-step transport with midpoint-averaged registers. The
        # source->camera march runs along the photon k = -p, i.e. opposite to the
        # traced-λ direction, over the positive affine spans dls.
        ta = _connection_dot(kcon_old, conn_old)
        dll1 = (dλ_old + dλ) / 2
        nn_tmp = nn_tmp .+ _transport_dNN(ta, nn) .* dll1
        nn = nn_tmp

        # plasma coupling at this sample
        s = sampler(x)
        if s.inside && !(s.bb[1] == 0.0 && s.bb[2] == 0.0 && s.bb[3] == 0.0)
            fv = primitive_four_vectors(s.uu[1], s.uu[2], s.uu[3],
                                        s.bb[1], s.bb[2], s.bb[3], gcov, gcon)
            j, α, ρ, tet, valid = eight_coefficients(cfg, s.rho, s.pgas, fv, gcov, gcon,
                                                     kcov, kcon, ν_obs, renorm_e1)
            if valid
                if intensity_only
                    j = SVector(j[1], 0.0, 0.0, 0.0)
                    α = SVector(α[1], 0.0, 0.0, 0.0)
                    ρ = _ZERO4S
                end
                # tet is the fluid tetrad already built inside eight_coefficients
                nt = _coord_to_tetrad_cov(nn, gcov, tet)
                ss = _stokes_from_tet(nt)
                se = stokes_couple(ss, j, α, ρ, dλ_cgs)
                nn = _tetrad_to_coord(se, tet)
            end
        end

        # second (predictor) transport with current registers
        nn_tmp = nn
        tg = _connection_dot(kcon, Γ)
        dll2 = (dλ + dλ_new) / 4
        nn = nn .+ _transport_dNN(tg, nn_tmp) .* dll2

        dλ_old = dλ
        kcon_old = kcon
        conn_old = Γ
    end

    # camera readout: rotate N into the camera tetrad, scale by ν_obs³
    y = ys[m]
    x = SVector(y[1], y[2], y[3], y[4])
    kcov = -SVector(y[5], y[6], y[7], y[8])     # photon k = -p
    gcov, gcon = metric_and_inverse(st, x)
    kcon = gcon * kcov
    # unit normal (Eulerian) observer at the camera
    ucon = _normal_observer(gcon)
    ucov = gcov * ucon
    # camera vertical, orthogonalized against u (blacklight up_con)
    vert = cam_up
    upc0 = ucon[1] * vert[1] -
           (ucov[2] * vert[2] + ucov[3] * vert[3] + ucov[4] * vert[4]) / ucov[1]
    up = SVector(upc0, vert[2] + ucon[2] * vert[1], vert[3] + ucon[3] * vert[1],
                 vert[4] + ucon[4] * vert[1])
    tet = fluid_tetrad(ucon, ucov, kcon, kcov, up, gcov, gcon; renorm_e1 = renorm_e1)
    nt = _coord_to_tetrad_cov(nn, gcov, tet)
    return _stokes_from_tet(nt) * ν_obs^3
end

"""
    unpolarized_march(st, cfg, sampler, ys, dls, ν_obs, L_unit) -> Float64

Stokes-I fast mode: stable exponential propagator over the invariant intensity
(blacklight unpolarized_transfer), with the per-sample `unpolarized_coefficients` and
`unpolarized_segment` operator the fused kernels use. Returns I_ν at the camera.
"""
function unpolarized_march(st, cfg::SimCoeffConfig{T}, sampler, ys, dls,
                           ν_obs, L_unit) where {T}
    # Accumulator adopts the config's scalar type so ForwardDiff Dual sensitivities
    # flow through the march; SimCoeffConfig{Float64} gives a plain 0.0.
    Iinv = zero(T)
    inv_nu = 1.0 / ν_obs
    m = length(ys)
    @inbounds for n in 1:m
        y = ys[n]
        x = SVector(y[1], y[2], y[3], y[4])
        s = sampler(x)
        (!s.inside || (s.bb[1] == 0.0 && s.bb[2] == 0.0 && s.bb[3] == 0.0)) && continue
        (s.rho <= 0.0 || s.pgas <= 0.0) && continue
        gcov, gcon = metric_and_inverse(st, x)
        kcov = -SVector(y[5], y[6], y[7], y[8])  # photon k = -p
        fv = primitive_four_vectors(s.uu[1], s.uu[2], s.uu[3],
                                    s.bb[1], s.bb[2], s.bb[3], gcov, gcon)
        j, a = unpolarized_coefficients(cfg, s.rho, s.pgas, fv, kcov, ν_obs)
        Iinv = couple_unpolarized(Iinv, j, a, (n == 1 ? 0.0 : dls[n-1]) * L_unit * inv_nu)
    end
    return Iinv * ν_obs^3
end
