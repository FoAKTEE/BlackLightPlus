# The reverse (camera → source) transfer marches. Each step samples the plasma at the
# current state, accumulates its segment operator behind the ones already seen, and
# advances the geodesic under the march control:
#   Stokes I:  I_obs = ν_obs³ Σ_n B_n Π_{k<n} A_k, with a running transmission and sum;
#   IQUV:      S += G (R_n b_n), G ← G (R_n M_n R_nᵀ), with (M_n, b_n) the Mueller operator
#              in the fluid tetrad and R_n the Q-U rotation into the camera frame
#              parallel-transported along the ray, so S ends in the camera frame
#              (cf. ipole, Krang).
# The geodesic runs in the lane of the initial state; the transfer runs in Float64 from
# the converted state. A program `prog` carries `st` (transfer-lane spacetime), `control`
# (`SpacetimeMarch` or `StoredPath`), `provider` (a `PlasmaProvider`), `params` (a
# `TransferParams`, or for Stokes I an `NTuple` of sets differing only in `units`) and
# `frame::CameraFrame` (Float64; its `up` is the camera polarization vertical).

@inline _first_params(p::TransferParams) = p
@inline _first_params(ps::Tuple) = ps[1]
@inline _unit(::Type{T}) where {T<:Number} = one(T)
@inline _unit(::Type{SVector{K,T}}) where {K,T} = SVector(ntuple(_ -> one(T), Val(K)))

# One sample of one parameter set: decode at the metric of `x`, coefficients, segment.
@inline function _segment_ab(p::TransferParams, st, prim, x, kcov, ds)
    g = metric_and_inverse(st, x)
    state = decode(prim, g, p.units, p.electrons, p.cuts)
    j, a = coefficients(StokesI(), p, state, kcov, p.nu_obs)
    return segment(StokesI(), j, a, ds)
end
# A ladder of K sets: one lane per set. The metric is re-evaluated per set, which holds
# fewer values live across the K lanes than sharing it.
@inline function _segment_ab(ps::NTuple{K,TransferParams}, st, prim, x, kcov, ds) where {K}
    ab = ntuple(Val(K)) do k
        _segment_ab(ps[k], st, prim, x, kcov, ds)
    end
    return SVector{K,Float64}(ntuple(k -> ab[k][1], Val(K))),
           SVector{K,Float64}(ntuple(k -> ab[k][2], Val(K)))
end

# Accumulate the sample at `y` (affine step `h`) into the intensity `Iinv` behind the
# transmission `trans`; a sample that neither emits nor absorbs leaves both untouched.
@inline function _accumulate(p::TransferParams, provider, st, y, h, Iinv, trans)
    x = SVector{4,Float64}(y[1], y[2], y[3], y[4])
    state, _ = @inline sample(provider, st, x, p)
    if state.valid
        kcov = -SVector{4,Float64}(y[5], y[6], y[7], y[8])   # photon k = -p
        ds = Float64(h) * p.units.L_unit / p.nu_obs
        j, a = coefficients(StokesI(), p, state, kcov, p.nu_obs)
        A, B = segment(StokesI(), j, a, ds)
        Iinv = Iinv .+ trans .* B
        trans = trans .* A
    end
    return Iinv, trans
end
# The ladder samples the primitives once and decodes them per set, so it needs a grid
# provider; a sample with rest mass applies every lane's operator.
@inline function _accumulate(ps::NTuple{K,TransferParams}, provider::GridProvider, st, y, h,
                             Iinv, trans) where {K}
    x = SVector{4,Float64}(y[1], y[2], y[3], y[4])
    prim = @inline primitives(provider, x)
    if prim.rho > 0.0
        kcov = -SVector{4,Float64}(y[5], y[6], y[7], y[8])   # photon k = -p
        g1 = ps[1]
        ds = Float64(h) * g1.units.L_unit / g1.nu_obs
        A, B = _segment_ab(ps, st, prim, x, kcov, ds)
        Iinv = Iinv .+ trans .* B
        trans = trans .* A
    end
    return Iinv, trans
end

"""
    march(::StokesI, y0::RayState{Tg}, prog) -> output_type(StokesI(), prog.params)

Observed Stokes I of the camera ray `y0 = [x^μ; p_μ]` (`p = -k`, `p_t = 1`): the ray is
advanced under `prog.control` in the lane `Tg` of `y0` for at most
`max_steps(prog.control)` steps; each sample of `prog.provider` that emits or absorbs
contributes its operator from [`segment`](@ref), and the result is
`I_ν = ν_obs³ Σ_n B_n Π_{k<n} A_k` (cgs). With an `NTuple{K}` of parameter sets
(a mass-scale ladder; grid providers only) the primitives are sampled once per step
and the result is an `SVector{K,Float64}` whose lane `k` performs the operations of
the march of set `k` alone. Pure and allocation-free.
"""
@inline function march(::StokesI, y0::SVector{8,Tg}, prog) where {Tg}
    ctl = prog.control
    ps = prog.params
    g1 = _first_params(ps)
    st = transfer_spacetime(prog)
    T = output_type(StokesI(), ps)
    y = y0
    Iinv = zero(T)
    trans = _unit(T)
    r_prev = Tg(Inf)
    for it in 1:max_steps(ctl)
        m = march_control(ctl, y, r_prev, it)
        m.stop && break
        r_prev = m.R
        Iinv, trans = _accumulate(ps, prog.provider, st, y, m.h, Iinv, trans)
        y = geodesic_step(ctl, y, m.h, it)
    end
    return Iinv .* g1.nu_obs^3
end

const _S0 = SVector(0.0, 0.0, 0.0, 0.0)
const _G0 = one(SMatrix{4,4,Float64})

# The camera polarization leg e_(2) and the camera observer at the ray start: the
# normal observer and the tetrad with e_(3) along k and e_(2) along the image
# vertical `cu`, built in Float64 from the converted state and narrowed to the lane.
@inline function _camera_frame(st, y0::SVector{8,Tg}, cu, renorm) where {Tg}
    y64 = SVector{8,Float64}(y0)
    x0 = SVector(y64[1], y64[2], y64[3], y64[4])
    kcov0 = -SVector(y64[5], y64[6], y64[7], y64[8])
    gcov0, gcon0 = metric_and_inverse(st, x0); kcon0 = gcon0 * kcov0
    e0c = normal_observer(gcon0)
    e0cov0 = gcov0 * e0c
    upc0 = e0c[1]*cu[1] - (e0cov0[2]*cu[2] + e0cov0[3]*cu[3] + e0cov0[4]*cu[4]) / e0cov0[1]
    upcam = SVector(upc0, cu[2]+e0c[2]*cu[1], cu[3]+e0c[3]*cu[1], cu[4]+e0c[4]*cu[1])
    Ecam = fluid_tetrad(e0c, e0cov0, kcon0, kcov0, upcam, gcov0, gcon0; renorm_e1 = renorm)
    eU  = SVector{4,Tg}(Ecam[3, 1], Ecam[3, 2], Ecam[3, 3], Ecam[3, 4])
    eUu = SVector{4,Tg}(e0c)
    return eU, eUu
end

"""
    march(::IQUV, y0::RayState{Tg}, prog) -> SVector{4,Float64}

Observed `(I, Q, U, V)` (cgs, `ν_obs³` times the invariant Stokes vector) of the camera
ray `y0` in the camera polarization frame: the normal observer at the camera with
`e_(2)` along `prog.frame.up`, parallel-transported with the geodesic by RK4 in the
lane `Tg` (`geodesic_transport_rhs`). Each valid sample contributes its Mueller
operator in the fluid tetrad, rotated into the transported frame, and the accumulated
Stokes vector is kept admissible. The march also ends where the transported frame
admits no spacelike screen leg (`fluid_tetrad_checked`), i.e. the photon momentum has
drifted off the null cone on a step far too coarse for the local curvature. `params`
is one `Float64` `TransferParams`. Pure and allocation-free.
"""
@inline function march(::IQUV, y0::SVector{8,Tg}, prog) where {Tg}
    ctl = prog.control
    p = prog.params
    st = transfer_spacetime(prog)
    stg = geodesic_spacetime(ctl)
    renorm = renorm_e1(ctl)
    y = y0
    eU, eUu = _camera_frame(st, y0, SVector{4,Float64}(prog.frame.up), renorm)
    S = _S0; G = _G0
    r_prev = Tg(Inf)
    for it in 1:max_steps(ctl)
        m = march_control(ctl, y, r_prev, it)
        m.stop && break
        h = m.h
        r_prev = m.R
        x = SVector{4,Float64}(y[1], y[2], y[3], y[4])
        state, g = @inline sample(prog.provider, st, x, p)
        if state.valid
            gcov, gcon = g
            kcov = -SVector{4,Float64}(y[5], y[6], y[7], y[8]); kcon = gcon * kcov
            j, al, ro, tet_fl, valid = coefficients(IQUV(), p, state, kcov, kcon, g, p.nu_obs,
                                                    renorm)
            if valid
                # the readout screen is the transported camera frame: fluid_tetrad is
                # built from parallel-transport-invariant operations, so E = PT(Ecam)
                eU64 = SVector{4,Float64}(eU); eUu64 = SVector{4,Float64}(eUu)
                ucov = gcov * eUu64
                E, screen_ok = fluid_tetrad_checked(eUu64, ucov, kcon, kcov, eU64, gcov,
                                                    gcon; renorm_e1 = renorm)
                screen_ok || break
                ds = Float64(h) * p.units.L_unit / p.nu_obs
                M, b = segment(IQUV(), j, al, ro, ds)
                c2, s2 = qu_rotation(tet_fl, E, gcov)
                S = stokes_admissible(S + G * rotate_qu(c2, s2, b))
                G = G * rotate_qu(c2, s2, M)
            end
        end
        # low-storage streaming RK4 of {y, eU, eUu}, the operations of `rk4_step`
        dy, (de, du) = geodesic_transport_rhs(stg, y, (eU, eUu))
        ay = dy; ae = de; au = du
        ys = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
        dy, (de, du) = geodesic_transport_rhs(stg, ys, (es, us))
        ay = ay + 2*dy; ae = ae + 2*de; au = au + 2*du
        ys = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
        dy, (de, du) = geodesic_transport_rhs(stg, ys, (es, us))
        ay = ay + 2*dy; ae = ae + 2*de; au = au + 2*du
        ys = y + h*dy; es = eU + h*de; us = eUu + h*du
        dy, (de, du) = geodesic_transport_rhs(stg, ys, (es, us))
        ay = ay + dy; ae = ae + de; au = au + du
        y   = next_state(ctl, y + (h/6)*ay, it)
        eU  = eU  + (h/6)*ae
        eUu = eUu + (h/6)*au
    end
    return S * p.nu_obs^3
end
