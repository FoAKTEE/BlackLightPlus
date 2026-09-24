# Fixed-control reverse (camera -> source) transfer marches, one per radiative-transfer
# mode, shared by the CPU `render` and every fused imaging kernel. Each march steps the
# null geodesic itself (fixed-control RK4, the stop rule of the CPU sampled tracer),
# samples the fluid through an inlined `sample(y)` closure and accumulates the observed
# intensity in reverse:
#   Stokes I:  I_obs = Σ_n B_n Π_{k<n} A_k with the segment operators (A_n, B_n) of
#              `unpolarized_segment`, a running transmission `trans` and sum `Iinv`;
#   IQUV:      S += G (R_n b_n), G <- G (R_n M_n R_nᵀ) with (M_n, b_n) = `stokes_segment`
#              in the fluid tetrad and R_n the Q-U rotation into the camera polarization
#              frame parallel-transported along the ray (`geodesic_transport_rhs`), so S
#              ends in the camera frame with no final rotation (cf. ipole/Krang).
# The per-sample physics (metric, fluid decode, cuts, coefficients, tetrads, segment
# operators) is the Radiation/Geodesics code; the marches are allocation-free and
# isbits-generic, so KernelAbstractions kernels compile them unchanged.

# ---------------------------------------------------------------- march control
"""
    KerrMarch(a, r_h, C, escape, R_em)

Fixed-control march on single Kerr (M = 1, spin `a`): step h = (r - r_h)/C in the
Kerr-Schild radius, stop by `march_stop` (horizon backstop, redshift cut, outbound
escape past `escape` or `R_em`). All fields share the geodesic-lane type T
(Float32 or Float64): the geodesic and the parallel-transported frame run in T, the
radiative transfer in Float64 with the parameter set's fp64 spin `gp.bh_a`.
"""
struct KerrMarch{T<:Real}
    a::T
    r_h::T
    C::T
    escape::T
    R_em::T
end

"""
    SpacetimeMarch(sp, C, escape, R_em)

Fixed-control march in a stationary spacetime `sp` (superposed Kerr-Schild binary, ADM
grid, or Kerr-Schild), Float64: step and capture from `spacetime_control(sp, y, C)`,
outbound escape on its radius past `escape` or `R_em`.
"""
struct SpacetimeMarch{S}
    sp::S
    C::Float64
    escape::Float64
    R_em::Float64
end

"""
    StoredPath(sp, ys, dls)

March along a stored trajectory of spacetime `sp` (a `trace_rays`/`trace_rays_sampled`
ray: states `ys` source → camera, `dls[n]` the affine step from state n to n+1): the
reverse march visits the stored states camera → source, pairs the sample at state n
with the step dls[n-1] behind it, and (polarized) transports the camera frame over each
stored step with the same RK4.
"""
struct StoredPath{S,V,D}
    sp::S
    ys::V
    dls::D
end

# (stop, h, r) at march step `it`: the stop decision, the step to take, the radius for
# the next escape test.
@inline function march_control(c::KerrMarch{T}, y::SVector{8,T}, r_prev::T, it) where {T}
    r, En = ks_radius_energy(y, c.a)
    return march_stop(r - c.r_h, En, r, r_prev, c.escape, c.R_em), (r - c.r_h) / c.C, r
end
@inline function march_control(c::SpacetimeMarch, y::SVector{8,Float64}, r_prev::Float64, it)
    cap, h, R = spacetime_control(c.sp, y, c.C)
    return cap || march_escaped(R, r_prev, c.escape, c.R_em), h, R
end
@inline function march_control(c::StoredPath, y, r_prev, it)
    n = length(c.ys) + 1 - it                            # stored index of the state y
    return n < 1, n > 1 ? c.dls[n-1] : zero(eltype(c.dls)), r_prev
end

# The state after step `it`: the RK4 result on a fixed-control march, the next stored
# state on a stored path (the RK4 result there only carries the frame).
@inline next_state(::Union{KerrMarch,SpacetimeMarch}, y_rk4, it) = y_rk4
@inline function next_state(c::StoredPath, y_rk4, it)
    n = length(c.ys) + 1 - it
    return n > 1 ? c.ys[n-1] : y_rk4
end
@inline geodesic_step(c::Union{KerrMarch,SpacetimeMarch}, y, h, it) =
    rk4_step_fixed(geodesic_spacetime(c), y, h)
@inline geodesic_step(c::StoredPath, y, h, it) = next_state(c, y, it)

# Spacetime of the geodesic lane (the march's own number type) and of the transfer lane.
@inline geodesic_spacetime(c::KerrMarch) = KerrSchildCartesian(c.a)
@inline geodesic_spacetime(c::Union{SpacetimeMarch,StoredPath}) = c.sp
@inline transfer_spacetime(::KerrMarch, gp) = KerrSchildCartesian(gp.bh_a)
@inline transfer_spacetime(c::Union{SpacetimeMarch,StoredPath}, gp) = c.sp
# e_(1) of the fluid tetrad needs renormalization off single CKS (|det g| ≠ 1).
@inline renorm_e1(::KerrMarch) = false
@inline renorm_e1(::SpacetimeMarch) = true
@inline renorm_e1(c::StoredPath) = !(c.sp isa KerrSchildCartesian)

# ---------------------------------------------------------------- per-sample physics
# prim = (rho, u1, u2, u3, eint, b1, b2, b3) in code units (normal-frame velocity, lab
# field), the layout of every fluid sampler.
@inline function _prim_fv(prim, gcov, gcon)
    return primitive_four_vectors(prim[2], prim[3], prim[4], prim[6], prim[7], prim[8],
                                  gcov, gcon)
end

# One sample -> segment operator (A, B) of the invariant intensity. `gp` is one
# parameter set, or an NTuple of sets differing only in units (mass_scale ladder), which
# gives SVector-valued (A, B) with one lane per set.
@inline function _segment_ab(gp, st, prim, x, kcov, ds)
    gcov, gcon = metric_and_inverse(st, x)
    j, a = unpolarized_coefficients(gp, prim[1], pgas_from_eint(prim[5], gp.plasma_gamma),
                                    _prim_fv(prim, gcov, gcon), kcov, gp.nu_obs)
    return unpolarized_segment(j, a, ds)
end
@inline function _segment_ab(gps::NTuple{K,Any}, st, prim, x, kcov, ds) where {K}
    # the metric is re-evaluated per set: holding it live across the K lanes costs more
    # device registers than recomputing it
    ab = ntuple(Val(K)) do k
        _segment_ab(gps[k], st, prim, x, kcov, ds)
    end
    return SVector{K,Float64}(ntuple(k -> ab[k][1], Val(K))),
           SVector{K,Float64}(ntuple(k -> ab[k][2], Val(K)))
end

@inline _first_params(gp) = gp
@inline _first_params(gps::Tuple) = gps[1]
@inline _unit(::Type{T}) where {T<:Number} = one(T)
@inline _unit(::Type{SVector{K,T}}) where {K,T} = SVector(ntuple(_ -> one(T), Val(K)))

"""
    reverse_unpolarized_march(y0, sample, T, gp,
                              ctl::Union{KerrMarch,SpacetimeMarch,StoredPath}, nmax) -> I_ν

Stokes-I march of one camera ray `y0` (state [x^μ; p_μ], p = -k): fixed-control RK4
under `ctl` (or the stored states of a `StoredPath`), fluid from `sample(y)::SVector{8,Float64}`, invariant coefficients from the
parameter set `gp` (`coefficient_params` + `nu_obs`, `L_unit`), reverse accumulation
into the observed I_ν = ν_obs³ I_inv. `T` is the accumulator type: Float64, a
ForwardDiff Dual (parameter derivatives), or SVector{K,Float64} with `gp` an NTuple of K
sets. The transfer runs in Float64 from the converted state in every lane type.
"""
@inline function reverse_unpolarized_march(y0::SVector{8,Tg}, sample::F, ::Type{T}, gp,
                                           ctl, nmax) where {F,T,Tg}
    g1 = _first_params(gp)
    st = transfer_spacetime(ctl, g1)
    y = y0
    Iinv = zero(T)
    trans = _unit(T)
    r_prev = Tg(Inf)
    for it in 1:nmax
        stop, h, r = march_control(ctl, y, r_prev, it)
        stop && break
        r_prev = r
        # force-inline the sampler: left as a call it perturbs register allocation of
        # this body on the device
        prim = @inline sample(y)
        if prim[1] > 0.0
            x = SVector{4,Float64}(y[1], y[2], y[3], y[4])
            kcov = -SVector{4,Float64}(y[5], y[6], y[7], y[8])   # photon k = -p
            ds = Float64(h) * g1.L_unit / g1.nu_obs
            A, B = _segment_ab(gp, st, prim, x, kcov, ds)
            Iinv = Iinv .+ trans .* B
            trans = trans .* A
        end
        y = geodesic_step(ctl, y, h, it)
    end
    return Iinv .* g1.nu_obs^3
end

# ---------------------------------------------------------------- polarized helpers
const _S0 = SVector(0.0, 0.0, 0.0, 0.0)
const _G0 = one(SMatrix{4,4,Float64})
@inline _intensity_only(gp) = hasfield(typeof(gp), :intensity_only) && gp.intensity_only

# One sample -> (j, α, ρ, fluid tetrad, valid): primitive decode + `eight_coefficients`.
@inline function _sample_coefficients(prim, kcov, kcon, gcov, gcon, gp, renorm::Bool)
    return eight_coefficients(gp, prim[1], pgas_from_eint(prim[5], gp.plasma_gamma),
                              _prim_fv(prim, gcov, gcon), gcov, gcon, kcov, kcon,
                              gp.nu_obs, renorm)
end

# Stokes rotation fluid tetrad -> readout tetrad E: a pure Q-U rotation by 2χ (I, V
# fixed), χ the angle of E's e_(1) in the fluid (e_(1), e_(2)) basis (the coherency
# round trip reduced to its rotation). Returns (cos 2χ, sin 2χ).
@inline function _qu_rotation(tet_fl::SMatrix{4,4}, E::SMatrix{4,4}, gcov::SMatrix{4,4})
    e1f = SVector(tet_fl[2,1], tet_fl[2,2], tet_fl[2,3], tet_fl[2,4])
    e2f = SVector(tet_fl[3,1], tet_fl[3,2], tet_fl[3,3], tet_fl[3,4])
    E1  = SVector(E[2,1], E[2,2], E[2,3], E[2,4])
    c = dot(E1, gcov * e1f); s = dot(E1, gcov * e2f)
    nrm = sqrt(c*c + s*s); c /= nrm; s /= nrm
    return c*c - s*s, 2.0*s*c
end

# R b and R M Rᵀ for R = [1 0 0 0; 0 c s 0; 0 -s c 0; 0 0 0 1]: only the Q, U rows (and
# columns) mix.
@inline _rotate_qu(c, s, b::SVector{4}) =
    SVector(b[1], c * b[2] + s * b[3], c * b[3] - s * b[2], b[4])
@inline function _rotate_qu(c, s, M::SMatrix{4,4})
    A = SMatrix{4,4}(M[1,1], c*M[2,1] + s*M[3,1], c*M[3,1] - s*M[2,1], M[4,1],
                     M[1,2], c*M[2,2] + s*M[3,2], c*M[3,2] - s*M[2,2], M[4,2],
                     M[1,3], c*M[2,3] + s*M[3,3], c*M[3,3] - s*M[2,3], M[4,3],
                     M[1,4], c*M[2,4] + s*M[3,4], c*M[3,4] - s*M[2,4], M[4,4])   # R M
    return SMatrix{4,4}(A[1,1], A[2,1], A[3,1], A[4,1],
                        c*A[1,2] + s*A[1,3], c*A[2,2] + s*A[2,3], c*A[3,2] + s*A[3,3], c*A[4,2] + s*A[4,3],
                        c*A[1,3] - s*A[1,2], c*A[2,3] - s*A[2,2], c*A[3,3] - s*A[3,2], c*A[4,3] - s*A[4,2],
                        A[1,4], A[2,4], A[3,4], A[4,4])                        # (R M) Rᵀ
end

"""
    reverse_polarized_march(y0, sample, gp,
                            ctl::Union{KerrMarch,SpacetimeMarch,StoredPath}, nmax)
        -> SVector{4} (I, Q, U, V)

Full-Stokes march of one camera ray: fixed-control RK4 of the geodesic (or the stored
states of a `StoredPath`) together with RK4 parallel transport of the camera frame (its up leg e_(2) and 4-velocity), reverse
Mueller-chain accumulation (file header). The camera frame is the normal (Eulerian)
observer with the image vertical (`gp.cu1..cu4`). The march also stops where the
transported frame admits no spacelike screen leg (`fluid_tetrad_checked`), i.e. the
photon momentum has drifted off the null cone on steps far too coarse for the local
curvature. Returns the observed Stokes ν_obs³ S in
the camera frame.
"""
@inline function reverse_polarized_march(y0::SVector{8,Tg}, sample::F, gp, ctl,
                                         nmax) where {F,Tg}
    st = transfer_spacetime(ctl, gp)
    stg = geodesic_spacetime(ctl)
    renorm = renorm_e1(ctl)
    y = y0
    # camera frame: built in Float64 from the converted initial state, then narrowed
    # onto the geodesic lane
    y64 = SVector{8,Float64}(y0)
    x0 = SVector(y64[1], y64[2], y64[3], y64[4])
    kcov0 = -SVector(y64[5], y64[6], y64[7], y64[8])
    gcov0, gcon0 = metric_and_inverse(st, x0); kcon0 = gcon0 * kcov0
    e0c = _normal_observer(gcon0)
    e0cov0 = gcov0 * e0c
    cu = SVector(gp.cu1, gp.cu2, gp.cu3, gp.cu4)
    upc0 = e0c[1]*cu[1] - (e0cov0[2]*cu[2] + e0cov0[3]*cu[3] + e0cov0[4]*cu[4]) / e0cov0[1]
    upcam = SVector(upc0, cu[2]+e0c[2]*cu[1], cu[3]+e0c[3]*cu[1], cu[4]+e0c[4]*cu[1])
    Ecam = fluid_tetrad(e0c, e0cov0, kcon0, kcov0, upcam, gcov0, gcon0; renorm_e1 = renorm)
    eU  = SVector{4,Tg}(Ecam[3, 1], Ecam[3, 2], Ecam[3, 3], Ecam[3, 4])   # camera e_(2)
    eUu = SVector{4,Tg}(e0c)                             # camera 4-velocity
    S = _S0; G = _G0
    r_prev = Tg(Inf)
    for it in 1:nmax
        stop, h, r = march_control(ctl, y, r_prev, it)
        stop && break
        r_prev = r
        prim = @inline sample(y)
        if prim[1] > 0.0
            x = SVector{4,Float64}(y[1], y[2], y[3], y[4])
            gcov, gcon = metric_and_inverse(st, x)
            kcov = -SVector{4,Float64}(y[5], y[6], y[7], y[8]); kcon = gcon * kcov
            j, al, ro, tet_fl, valid = _sample_coefficients(prim, kcov, kcon, gcov, gcon,
                                                            gp, renorm)
            if _intensity_only(gp)                      # diagnostic: Stokes-I physics only
                j = SVector(j[1], 0.0, 0.0, 0.0); al = SVector(al[1], 0.0, 0.0, 0.0)
                ro = _S0
            end
            if valid
                # readout screen = the transported camera frame; fluid_tetrad is built
                # from parallel-transport-invariant operations, so E == PT(Ecam)
                eU64 = SVector{4,Float64}(eU); eUu64 = SVector{4,Float64}(eUu)
                ucov = gcov * eUu64
                E, screen_ok = fluid_tetrad_checked(eUu64, ucov, kcon, kcov, eU64, gcov,
                                                    gcon; renorm_e1 = renorm)
                # no spacelike screen leg: the geodesic step is far too coarse for the
                # local curvature (k off the null cone), and the march stops there
                screen_ok || break
                ds = Float64(h) * gp.L_unit / gp.nu_obs
                M, b = stokes_segment(j, al, ro, ds)
                c2, s2 = _qu_rotation(tet_fl, E, gcov)
                S = stokes_admissible(S + G * _rotate_qu(c2, s2, b))
                G = G * _rotate_qu(c2, s2, M)
            end
        end
        # low-storage streaming RK4 of {y, eU, eUu}: the ops of rk4_step_fixed's combine
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
    return S * gp.nu_obs^3
end
