# Images of an emission surface by the numerical march: every camera ray marched under the
# plan's budget, and at each crossing of the surface the crossing state found to the
# integrator's order (`surface_crossing`), the k-th crossing being image order k - 1.

"""
    CrossingProgram(st, budget, rays, disk, orders)

One screen point of the marched image of the equatorial disk `disk` on the Kerr spacetime
`st` (geodesic lane, `budget` its fixed-control budget): the camera ray of `rays` is
advanced by `march_step(budget.integrator, …)` under `march_control`, and every step whose
end lies on the other side of the plane `z = 0` yields the crossing state
`surface_crossing(budget.integrator, st, y, h, EquatorialPlane())`; a ninth-order march
carries its state compensated (`march_step_compensated`). At crossing `k` the
landing of order `k − 1` is read off the state: the Kerr–Schild (= Boyer–Lindquist)
radius `r`, the photon's `λ = −p_φ/p_t` and `η = r² p_z²/p_t²` (`p_θ = −r p_z` on the
equator), the emitted radial direction `νr = −sign(dr/dλ)` of the backward ray, the
Kerr–Schild time `x^t` and azimuth; the intensity is
`equatorial_disk_intensity(r, νr, a, λ, η)` with the disk's knobs. The march stops on
the budget's rule or after the highest order of `orders` (an `NTuple{K,Int}` of
ascending orders from 0). Called as `prog(item, i, outs...)` with the `5K + 1` vectors of
[`crossing_outputs`](@ref): quantity-major as [`analytic_outputs`](@ref) (intensity,
radius, time, azimuth, radial sign), then the step count. A missing crossing gives
intensity 0 and NaN observables; a ray without a positive-energy null root at the camera
gives NaN everywhere.
"""
struct CrossingProgram{S,B,R,D,O} <: Function
    st::S
    budget::B
    rays::R
    disk::D
    orders::O
end
Adapt.@adapt_structure CrossingProgram

"""
    crossing_outputs(K, n) -> NTuple{5K + 1,Vector{Float64}}

The output vectors of a [`CrossingProgram`](@ref) of `K` orders over `n` items: the
[`analytic_outputs`](@ref) layout followed by the step count.
"""
crossing_outputs(K::Integer, n::Integer) =
    ntuple(_ -> Vector{Float64}(undef, n), ANALYTIC_QUANTITIES * K + 1)

const _NO_LANDING = (0.0, NaN, NaN, NaN, NaN)

# The landing quantities (I, r, t, φ, νr) of the crossing state yc on Kerr of spin a.
@inline function crossing_landing(disk::EquatorialDisk, a, yc::SVector{8})
    y = SVector{8,Float64}(yc)
    x, yy, pt = y[2], y[3], y[5]
    r = ks_radius_spin(a, x, yy, y[4])
    λ = -(x * y[7] - yy * y[6]) / pt
    η = r * r * y[8] * y[8] / (pt * pt)
    v = raise_index(KerrSchildCartesian(Float64(a)), SVector(y[1], x, yy, y[4]),
                    SVector(pt, y[6], y[7], y[8]))
    νr = x * v[2] + yy * v[3] > 0 ? -1.0 : 1.0
    I = equatorial_disk_intensity(r, νr, Float64(disk.a), λ, η; sub_kep = disk.sub_kep,
                                  betar = disk.betar, betaphi = disk.betaphi,
                                  gamma_p = disk.gamma_p, mu_p = disk.mu_p,
                                  sigma_p = disk.sigma_p, redshift_power = disk.redshift_power,
                                  rin = disk.rin)
    φ = atan(yy, x) - atan(Float64(a), r)
    return (Float64(I), r, y[1], φ, νr)
end

@inline ks_radius_spin(a, x, y, z) = ks_radius(KerrSchildCartesian(Float64(a)), SVector(0.0, x, y, z))

function (p::CrossingProgram{S,B,R,D,O})(item, i, outs::Vararg{Any,N}) where {S,B,R,D,K,O<:NTuple{K,Int},N}
    N == ANALYTIC_QUANTITIES * K + 1 ||
        throw(ArgumentError("CrossingProgram: $N outputs for $K orders"))
    y0, ok = camera_state(p.rays, item)
    T = typeof(p.budget.C)
    maps = ntuple(_ -> _NO_LANDING, Val(K))
    nmax = last(p.orders) + 1
    n = 0
    if ok
        y = SVector{8,T}(y0)
        lo = zero(y)
        r_prev = T(Inf)
        k = 0
        integ = p.budget.integrator
        while n < p.budget.nmax
            m = march_control(p.st, y, r_prev, p.budget)
            m.stop && break
            r_prev = m.R
            if integ isa ClassicalRK4
                yn = march_step(integ, p.st, y, m.h)
            else
                yn, lo = march_step_compensated(integ, p.st, y, lo, m.h)
            end
            n += 1
            if y[4] != 0 && (yn[4] == 0 || signbit(yn[4]) != signbit(y[4]))
                k += 1
                if any(==(k - 1), p.orders)
                    yc, _ = surface_crossing(integ, p.st, y, m.h, EquatorialPlane())
                    maps = let l = crossing_landing(p.disk, p.st.a, yc), mp = maps, n0 = k - 1
                        ntuple(j -> p.orders[j] == n0 ? l : mp[j], Val(K))
                    end
                end
                k >= nmax && break
            end
            y = yn
        end
    else
        maps = ntuple(_ -> (NaN, NaN, NaN, NaN, NaN), Val(K))
    end
    store_all!(outs, i, (quantity_major(maps)..., Float64(n)))
    return nothing
end

"""
    render_crossings(plan, disk::EquatorialDisk; orders = (0, 1, 2), coords = nothing,
                     backend = nothing) -> NamedTuple

The image of the equatorial disk `disk` by the numerical march of `plan` on Kerr
([`CrossingProgram`](@ref)): the plan's camera rays (or the image coordinates `coords`,
a vector of `(α, β)` tuples or a `CoordList`) marched on its geodesic lane under its
budget (integrator, step divisor, redshift cut, horizon backstop), with the landing of
order `n ∈ orders` at the `(n + 1)`-th crossing of the equator. With the precise mode
(`Plans.precise`) and the camera at infinity it reproduces the closed-form image of
[`render_analytic`](@ref) to rounding error away from the horizon. Returns `(I, Ib, rs,
ts, phis, redshift_sign, steps, orders, plan, run)` as `render_analytic` (`ts` the
Kerr–Schild time, `phis` the Kerr–Schild azimuth), each `npix × npix` (vectors for
`coords`), and `steps`, the steps per ray.

Determinism: bitwise per item, whatever the executor, device count or chunking.
"""
function render_crossings(plan::Plan, disk::EquatorialDisk; orders = (0, 1, 2), coords = nothing,
                          backend = nothing)
    t0 = time_ns()
    exec = resolve(backend)
    analytic_spin(plan, disk)
    ot = map(Int, Tuple(orders))
    (issorted(collect(ot)) && first(ot) >= 0) ||
        throw(ArgumentError("render_crossings: orders must ascend from 0 or above; got $orders"))
    gen = coords === nothing ? plan.pixels : coords isa CoordList ? coords : CoordList(coords)
    rays = CameraRays(plan.spacetime, plan.frame, gen)
    prog = CrossingProgram(plan.geodesic_spacetime, plan.geodesic_budget, rays, disk, ot)
    N = coords === nothing ? npixels(plan) : length(gen.α)
    K = length(ot)
    outs = crossing_outputs(K, N)
    foreach_item!(exec, prog, 1:N; outs)
    shape(v) = coords === nothing ? reshape(v, plan.camera.npix, plan.camera.npix) : v
    q(qq) = [shape(outs[(qq - 1) * K + b]) for b in 1:K]
    Ib, rs, ts, phis, rsgn = q(1), q(2), q(3), q(4), q(5)
    I = zero(Ib[1])
    for b in 1:K
        I .+= Ib[b]
    end
    return (I = I, Ib = Ib, rs = rs, ts = ts, phis = phis, redshift_sign = rsgn,
            steps = shape(outs[end]), orders = collect(ot), plan = plan,
            run = run_record(exec, plan, t0))
end
