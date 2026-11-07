# Exact-Kerr images of emitting surfaces at any polar angle, polarized or not: per screen
# point the closed-form state once, then for every image order and every material the
# crossings of its surface (`image_crossings`), their landings and the Stokes I, Q, U.

"""
    SurfaceProgram(gen, theta_o, a, ro, materials, bands)

One screen point of the exact-Kerr surface tracer: `(α, β) = image_coords(gen, item)`,
`prep = surface_prep(α, β, theta_o, a; ro)` and each material's `surface_terms` once, then
for each order `n` of `bands` (an `NTuple{K,Int}`) and each material of the tuple
`materials` the crossings `image_crossings`, their landings and
`polarized_emission(material, landing, α, β, theta_o)`, summed. Called as
`prog(item, i, outs...)` with the `7K` vectors of [`surface_outputs`](@ref):
`outs[(q - 1) K + b][i]` is quantity `q` of band `b`, the quantities being Stokes I, Q,
U, and the landing radius, emission time, azimuth and radial sign of the first valid
crossing (of the first material that has one; NaN when none).
"""
struct SurfaceProgram{G,M,B} <: Function
    gen::G
    theta_o::Float64
    a::Float64
    ro::Float64
    materials::M
    bands::B
end
Adapt.@adapt_structure SurfaceProgram

"Quantities a [`SurfaceProgram`](@ref) writes per band: I, Q, U, radius, time, azimuth, sign."
const SURFACE_QUANTITIES = 7

"""
    surface_outputs(K, n) -> NTuple{7K,Vector{Float64}}

The output vectors of a [`SurfaceProgram`](@ref) of `K` bands over `n` items,
quantity-major.
"""
surface_outputs(K::Integer, n::Integer) =
    ntuple(_ -> Vector{Float64}(undef, n), SURFACE_QUANTITIES * K)

# Stokes and the first valid landing of one material's crossings of order n; `θs` is the
# material's `surface_terms`.
@inline function paint_material(prep, mat, θs, n, α, β, θo)
    m1, m2, k = image_crossings(prep, θs, n)
    S = (0.0, 0.0, 0.0)
    L = (NaN, NaN, NaN, NaN)
    k == 0 && return S, L
    l = surface_landing(prep, θs, m1)
    if l.valid
        S = map(Float64, polarized_emission(mat, l, α, β, θo))
        L = (Float64(l.r), Float64(l.t), Float64(l.φ), Float64(l.νr))
    end
    k == 1 && return S, L
    l2 = surface_landing(prep, θs, m2)
    l2.valid || return S, L
    S2 = map(Float64, polarized_emission(mat, l2, α, β, θo))
    isnan(L[1]) && (L = (Float64(l2.r), Float64(l2.t), Float64(l2.φ), Float64(l2.νr)))
    return (S[1] + S2[1], S[2] + S2[2], S[3] + S2[3]), L
end

# One band: the materials' Stokes summed in tuple order, the first valid landing; `terms`
# holds each material's `surface_terms`.
@inline function paint_band(prep, mats::Tuple, terms::Tuple, n, α, β, θo)
    S, L = paint_material(prep, first(mats), first(terms), n, α, β, θo)
    return _paint_rest(prep, Base.tail(mats), Base.tail(terms), n, α, β, θo, S, L)
end
@inline _paint_rest(prep, ::Tuple{}, ::Tuple{}, n, α, β, θo, S, L) = (S..., L...)
@inline function _paint_rest(prep, mats::Tuple, terms::Tuple, n, α, β, θo, S, L)
    S2, L2 = paint_material(prep, first(mats), first(terms), n, α, β, θo)
    L = isnan(L[1]) ? L2 : L
    return _paint_rest(prep, Base.tail(mats), Base.tail(terms), n, α, β, θo,
                       (S[1] + S2[1], S[2] + S2[2], S[3] + S2[3]), L)
end

@inline surface_major(maps) = (map(m -> m[1], maps)..., map(m -> m[2], maps)...,
                               map(m -> m[3], maps)..., map(m -> m[4], maps)...,
                               map(m -> m[5], maps)..., map(m -> m[6], maps)...,
                               map(m -> m[7], maps)...)

function (p::SurfaceProgram)(item, i, outs::Vararg{Any,N}) where {N}
    N == SURFACE_QUANTITIES * length(p.bands) ||
        throw(ArgumentError("SurfaceProgram: $N outputs for $(length(p.bands)) bands"))
    α, β = image_coords(p.gen, item)
    prep = surface_prep(α, β, p.theta_o, p.a; ro = p.ro)
    terms = map(m -> surface_terms(prep, surface_angle(m)), p.materials)
    maps = map(n -> paint_band(prep, p.materials, terms, n, α, β, p.theta_o), p.bands)
    store_all!(outs, i, surface_major(maps))
    return nothing
end

stack_bytes(::SurfaceProgram) = 32 * 1024

# The spin of the surface tracer: the plan's Kerr spin, shared by every material.
function analytic_spin(plan::Plan, mats::Tuple)
    st = plan.spacetime
    st isa KerrSchildCartesian ||
        throw(ArgumentError("the exact-Kerr tracer needs a KerrSchildCartesian plan; got " *
                            "$(nameof(typeof(st)))"))
    for m in mats
        material_spin(m) == st.a ||
            throw(ArgumentError("a material's spin $(material_spin(m)) differs from the plan's spin $(st.a)"))
    end
    return Float64(st.a)
end

const SurfaceMaterial = Union{SynchrotronSurface,PolarizedDisk}

"""
    render_analytic(plan, materials; bands = (0, 1, 2), ro = plan.camera.d, backend = nothing)
        -> NamedTuple

The exact-Kerr polarized image of `materials`, a `SynchrotronSurface`, a `PolarizedDisk`
or a tuple of materials (an `EquatorialDisk` among them contributes Stokes I), on the
plan's pixels: for every pixel and order `n ∈ bands` the crossings of each material's
surface ([`SurfaceProgram`](@ref); `ro = Inf` places the observer at infinity with the
regularized emission time). Returns `(I, Q, U, Ib, Qb, Ub, rs, ts, phis, redshift_sign,
bands, plan, run)`: per band `b` the Stokes maps `Ib[b]`, `Qb[b]`, `Ub[b]` (Q > 0 north,
U > 0 north-east, north the projected spin axis, east `−α`) and the landing radius, time,
azimuth and radial sign of the first valid crossing (NaN without one), each
`npix × npix`, and the band sums `I`, `Q`, `U` in band order.

Determinism: bitwise per item, whatever the executor, device count or chunking.
"""
function render_analytic(plan::Plan, materials::Union{SurfaceMaterial,Tuple};
                         bands = (0, 1, 2), ro::Real = plan.camera.d, backend = nothing)
    t0 = time_ns()
    exec = resolve(backend)
    mats = materials isa Tuple ? materials : (materials,)
    bt = map(Int, Tuple(bands))
    prog = SurfaceProgram(plan.pixels, plan.theta_o, analytic_spin(plan, mats), Float64(ro),
                          mats, bt)
    N = npixels(plan)
    K = length(bt)
    outs = surface_outputs(K, N)
    foreach_item!(exec, prog, 1:N; outs)
    n = plan.camera.npix
    Ib, Qb, Ub, rs, ts, phis, rsgn = ntuple(q -> band_maps(outs, q, K, n), SURFACE_QUANTITIES)
    I = zeros(n, n); Q = zeros(n, n); U = zeros(n, n)
    for b in 1:K
        I .+= Ib[b]; Q .+= Qb[b]; U .+= Ub[b]
    end
    return (I = I, Q = Q, U = U, Ib = Ib, Qb = Qb, Ub = Ub, rs = rs, ts = ts, phis = phis,
            redshift_sign = rsgn, bands = collect(bt), plan = plan,
            run = run_record(exec, plan, t0))
end
