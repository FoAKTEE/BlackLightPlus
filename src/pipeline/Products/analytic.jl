# Exact-Kerr images of an equatorial disk: per screen point the closed-form landing of
# every photon-ring layer and the disk intensity there, on the camera pixels or on the
# lensing bands of the Bardeen screen.

"""
    AnalyticProgram(gen, theta_o, a, ro, disk, bands)

One screen point of the exact-Kerr equatorial tracer: `(α, β) = image_coords(gen,
item)` (Bardeen coordinates), `prep = pixel_prep(α, β, theta_o, a; ro)` once, then for
each photon-ring order `n` of `bands` (an `NTuple{K,Int}`) the landing
`band_landing(prep, n)` and the intensity `intensity(disk, landing)`. Called as
`prog(item, i, outs...)` with the `5K` `Float64` vectors of [`analytic_outputs`](@ref):
`outs[(q - 1) K + b][i]` is quantity `q` of band `b`, the quantities being the band
intensity, the landing radius, the time of flight, the azimuth and the emitted
radial-velocity sign; an invalid landing gives intensity 0 and NaN observables. Needs a
32 KiB device stack.
"""
struct AnalyticProgram{G,D,B} <: Function
    gen::G
    theta_o::Float64
    a::Float64
    ro::Float64
    disk::D
    bands::B
end
Adapt.@adapt_structure AnalyticProgram

"Quantities an [`AnalyticProgram`](@ref) writes per band: intensity, radius, time, azimuth, sign."
const ANALYTIC_QUANTITIES = 5

"""
    analytic_outputs(K, n) -> NTuple{5K,Vector{Float64}}

The output vectors of an [`AnalyticProgram`](@ref) of `K` bands over `n` items,
quantity-major: the intensity of bands `1:K`, then their landing radius, time of flight,
azimuth and emitted radial-velocity sign.
"""
analytic_outputs(K::Integer, n::Integer) =
    ntuple(_ -> Vector{Float64}(undef, n), ANALYTIC_QUANTITIES * K)

@inline function paint(prep, n, disk)
    l = band_landing(prep, n)
    l.valid || return (0.0, NaN, NaN, NaN, NaN)
    return (Float64(intensity(disk, l)), Float64(l.r), Float64(l.t), Float64(l.φ), Float64(l.νr))
end

# The painted quantities of every band, quantity-major: the order of the outputs.
@inline quantity_major(maps) =
    (map(m -> m[1], maps)..., map(m -> m[2], maps)..., map(m -> m[3], maps)...,
     map(m -> m[4], maps)..., map(m -> m[5], maps)...)

# `outs[k][i] = vals[k]` for every k, unrolled over the tuples: indexing a tuple of host
# arrays at run time would put the tuple on the heap.
@inline store_all!(::Tuple{}, i, ::Tuple{}) = nothing
@inline function store_all!(outs::Tuple, i, vals::Tuple)
    @inbounds first(outs)[i] = first(vals)
    return store_all!(Base.tail(outs), i, Base.tail(vals))
end

function (p::AnalyticProgram)(item, i, outs::Vararg{Any,N}) where {N}
    N == ANALYTIC_QUANTITIES * length(p.bands) ||
        throw(ArgumentError("AnalyticProgram: $N outputs for $(length(p.bands)) bands"))
    α, β = image_coords(p.gen, item)
    prep = pixel_prep(α, β, p.theta_o, p.a; ro = p.ro)
    maps = map(n -> paint(prep, n, p.disk), p.bands)
    store_all!(outs, i, quantity_major(maps))
    return nothing
end

stack_bytes(::AnalyticProgram) = 32 * 1024

"""
    RadiusProgram(gen, theta_o, a, ro, disk, bands)

The intensity-only program of a steady axisymmetric disk, beside [`AnalyticProgram`](@ref):
`radius_prep` and `band_radius` (no azimuth or time integrals), writing the band intensity,
landing radius and emitted radial sign quantity-major into `3K` outputs; the intensity and
radius equal those of `AnalyticProgram` bit for bit.
"""
struct RadiusProgram{G,D,B} <: Function
    gen::G
    theta_o::Float64
    a::Float64
    ro::Float64
    disk::D
    bands::B
end
Adapt.@adapt_structure RadiusProgram

@inline function paint_radius(prep, n, disk)
    l = band_radius(prep, n)
    l.valid || return (0.0, NaN, NaN)
    return (Float64(intensity(disk, l)), Float64(l.r), Float64(l.νr))
end

function (p::RadiusProgram)(item, i, outs::Vararg{Any,N}) where {N}
    N == 3 * length(p.bands) ||
        throw(ArgumentError("RadiusProgram: $N outputs for $(length(p.bands)) bands"))
    α, β = image_coords(p.gen, item)
    prep = radius_prep(α, β, p.theta_o, p.a; ro = p.ro)
    maps = map(n -> paint_radius(prep, n, p.disk), p.bands)
    store_all!(outs, i, (map(m -> m[1], maps)..., map(m -> m[2], maps)..., map(m -> m[3], maps)...))
    return nothing
end

stack_bytes(::RadiusProgram) = 32 * 1024

# The spin of the analytic tracer: the plan's Kerr spin, which the disk must share.
function analytic_spin(plan::Plan, disk::EquatorialDisk)
    st = plan.spacetime
    st isa KerrSchildCartesian ||
        throw(ArgumentError("the exact-Kerr tracer needs a KerrSchildCartesian plan; got " *
                            "$(nameof(typeof(st)))"))
    disk.a == st.a ||
        throw(ArgumentError("the disk spin $(disk.a) differs from the plan's spin $(st.a)"))
    return Float64(st.a)
end

# The maps of quantity `q` of every band among the outputs `outs` of `K` bands, each an
# `n × n` matrix sharing its output vector.
band_maps(outs, q, K, n) = [reshape(outs[(q - 1) * K + b], n, n) for b in 1:K]

"""
    render_analytic(plan, disk::EquatorialDisk; bands = (0, 1, 2), ro = plan.camera.d,
                    backend = nothing) -> NamedTuple

The exact-Kerr image of the equatorial disk `disk` on the plan's pixels: for every pixel
(its image coordinates are Bardeen's `(α, β)`) and every photon-ring order `n ∈ bands`,
the closed-form equatorial landing (`pixel_prep` once per pixel with the observer at
inclination `plan.theta_o` and radius `ro`, then `band_landing`) and the disk's observed
intensity there ([`AnalyticProgram`](@ref)). The plan's spacetime must be Kerr with the
disk's spin; its matter is not used. Returns `(I, Ib, rs, ts, phis, redshift_sign,
bands, plan, run)`: per band `b` the image `Ib[b]` and the landing radius `rs[b]`, time
of flight `ts[b]`, azimuth `phis[b]` and emitted radial-velocity sign
`redshift_sign[b]` (each `npix × npix`, NaN off the disk; `Ib` zero there), and
`I = Σ_b Ib[b]`, summed in band order. `observables = false` skips the azimuth and time
integrals ([`RadiusProgram`](@ref), about four times faster): `ts` and `phis` are NaN, the
other maps are bit for bit those of the full program. The pixels are visited in their index order, so
every chunk of the executor writes a contiguous run of each map.

Determinism: bitwise per item, whatever the executor, device count or chunking.
"""
function render_analytic(plan::Plan, disk::EquatorialDisk; bands = (0, 1, 2),
                         ro::Real = plan.camera.d, observables::Bool = true, backend = nothing)
    t0 = time_ns()
    exec = resolve(backend)
    bt = map(Int, Tuple(bands))
    a = analytic_spin(plan, disk)
    N = npixels(plan)
    K = length(bt)
    n = plan.camera.npix
    if observables
        outs = analytic_outputs(K, N)
        foreach_item!(exec, AnalyticProgram(plan.pixels, plan.theta_o, a, Float64(ro), disk, bt),
                      1:N; outs)
        Ib, rs, ts, phis, rsgn = ntuple(q -> band_maps(outs, q, K, n), ANALYTIC_QUANTITIES)
    else
        outs = ntuple(_ -> Vector{Float64}(undef, N), 3K)
        foreach_item!(exec, RadiusProgram(plan.pixels, plan.theta_o, a, Float64(ro), disk, bt),
                      1:N; outs)
        Ib, rs, rsgn = ntuple(q -> band_maps(outs, q, K, n), 3)
        ts = [fill(NaN, n, n) for _ in 1:K]; phis = [fill(NaN, n, n) for _ in 1:K]
    end
    I = zeros(n, n)
    for b in 1:K
        I .+= Ib[b]
    end
    return (I = I, Ib = Ib, rs = rs, ts = ts, phis = phis, redshift_sign = rsgn,
            bands = collect(bt), plan = plan, run = run_record(exec, plan, t0))
end

"""
    photon_ring_layers(plan, disk::EquatorialDisk; bands = (0, 1, 2), dx = 0.02, box = 12.0,
                       npts = 100, ro = plan.camera.d, fudge = 1.0, backend = nothing)
        -> NamedTuple

The layered photon-ring image of the equatorial disk on the Bardeen screen: the grid of
spacing `dx` over `[-box, box]²` of lensing band 0 (`band_grid`), and for each order
`n ∈ bands` its own lensing band (`lensing_band(a, θo, n; npts, box)`), whose in-band grid
points alone are traced to their order-`n` landing and painted with the disk
intensity ([`RadiusProgram`](@ref) over the band's points). Confining each order to
its band keeps the `n ≥ 1` rings thin. `dx` may be a tuple with one spacing per band (AART's
per-order resolution): a band with a finer spacing than the first is traced on its own grid
(`band_points`, enumerated within the band, so the cost follows the band's area) and its
image deposited on the common grid, each fine node weighted by the ratio of the fine to the
common cell area, which keeps the ring flux of the fine grid. The plan's spacetime
must be Kerr with the disk's spin. Returns `(I, Ib, masks, grid, N, box, dx, dxs, points,
bands, plan, run)`: per band the `N × N` image `Ib[b]` (rows `β`) on the common grid of
spacing `dx = dxs[1]` and `masks[b]` (painted and positive), the composite
`I = Σ_b w_b Ib[b]` with `w = 1` for `n = 0` and `fudge` otherwise, the shared grid, and per
band its fine samples `points[b] = (α, β, I)` (empty for a band on the common grid).

Determinism: bitwise per item, whatever the executor, device count or chunking.
"""
function photon_ring_layers(plan::Plan, disk::EquatorialDisk; bands = (0, 1, 2),
                            dx = 0.02, box::Real = 12.0, npts::Integer = 100,
                            ro::Real = plan.camera.d, fudge::Real = 1.0, backend = nothing)
    t0 = time_ns()
    exec = resolve(backend)
    a = analytic_spin(plan, disk)
    θo = plan.theta_o
    dxs = dx isa Real ? ntuple(_ -> Float64(dx), length(bands)) : map(Float64, Tuple(dx))
    length(dxs) == length(bands) ||
        throw(ArgumentError("photon_ring_layers: $(length(dxs)) spacings for $(length(bands)) bands"))
    kw = (npts = Int(npts), box = Float64(box))
    dx0 = dxs[1]
    grid, _, N = band_grid(lensing_band(a, θo, 0; kw...)...; dx = dx0, box = kw.box)
    gen = CoordList(grid[:, 1], grid[:, 2])
    Ib = [zeros(N, N) for _ in bands]
    masks = [fill(false, N, N) for _ in bands]
    points = [(α = Float64[], β = Float64[], I = Float64[]) for _ in bands]
    for (b, n) in enumerate(bands)
        band = lensing_band(a, θo, Int(n); kw...)
        if dxs[b] == dx0
            _, bmask, _ = band_grid(band...; dx = dx0, box = kw.box)
            items = findall(vec(bmask))
            outs = ntuple(_ -> Vector{Float64}(undef, N * N), 3)
            prog = RadiusProgram(gen, θo, a, Float64(ro), disk, (Int(n),))
            foreach_item!(exec, prog, items; outs)
            paint_band!(Ib[b], masks[b], outs[1], outs[2], items, N)
        else
            pts = band_points(band...; dx = dxs[b], box = kw.box)
            m = length(pts.α)
            outs = ntuple(_ -> Vector{Float64}(undef, m), 3)
            prog = RadiusProgram(CoordList(pts.α, pts.β), θo, a, Float64(ro), disk, (Int(n),))
            foreach_item!(exec, prog, 1:m; outs)
            points[b] = (α = pts.α, β = pts.β, I = outs[1])
            bin_band!(Ib[b], masks[b], pts, outs[1], pts.N, N)
        end
    end
    I = zeros(N, N)
    for (b, n) in enumerate(bands)
        I .+= (n == 0 ? 1.0 : Float64(fudge)) .* Ib[b]
    end
    return (I = I, Ib = Ib, masks = masks, grid = grid, N = N, box = Float64(box),
            dx = dx0, dxs = collect(dxs), points = points, bands = collect(bands), plan = plan,
            run = run_record(exec, plan, t0))
end

# A fine band image deposited on the common N × N grid: each fine node adds its value
# times the ratio of the fine to the common cell area, (h_f/h_c)² with h = 2 box/(n − 1), to
# the cell of its nearest common node, so a cell holds the mean intensity over its area and
# the sum Σ I h² of every band is the fine one.
function bin_band!(img, mask, pts, vals, Nf, N)
    w = ((N - 1) / (Nf - 1))^2
    cell(k) = clamp(round(Int, (k - 1) * (N - 1) / (Nf - 1)) + 1, 1, N)
    for (j, v) in enumerate(vals)
        (isfinite(v) && v > 0) || continue
        img[cell(pts.iy[j]), cell(pts.ix[j])] += w * v
    end
    mask .= img .> 0
    return img
end

# Paint the valid landings of the band's grid points: point `idx` is pixel
# `[(idx - 1) ÷ N + 1, (idx - 1) % N + 1]` (row β, column α).
function paint_band!(img, mask, Ib, rs, items, N)
    for idx in items
        isfinite(rs[idx]) || continue
        ix = (idx - 1) % N + 1
        iy = (idx - 1) ÷ N + 1
        img[iy, ix] = Ib[idx]
        mask[iy, ix] = Ib[idx] > 0
    end
    return img
end
