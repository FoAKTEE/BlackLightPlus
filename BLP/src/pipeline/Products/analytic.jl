# Exact-Kerr images of an equatorial disk: per screen point the closed-form landing of
# every photon-ring layer and the disk intensity there, on the camera pixels or on the
# lensing bands of the Bardeen screen.

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

# The band maps of per-item SVector outputs: map `k` of band `b` as an `npix × npix` matrix.
band_maps(out::Vector{<:SVector}, b, n) = reshape([o[b] for o in out], n, n)

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
`I = Σ_b Ib[b]`, summed in band order.

Determinism: bitwise per item, whatever the executor, device count or chunking.
"""
function render_analytic(plan::Plan, disk::EquatorialDisk; bands = (0, 1, 2),
                         ro::Real = plan.camera.d, backend = nothing)
    t0 = time_ns()
    exec = resolve(backend)
    bt = map(Int, Tuple(bands))
    prog = AnalyticProgram(plan.pixels, plan.theta_o, analytic_spin(plan, disk), Float64(ro),
                           disk, bt)
    N = npixels(plan)
    outs = ntuple(_ -> Vector{SVector{length(bt),Float64}}(undef, N), 5)
    foreach_item!(exec, prog, plan.queue; outs)
    n = plan.camera.npix
    Ib, rs, ts, phis, rsgn = map(o -> [band_maps(o, b, n) for b in eachindex(bt)], outs)
    I = zeros(n, n)
    for b in eachindex(bt)
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
`n ∈ bands` its own lensing band (`lensing_band(a, θo, n; npts, box)`), whose in-band
grid points alone are traced to their order-`n` landing and painted with the disk
intensity ([`AnalyticProgram`](@ref) over the band's points). Confining each order to
its band keeps the `n ≥ 1` rings thin. The plan's spacetime must be Kerr with the disk's
spin. Returns `(I, Ib, masks, grid, N, box, dx, bands, plan, run)`: per band the `N × N`
image `Ib[b]` (rows `β`) and `masks[b]` (painted and positive), the composite
`I = Σ_b w_b Ib[b]` with `w = 1` for `n = 0` and `fudge` otherwise, and the shared
grid.

Determinism: bitwise per item, whatever the executor, device count or chunking.
"""
function photon_ring_layers(plan::Plan, disk::EquatorialDisk; bands = (0, 1, 2),
                            dx::Real = 0.02, box::Real = 12.0, npts::Integer = 100,
                            ro::Real = plan.camera.d, fudge::Real = 1.0, backend = nothing)
    t0 = time_ns()
    exec = resolve(backend)
    a = analytic_spin(plan, disk)
    θo = plan.theta_o
    kw = (npts = Int(npts), box = Float64(box))
    grid, _, N = band_grid(lensing_band(a, θo, 0; kw...)...; dx = Float64(dx), box = kw.box)
    gen = CoordList(grid[:, 1], grid[:, 2])
    Ib = [zeros(N, N) for _ in bands]
    masks = [fill(false, N, N) for _ in bands]
    for (b, n) in enumerate(bands)
        _, bmask, _ = band_grid(lensing_band(a, θo, Int(n); kw...)...; dx = Float64(dx),
                                box = kw.box)
        items = findall(vec(bmask))
        outs = ntuple(_ -> Vector{SVector{1,Float64}}(undef, N * N), 5)
        prog = AnalyticProgram(gen, θo, a, Float64(ro), disk, (Int(n),))
        foreach_item!(exec, prog, items; outs)
        paint_band!(Ib[b], masks[b], outs[1], outs[2], items, N)
    end
    I = zeros(N, N)
    for (b, n) in enumerate(bands)
        I .+= (n == 0 ? 1.0 : Float64(fudge)) .* Ib[b]
    end
    return (I = I, Ib = Ib, masks = masks, grid = grid, N = N, box = Float64(box),
            dx = Float64(dx), bands = collect(bands), plan = plan, run = run_record(exec, plan, t0))
end

# Paint the valid landings of the band's grid points: point `idx` is pixel
# `[(idx - 1) ÷ N + 1, (idx - 1) % N + 1]` (row β, column α).
function paint_band!(img, mask, Ib, rs, items, N)
    for idx in items
        isfinite(rs[idx][1]) || continue
        ix = (idx - 1) % N + 1
        iy = (idx - 1) ÷ N + 1
        img[iy, ix] = Ib[idx][1]
        mask[iy, ix] = Ib[idx][1] > 0
    end
    return img
end
