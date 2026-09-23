# gpu_analytic.jl — GPU port of the `:analytic` exact-Kerr equatorial tracer
# (render.jl::render_analytic). One pixel per thread direct-inverts the closed-form Kerr
# equatorial geodesic for each photon-ring layer n (JacobiElliptic is device-compatible)
# and paints the equatorial source — no ODE, no GRMHD snapshot. The n-independent
# elliptic machinery (`_emission_prep`) runs once per pixel and every band finishes
# in-register (`_emission_band`), so one launch + one DtoH copy per device covers all
# bands. Every stage is the same Float64 closed form the CPU evaluates, so the image
# agrees with `render_analytic` per pixel to round-off. CUDA is an optional dependency:
# this file is NOT included from BlackLightPlus.jl — callers include it explicitly.

using KernelAbstractions
using CUDA
using BlackLightPlus: equatorial_disk_intensity, EquatorialDisk
using BlackLightPlus.Geodesics: _emission_prep, _emission_band

# ---------------------------------------------------------------- fused kernel
# One thread per pixel in the contiguous global-pixel range [pix_lo+1, pix_lo+count],
# all ring orders n ∈ `bands` (an isbits NTuple). The five per-pixel maps are written as
# separate arrays laid out (pixel, band) column-major — band bi occupies the contiguous
# slice [t + (bi-1)*count] — so the host copies out with straight `copyto!`. The global
# pixel index maps to the column-major image M[iβ, iα], matching render_analytic's store.
# Invalid pixels: Ib = 0, observables NaN (the CPU off-disk sentinel).
@kernel function _ganalytic_kernel!(Ib, rs, ts, phis, rsgn, @Const(centres),
                                    theta_o, a, bands, ro, disk, has_source, npix, pix_lo, count)
    t = @index(Global)
    @inbounds if t <= count
        g = pix_lo + t                 # global pixel (for the screen coords)
        ia = (g - 1) ÷ npix + 1
        ib = (g - 1) % npix + 1
        alphaB = centres[ia]           # camera image-α == Bardeen αB
        betaB = centres[ib]
        prep = _emission_prep(alphaB, betaB, theta_o, a; ro = ro)
        o = 0
        for n in bands
            r, ph, tf, nur, valid, lam, eta = _emission_band(prep, n)
            # per-pixel paint seam (render.jl::_paint_layer): prep is reused across
            # bands, and the source dispatches through the isbits `has_source`/`disk`
            # pair — the CPU's `srcfun` closure is not device-callable.
            if valid
                if has_source
                    iv = equatorial_disk_intensity(disk, r, nur, a, lam, eta)
                else
                    iv = 1.0 / (r * r) # placeholder falloff (valid ⇒ rs > r_h)
                end
                Ib[t + o] = iv; rs[t + o] = r; ts[t + o] = tf; phis[t + o] = ph
                rsgn[t + o] = Float64(nur)
            else
                Ib[t + o] = 0.0; rs[t + o] = NaN; ts[t + o] = NaN; phis[t + o] = NaN
                rsgn[t + o] = NaN
            end
            o += count
        end
    end
end

"""
    render_analytic_gpu(cfg; devices=[0], bands=(0,1,2), ro=cfg.camera.distance,
                        source=nothing, stack_bytes=32*1024)
        -> NamedTuple (I, Ib, rs, ts, phis, redshift_sign, bands)

GPU twin of `render_analytic`: for every camera pixel and every photon-ring layer
`n ∈ bands`, direct-invert the exact-Kerr equatorial geodesic on device and paint the
equatorial source. Returns the same layered NamedTuple as `render_analytic` (per-band
`Ib`, layered `I = Σ_n Ib[n]`, and per-band observables, NaN off the disk). `source` is
`nothing` (the 1/rs^2 placeholder) or an `EquatorialDisk`. One kernel launch per device
covers all bands; pixels are split contiguously across `devices`.
"""
function render_analytic_gpu(cfg; devices = [0], bands = (0, 1, 2),
                             ro::Real = cfg.camera.distance, source = nothing,
                             stack_bytes::Integer = 32 * 1024)
    a = Float64(cfg.bh_a)
    theta_o = deg2rad(Float64(cfg.camera.inclination_deg))
    npix = Int(cfg.camera.resolution)
    fov = Float64(cfg.camera.fov)
    edges = range(-fov / 2, fov / 2; length = npix + 1)
    centres = collect((edges[1:end-1] .+ edges[2:end]) ./ 2)
    rof = Float64(ro)
    np2 = npix * npix

    has_source = source !== nothing
    disk = source === nothing ? EquatorialDisk(a) : source

    bandsT = map(Int, Tuple(bands))    # isbits NTuple for the in-kernel loop
    nb = length(bandsT)
    # host output maps (column-major, matching render_analytic's `Ib[bi][iβ, iα]`)
    Ib = [Matrix{Float64}(undef, npix, npix) for _ in 1:nb]
    rs = [Matrix{Float64}(undef, npix, npix) for _ in 1:nb]
    ts = [Matrix{Float64}(undef, npix, npix) for _ in 1:nb]
    phis = [Matrix{Float64}(undef, npix, npix) for _ in 1:nb]
    rsign = [Matrix{Float64}(undef, npix, npix) for _ in 1:nb]

    nd = length(devices)
    # contiguous per-device pixel split
    ranges = [(div((di - 1) * np2, nd) + 1):(div(di * np2, nd)) for di in 1:nd]

    # Per-device setup. The kernel fully inlines the quartic-root + elliptic machinery
    # into one large device frame; CUDA's default 1 KB per-thread stack overflows it at
    # runtime (ERROR_ILLEGAL_ADDRESS) even though it compiles cleanly, so raise
    # LIMIT_STACK_SIZE. Then upload centres and allocate the 5 output arrays — each
    # cnt*nb long, one contiguous cnt-slice per band.
    centres_d = Dict{Int,CuArray{Float64,1}}()
    outbufs = Dict{Int,NTuple{5,CuArray{Float64,1}}}()
    for (di, dev) in enumerate(devices)
        CUDA.device!(dev)
        CUDA.limit!(CUDA.LIMIT_STACK_SIZE, Int(stack_bytes))
        centres_d[dev] = CuArray(centres)
        cnt = length(ranges[di])
        outbufs[dev] = ntuple(_ -> CuArray{Float64}(undef, cnt * nb), 5)
    end

    @sync for di in 1:nd
        rng = ranges[di]
        isempty(rng) && continue
        dev = devices[di]
        @async begin
            CUDA.device!(dev)
            backend = CUDABackend()
            cnt = length(rng)
            Ibd, rsd, tsd, phd, rgd = outbufs[dev]
            _ganalytic_kernel!(backend)(Ibd, rsd, tsd, phd, rgd, centres_d[dev],
                                        theta_o, a, bandsT, rof, disk, has_source,
                                        npix, first(rng) - 1, cnt; ndrange = cnt)
            KernelAbstractions.synchronize(backend)
            # direct DtoH into the contiguous linear slice (no intermediate Array)
            o = first(rng)
            for bi in 1:nb
                off = (bi - 1) * cnt + 1
                copyto!(Ib[bi], o, Ibd, off, cnt)
                copyto!(rs[bi], o, rsd, off, cnt)
                copyto!(ts[bi], o, tsd, off, cnt)
                copyto!(phis[bi], o, phd, off, cnt)
                copyto!(rsign[bi], o, rgd, off, cnt)
            end
        end
    end

    I = zeros(npix, npix)
    for bi in 1:nb
        @inbounds I .+= Ib[bi]
    end
    return (I = I, Ib = Ib, rs = rs, ts = ts, phis = phis, redshift_sign = rsign,
            bands = collect(bandsT))
end
