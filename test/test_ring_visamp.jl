# AART-style equatorial g-disk source, layered photon-ring images, and visibility amplitudes. Checks:
#   (a) G-FACTOR : Schwarzschild circular-orbit limit g=√(1−3/r) for a λ=0 photon;
#       g→1 at large r; gGas(inside ISCO) ≡ gDisk(outside) at the ISCO junction.
#   (b) PROFILE  : Johnson-SU emissivity positive, integrable, peak at μ (γ=0).
#   (c) LAYERS   : n=1 flux < n=0, n=2 < n=1 (demagnification); each ring confined
#       to its lensing band (nonzero pixels ⊂ band polygon), n≥1 straddles crit curve.
#   (d) VISAMP   : thin ring of diameter d μas → |V| minima spacing matches the
#       analytic 1/(d·rad) periodicity to a few % (the ring ruler);
#   (e) AUTODIFF : ForwardDiff through equatorial_disk_intensity w.r.t. a source knob.

using Test
using BlackLightPlus
using BlackLightPlus: gfactor_disk, gfactor_circular, gfactor_plunge, johnson_su_profile,
    equatorial_disk_intensity, EquatorialDisk, isco_prograde,
    radon_cut, visamp, visamp_minima, tukey_window, MUAS_TO_RAD,
    conserved_quantities, critical_curve, lensing_band, photon_ring_layers
using ForwardDiff

# local winding-number point-in-polygon (mirrors Geodesics._in_polygon) so the
# confinement check does not reach into module internals.
function _inpoly(px, py, xs, ys)
    n = length(xs); wn = 0
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        yi = ys[i]; yj = ys[j]
        if yi <= py
            (yj > py && ((xs[j]-xs[i])*(py-yi) - (px-xs[i])*(yj-yi)) > 0) && (wn += 1)
        else
            (yj <= py && ((xs[j]-xs[i])*(py-yi) - (px-xs[i])*(yj-yi)) < 0) && (wn -= 1)
        end
    end
    return wn != 0
end

@testset "equatorial g-disk source + photon rings + visamps" begin

    # ── CHECK (a): REDSHIFT FACTOR ────────────────────────────────────────────
    @testset "gate a: g-factor limits & ISCO continuity" begin
        # Schwarzschild (a→0) Keplerian, λ=0 photon: g = 1/uᵗ = √(1−3/r) exactly.
        a = 0.0
        for r in (6.0, 10.0, 30.0, 100.0)
            g = gfactor_disk(r, a, 1, 0.0, 0.0)
            @test isapprox(g, sqrt(1 - 3 / r); rtol=1e-10)
        end
        # large radius ⇒ g → 1 (flat space, static emitter limit)
        for a in (0.0, 0.5, 0.94)
            λ, η = conserved_quantities(0.0, 1.0, deg2rad(60.0), a)
            @test isapprox(gfactor_disk(2000.0, a, 1, λ, η), 1.0; atol=2e-3)
        end
        # gDisk/gGas continuity at the ISCO junction (uʳ_plunge → 0 there).
        for a in (0.3, 0.6, 0.94)
            isco = isco_prograde(a)
            λ, η = conserved_quantities(-5.0, 2.0, deg2rad(17.0), a)
            gp = gfactor_disk(isco + 1e-6, a, 1, λ, η)   # circular branch
            gm = gfactor_disk(isco - 1e-6, a, 1, λ, η)   # plunging branch
            @test isapprox(gp, gm; rtol=1e-4)
            # and the branch functions agree in the limit
            @test isapprox(gfactor_circular(isco, a, 1, λ, η),
                           gfactor_plunge(isco, a, 1, λ, η); rtol=1e-6)
        end
        # g is real, finite and positive over the disk
        a = 0.94
        for r in range(isco_prograde(a) + 0.01, 20.0; length=50)
            λ, η = conserved_quantities(-4.0, 3.0, deg2rad(17.0), a)
            g = gfactor_disk(r, a, 1, λ, η)
            @test isfinite(g) && g > 0
        end
    end

    # ── CHECK (b): JOHNSON-SU PROFILE ─────────────────────────────────────────
    @testset "gate b: Johnson-SU emissivity" begin
        rr = collect(range(0.05, 40.0; length=40000))
        # γ = 0 ⇒ unimodal, peak exactly at μ
        for mu in (2.0, 3.5, 6.0)
            p = [johnson_su_profile(r; mu=mu, sigma=0.5, gamma=0.0) for r in rr]
            @test all(>(0), p)                          # strictly positive
            @test isapprox(rr[argmax(p)], mu; atol=2e-3)  # peak at μ
        end
        # AART default (γ=-3/2): positive, integrable (finite normalization), decays
        p = [johnson_su_profile(r; mu=0.66, sigma=0.5, gamma=-1.5) for r in rr]
        @test all(x -> x > 0 && isfinite(x), p)
        integral = sum((p[1:end-1] .+ p[2:end]) ./ 2 .* diff(rr))   # trapezoid
        @test isfinite(integral) && integral > 0
        @test johnson_su_profile(40.0; mu=0.66, sigma=0.5, gamma=-1.5) <
              johnson_su_profile(2.0; mu=0.66, sigma=0.5, gamma=-1.5)   # decays outward
    end

    # ── CHECK (c): LAYERED PHOTON-RING IMAGE ──────────────────────────────────
    @testset "gate c: layered rings — demagnification & band confinement" begin
        a = 0.94
        θo = deg2rad(17.0)
        disk = EquatorialDisk(a)
        cfg = RunConfig(bh_a=a, tracer=:analytic,
                        camera=CameraConfig(inclination_deg=17.0, distance=1e4,
                                            fov=16.0, resolution=64))
        L = photon_ring_layers(cfg; disk=disk, dx=0.08, box=8.0, npts=60)
        F = [sum(L.Ib[b]) for b in 1:3]
        @test F[1] > 0 && F[2] > 0 && F[3] > 0
        @test F[2] < F[1]                               # n=1 demagnified vs n=0
        @test F[3] < F[2]                               # n=2 demagnified vs n=1
        # thinning: successive rings occupy fewer pixels
        npx = [count(>(0), L.Ib[b]) for b in 1:3]
        @test npx[3] < npx[2] < npx[1]
        @info "gate c: layer fluxes / pixels" F demag_ratios=(F[2]/F[1], F[3]/F[2]) npx

        # confinement: every nonzero n≥1 pixel lies inside its lensing band
        N = L.N
        xs = range(-L.box, L.box; length=N)
        for n in (1, 2)
            inner, outer = lensing_band(a, θo, n; npts=60, box=L.box)
            ox = outer[:, 1]; oy = outer[:, 2]
            ix_in = inner[:, 1]; iy_in = inner[:, 2]
            img = L.Ib[n + 1]
            allin = true
            ρmin = Inf; ρmax = 0.0
            for iy in 1:N, ix in 1:N
                img[iy, ix] > 0 || continue
                α = xs[ix]; β = xs[iy]
                inside_band = _inpoly(α, β, ox, oy) && !_inpoly(α, β, ix_in, iy_in)
                inside_band || (allin = false)
                ρ = sqrt(α^2 + β^2)
                ρmin = min(ρmin, ρ); ρmax = max(ρmax, ρ)
            end
            @test allin                                 # ring ⊂ its lensing band
            # n≥1 ring straddles the critical curve in image radius
            αc, βc = critical_curve(a, θo; n=120)
            ρc = sqrt.(αc .^ 2 .+ βc .^ 2)
            @test ρmin < sum(ρc) / length(ρc) < ρmax
        end
    end

    # ── CHECK (d): VISIBILITY-AMPLITUDE RING RULER ────────────────────────────
    @testset "gate d: visamp minima ↔ ring diameter" begin
        # synthetic thin Gaussian ring of angular diameter d μas on an fov-μas grid
        function ring_image(N, fov_muas, d_muas, wsig)
            Δ = fov_muas / N; c = (N + 1) / 2; R = d_muas / 2
            img = zeros(N, N)
            for iβ in 1:N, iα in 1:N
                ρ = sqrt(((iα - c) * Δ)^2 + ((iβ - c) * Δ)^2)
                img[iβ, iα] = exp(-0.5 * ((ρ - R) / wsig)^2)
            end
            img
        end
        N = 512; fov = 160.0; wsig = 1.5
        for d in (30.0, 45.0)
            img = ring_image(N, fov, d, wsig)
            pred = 1 / (d * MUAS_TO_RAD * 1e9)          # Gλ spacing = 1/(d·rad·1e9)
            res = visamp(img; angles=(0.0, 90.0), fov_muas=fov, pad=16, maxbaseline=120.0)
            for r in res
                mins = visamp_minima(r.baselines, r.amplitude; skip=1, bl_min=2.0)
                @test length(mins) >= 5
                sp = diff(mins)
                meansp = sum(sp) / length(sp)
                err = abs(meansp - pred) / pred
                @info "gate d: ring ruler" d angle=r.angle predicted=pred measured=meansp err
                @test err < 0.03                        # within a few %
            end
        end
        # radon_cut sanity: projection of a centred ring is symmetric & non-negative
        img = ring_image(256, 100.0, 30.0, 1.0)
        proj, sax = radon_cut(img, 0.0; fov_muas=100.0)
        @test all(>=(0), proj)
        @test length(proj) == 256
        @test isapprox(proj[1:end], reverse(proj); rtol=0, atol=1e-9 * maximum(proj))
        # Tukey window: unit centre, tapered (zero) ends, symmetric
        w = tukey_window(101, 0.5)
        @test w[51] == 1.0
        @test w[1] < 1e-12 && w[end] < 1e-12
        @test w ≈ reverse(w)
    end

    # ── CHECK (e): AUTODIFF SMOKE ─────────────────────────────────────────────
    @testset "gate e: ForwardDiff through equatorial_disk_intensity" begin
        a = 0.94
        λ, η = conserved_quantities(-4.0, 3.0, deg2rad(17.0), a)
        rs = 4.0
        h = 1e-6
        # w.r.t. each source knob and the spin
        f_sk(sk) = equatorial_disk_intensity(rs, 1, a, λ, η; sub_kep=sk)
        f_mu(mu) = equatorial_disk_intensity(rs, 1, a, λ, η; mu_p=mu)
        f_a(aa) = equatorial_disk_intensity(rs, 1, aa, λ, η)
        for (f, x0) in ((f_sk, 1.0), (f_mu, 0.66), (f_a, a))
            d = ForwardDiff.derivative(f, x0)
            fd = (f(x0 + h) - f(x0 - h)) / (2h)
            @test isfinite(d)
            @test isapprox(d, fd; rtol=1e-5)
        end
    end

    # ── REGRESSION: photon-ring masks must be race-free Bool arrays ──────────
    # photon_ring_layers paints masks[b] under Threads.@threads. A BitMatrix
    # (falses(N,N)) races on packed 64-bit chunks and silently drops bits; a
    # byte-per-element Matrix{Bool} does not. Check the type AND the invariant.
    @testset "gate f: photon-ring masks are race-free" begin
        a = 0.94
        disk = EquatorialDisk(a)
        cfg = RunConfig(bh_a=a, tracer=:analytic,
                        camera=CameraConfig(inclination_deg=17.0, distance=1e4,
                                            fov=16.0, resolution=64))
        L = photon_ring_layers(cfg; disk=disk, dx=0.12, box=6.0, npts=40)
        for b in 1:length(L.bands)
            @test L.masks[b] isa AbstractMatrix{Bool}
            @test !(L.masks[b] isa BitMatrix)
            @test L.masks[b] == (L.Ib[b] .> 0)   # mask exactly tracks painted flux
        end
    end
end

# ── regression: full-grid render_analytic layers stay in their lensing bands ──
# Pre-fix, sn/cn-aliased ghost landings passed the `valid` flag across the whole
# shadow interior: 33% of the n=1 and 96% of the n=2 layer flux fell OUTSIDE the
# layer's lensing band. The Mino-budget validity bound must confine each layer.
@testset "render_analytic layers confined to lensing bands" begin
    a = 0.94; θo = deg2rad(17.0)
    cfg = RunConfig(bh_a=a, tracer=:analytic,
                    camera=CameraConfig(inclination_deg=17.0, distance=1000.0,
                                        fov=24.0, resolution=96))
    out = render_analytic(cfg)
    edges = range(-12.0, 12.0; length=97)
    cent = collect((edges[1:end-1] .+ edges[2:end]) ./ 2)
    for n in (1, 2)
        inner, outer = lensing_band(a, θo, n; npts=100)
        ox = outer[:, 1]; oy = outer[:, 2]
        ixp = inner[:, 1]; iyp = inner[:, 2]
        img = out.Ib[n + 1]
        fin = 0.0; fout = 0.0
        for iy in 1:96, ix in 1:96
            v = img[iy, ix]
            v > 0 || continue
            αB = cent[ix]; β = cent[iy]         # camera image-α == Bardeen α
            inband = _inpoly(αB, β, ox, oy) && !_inpoly(αB, β, ixp, iyp)
            inband ? (fin += v) : (fout += v)
        end
        @test fin > 0
        @test fout < 0.15 * (fin + fout)        # pre-fix: 0.33 (n=1), 0.96 (n=2)
    end
end

# ── parity: fused gpu_analytic kernel (KA CPU backend) == render_analytic ─────
# gpu_analytic.jl is standalone (`using CUDA` loads without touching a device; the
# kernel runs on the KernelAbstractions CPU backend — the test_transfer pattern).
# ONE launch computes every band from a shared per-pixel `_emission_prep`; all
# five per-band maps must match the CPU render bit-for-bit (identical Float64
# closed form on both sides).
@isdefined(_ganalytic_kernel!) || include(joinpath(@__DIR__, "..", "src", "gpu_analytic.jl"))
using KernelAbstractions

@testset "gpu_analytic fused kernel (KA CPU backend) == render_analytic" begin
    a = 0.94; ro = 1000.0; npix = 32; fov = 16.0
    θo = deg2rad(17.0)
    cfg = RunConfig(bh_a=a, tracer=:analytic,
                    camera=CameraConfig(inclination_deg=17.0, distance=ro,
                                        fov=fov, resolution=npix))
    bands = (0, 1, 2); nb = length(bands)
    np2 = npix * npix
    edges = range(-fov / 2, fov / 2; length=npix + 1)
    centres = collect((edges[1:end-1] .+ edges[2:end]) ./ 2)
    for source in (nothing, EquatorialDisk(a))
        ref = render_analytic(cfg; source=source)
        bufs = ntuple(_ -> fill(NaN, np2 * nb), 5)
        disk = source === nothing ? EquatorialDisk(a) : source
        _ganalytic_kernel!(CPU())(bufs..., centres, θo, a, bands, ro, disk,
                                  source !== nothing, npix, 0, np2; ndrange=np2)
        KernelAbstractions.synchronize(CPU())
        maps = (ref.Ib, ref.rs, ref.ts, ref.phis, ref.redshift_sign)
        for bi in 1:nb, (mi, m) in enumerate(maps)
            got = reshape(bufs[mi][(bi-1)*np2+1:bi*np2], npix, npix)
            @test isequal(got, m[bi])           # bitwise, incl. NaN off-disk
        end
    end
end

# ── mirror path: θo > 90° (equatorial-mirror branch of _emission_prep) parity ──
# render_analytic and the fused GPU kernel now paint through the SAME per-pixel seam
# (render.jl::_paint_layer / its device inline); the θo→π−θo, β→−β mirror branch is
# exercised here beyond the 17° case above. Same bitwise
# contract for source=nothing (placeholder) and the g-disk.
@testset "gpu_analytic == render_analytic on the θo>90 mirror path" begin
    a = 0.94; ro = 1000.0; npix = 32; fov = 16.0
    θo = deg2rad(163.0)                      # cos θo < 0 ⇒ mirror branch
    cfg = RunConfig(bh_a=a, tracer=:analytic,
                    camera=CameraConfig(inclination_deg=163.0, distance=ro,
                                        fov=fov, resolution=npix))
    bands = (0, 1, 2); nb = length(bands); np2 = npix * npix
    edges = range(-fov / 2, fov / 2; length=npix + 1)
    centres = collect((edges[1:end-1] .+ edges[2:end]) ./ 2)
    for source in (nothing, EquatorialDisk(a))
        ref = render_analytic(cfg; source=source)
        @test any(>(0), ref.I)               # mirror branch still lands on the disk
        bufs = ntuple(_ -> fill(NaN, np2 * nb), 5)
        disk = source === nothing ? EquatorialDisk(a) : source
        _ganalytic_kernel!(CPU())(bufs..., centres, θo, a, bands, ro, disk,
                                  source !== nothing, npix, 0, np2; ndrange=np2)
        KernelAbstractions.synchronize(CPU())
        maps = (ref.Ib, ref.rs, ref.ts, ref.phis, ref.redshift_sign)
        for bi in 1:nb, (mi, m) in enumerate(maps)
            got = reshape(bufs[mi][(bi-1)*np2+1:bi*np2], npix, npix)
            @test isequal(got, m[bi])           # bitwise, incl. NaN off-disk
        end
    end
end
