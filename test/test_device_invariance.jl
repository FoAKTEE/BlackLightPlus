# Device-count invariance of the multi-device drivers (results/blp/multigpu/dag.md,
# MG03, MG07). Every imaging driver renders on nd = 1, 2, 3, 5 logical devices on the
# KernelAbstractions CPU backend (HostDevice): nd = 1 through the plain `devices` list,
# nd > 1 through a DevicePool with 2 workers per device and 7-ray chunks, so the 100 rays
# (10²) form 15 chunks that the workers pull in whatever order they free up. Each ray is
# one thread with no cross-ray reduction and lands at its own index, so every image is
# bitwise identical across nd, queue order, chunk size and worker timing.
#
# Monte Carlo: the superphoton streams and the reduced spectrum are independent of the
# device (partition) count, and a scattering run is reproducible run to run (findings
# C1-C3 of results/blp/multigpu/design.md §3, resolved by MG05/MG06):
#   C1  a child's stream is keyed by its genealogy (`mc_child_rng`), not by the atomic
#       buffer slot that stores it;
#   C2  device shares are contiguous global-id ranges of one seed: the stream set is the
#       same for any device count, with no key reuse at five devices;
#   C3  spectra and diagnostics are exact integer-limb sums (exact_sum.jl), and CPU
#       child keys do not depend on the chunk: bitwise independent of the chunk count.
# The device scatter driver runs on HostDevices (KA CPU backend) always, and on CUDA
# devices when CUDA is functional.
@isdefined(DI_NDEV) || include(joinpath(@__DIR__, "device_decks.jl"))
using BlackLightPlus: SphereModel, ScatteringSphere, mc_spectrum, mc_spectrum_scatter
using BlackLightPlus: HotcrossTable
using BlackLightPlus.Constants: SIGMA_THOMSON
using BlackLightPlus.MonteCarlo: NW, NT

# Render with nd = 1, 2, 3, 5 logical devices; assert nonzero (and, unless the output
# carries NaN sentinels, finite) output and bitwise equality of every nd > 1 image with
# the nd = 1 image.
function di_check(render; finite = true)
    ref = render(di_devices(1))
    b1 = di_bits(ref)
    fl = reinterpret(Float64, b1)
    finite && @test all(isfinite, fl)
    @test count(x -> isfinite(x) && !iszero(x), fl) > 0
    for nd in DI_NDEV[2:end]
        @test di_bits(render(di_devices(nd))) == b1
    end
    return ref
end

@testset "device-count invariance (KA CPU backend, HostDevice)" begin

@testset "imaging: bitwise identical for 1, 2, 3, 5 devices" begin
    cfg = di_cfg()
    fg = di_uniform()
    snap = di_amr()

    @testset "fused unpolarized (:f64)" begin
        di_check(d -> render_gpu_fused(cfg, fg; devices = d))
    end
    @testset "fused unpolarized (:mixed, rescue pool split)" begin
        fr = Ref(0.0)
        di_check(d -> render_gpu_fused(cfg, fg; devices = d, precision = :mixed,
                                       rescued_frac = fr))
        @test fr[] > 0                             # the rescue sweep ran
    end
    @testset "fused K-variant" begin
        ms = [cfg.mass_scale_cgs, 2.0 * cfg.mass_scale_cgs]
        di_check(d -> render_gpu_fused_kvariant(cfg, fg, ms; devices = d))
    end
    @testset "fused forward-mode AD" begin
        di_check(d -> render_gpu_fused_grad(cfg, fg, :mass_scale; devices = d))
    end
    @testset "fused polarized" begin
        di_check(d -> render_gpu_polarized(cfg, fg; devices = d))
    end
    @testset "native AMR unpolarized" begin
        di_check(d -> render_gpu_amr(cfg, snap; devices = d))
    end
    @testset "native AMR polarized" begin
        di_check(d -> render_gpu_amr_polarized(cfg, snap; devices = d))
    end
    @testset "generic spacetime (superposed binary), unpolarized + polarized" begin
        st = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
        di_check(d -> render_gpu_spacetime(cfg, st, snap; devices = d))
        di_check(d -> render_gpu_spacetime(cfg, st, snap; devices = d, polarized = true))
        cam = PlaneParallelCamera(60.0, 1000.0, 24.0, 10)
        di_check(d -> trace_shadow_gpu(st, camera_rays(st, cam); C = 40.0, escape = 1100.0,
                                       nmax = 3000, devices = d))
    end
    @testset "slow light (uniform and native AMR, I and IQUV)" begin
        fgB = scale_grid_density(fg, 1.3)
        win(d) = upload_window(fg, fgB, 0.0, 30.0, d)
        di_check(d -> render_gpu_fused_slow(cfg, win(d); devices = d, t_obs = 15.0))
        di_check(d -> render_gpu_polarized_slow(cfg, win(d); devices = d, t_obs = 15.0))
        snapB = di_amr(scale = 1.3f0)
        awin(d) = DeviceAMRWindow(upload_amr(snap, d; upload_bounds = false),
                                  upload_amr(snapB, d; upload_bounds = false), 0.0, 1 / 30)
        di_check(d -> render_gpu_amr_slow(cfg, awin(d); devices = d, t_obs = 15.0))
        di_check(d -> render_gpu_amr_polarized_slow(cfg, awin(d); devices = d, t_obs = 15.0))
    end
    @testset "stiffness telemetry, two-phase tracer, analytic" begin
        di_check(d -> stiffness_telemetry(cfg, fg; devices = d))
        cam = PlaneParallelCamera(60.0, 1000.0, 24.0, 10)
        rays = camera_rays(KerrSchildCartesian(0.9), cam)
        di_check(d -> trace_rays_gpu(KerrSchildCartesian(0.9), rays; devices = d, chunk = 9))
        di_check(d -> render_analytic_gpu(di_cfg(res = 8); devices = d); finite = false)
    end
end

@testset "pool: queue order, chunking and worker timing leave images bitwise unchanged" begin
    cfg = di_cfg()
    fg = di_uniform()
    snap = di_amr()
    ref = render_gpu_fused(cfg, fg; devices = di_hosts(1))
    refp = render_gpu_amr_polarized(cfg, snap; devices = di_hosts(1))
    rev = collect(100:-1:1)
    @test render_gpu_fused(cfg, fg; devices = DI_POOLS[3], order = rev) == ref
    @test di_bits(render_gpu_amr_polarized(cfg, snap; devices = DI_POOLS[2], order = rev)) ==
          di_bits(refp)
    # random pools (1-5 devices, 1-3 workers each, 1-40 rays per chunk), random queue
    # order, and a random delay before every chunk launch
    rng = Random.Xoshiro(20260923)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam)
    a = cfg.bh_a; r_h = 1 + sqrt(1 - a * a)
    for trial in 1:6
        pool = DevicePool(di_hosts(rand(rng, 1:5)); lanes = rand(rng, 1:3),
                          chunk = rand(rng, 1:40))
        dg = upload_grid(fg, pool)
        launch = _fused_launch(dg, _gimg_params(cfg), a, r_h, 80.0, 1100.0,
                               sqrt(3.0) * abs(dg.xo), cfg.max_steps, nothing)
        delays = rand(rng, 100) .* 0.004
        delayed = (lane, outs, y0d, n) -> (sleep(delays[n]); launch(lane, outs, y0d, n))
        img = Vector{Float64}(undef, length(rays))
        march!((img,), pool, [MarchPart(Random.randperm(rng, length(rays)), HostRays(rays),
                                        delayed)])
        @test reshape(img, size(ref)) == ref
        close(pool)
    end
end

# ------------------------------------------------------------------ Monte Carlo
@testset "MC: GPU driver stream assignment (_mc_device_shares)" begin
    seed = 0x5ca77e5
    N = 1001
    # the Philox (key, counter-word) identity of every launched primary stream
    streams(nd) = Set((r.key, r.sid) for s in _mc_device_shares(N, nd)
                      for r in (BlackLightPlus.MonteCarlo.mc_primary_rng(seed, g)
                                for g in s.first:(s.first + s.n - 1)))
    for nd in 1:5
        sh = _mc_device_shares(N, nd)
        @test sum(s.n for s in sh) == N           # every superphoton launched once
        @test length(streams(nd)) <= N
    end
    @test _mc_device_shares(N, 1)[1] == (first = 1, n = N)   # devices=[d] == single device
    # the primary stream set does not depend on the device count
    @test streams(2) == streams(1)                # C2
    @test streams(3) == streams(1)                # C2
    # no two superphotons share a stream, for any device count
    for nd in 2:4
        @test length(streams(nd)) == N
    end
    @test length(streams(5)) == N                 # C2: no key reuse at five devices
end

# CPU MC: `nthreads` sets the chunk count, the CPU analogue of the device count.
@testset "MC: CPU chunk-count invariance and reproducibility" begin
    sph = SphereModel()
    thin(k) = mc_spectrum(sph, 20_000; nthreads = k, seed = 0xC0FFEE, n_ebins = 60)
    t1 = thin(1)
    for k in (2, 3)
        tk = thin(k)
        @test tk.count == t1.count                # same superphotons, integer bins
        # bitwise spectrum: exact per-chunk sums merged by integer addition
        @test di_bits(tk.sumw) == di_bits(t1.sumw)            # C3
    end

    tbl = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))   # Thomson table
    ssph = ScatteringSphere()
    scat(k) = mc_spectrum_scatter(ssph, tbl, 5_000; nthreads = k, seed = 0xBEEF,
                                  n_ebins = 60, max_scatt = 2)
    s2a = scat(2); s2b = scat(2)
    @test di_bits(s2a.total.sumw) == di_bits(s2b.total.sumw)  # run-to-run, fixed chunks
    @test s2a.n_children == s2b.n_children
    s1 = scat(1)
    @test s1.n_children > 0                       # scattering is exercised
    for k in (2, 3)
        # identical children and spectrum for any chunk count
        @test di_bits(scat(k).total.sumw) == di_bits(s1.total.sumw)   # C3
    end
end

@testset "MC: GPU scatter reproducibility and device split" begin
    tbl = HotcrossTable(fill(log10(SIGMA_THOMSON), NW + 1, NT + 1))
    # the device kernels on the KA CPU backend (HostDevice), always run
    hrun(nd) = mc_spectrum_gpu(ScatteringSphere(), tbl, 5_000; devices = di_devices(nd),
                               seed = 0xBEEF, n_ebins = 60, max_scatt = 2)
    a = hrun(1); b = hrun(1)
    @test a.n_children > 0
    @test di_bits(a.total.sumw) == di_bits(b.total.sumw)                  # C1: run to run
    @test di_bits(hrun(2).total.sumw) == di_bits(a.total.sumw)            # C1, C2: 1 vs 2
    if CUDA.functional()
        run(devs) = mc_spectrum_gpu(ScatteringSphere(), tbl, 200_000; devices = devs,
                                    seed = 0xBEEF, max_scatt = 2)
        g = run([0])
        @test di_bits(g.total.sumw) == di_bits(run([0]).total.sumw)       # C1 on CUDA
        if length(CUDA.devices()) >= 2
            @test di_bits(run([0, 1]).total.sumw) == di_bits(g.total.sumw)  # C1, C2 on CUDA
        else
            @test_skip false                      # needs two CUDA devices
        end
    else
        @test_skip false                          # CUDA run to run: needs a CUDA device
        @test_skip false                          # CUDA 1 vs 2: needs two CUDA devices
    end
end

end
