# Slow-light (time-dependent snapshot series) tests. Slabs: (a) the fast-light
# limit (identity series / inv_dt=0) reproduces the single-snapshot sample/render
# bit-for-bit; (b) sampled primitives are the exact linear-in-time blend of the
# bracketing dumps, matching an analytic sine to Float32 storage precision; (c)
# bracket selection is causal/monotonic along a backward-traced ray (X⁰ decreases
# camera→source, t_emit = t_obs + X⁰); (d) movies modulate the flux and their
# period average matches the static image to ~3%; (e-k) GPU/KA kernels vs the CPU
# march (corr > 0.999) plus regression pins (dump-time axis, blend short-circuits,
# order/wg invariance). GPU slabs need CUDA + BLP_TEST_GRMHD and are skipped otherwise.

using Test, StaticArrays, Statistics, Printf
using BlackLightPlus
using BlackLightPlus: trace_rays_sampled
using BlackLightPlus.Simulation: AthenaKSnapshot, sample_primitives, sample_primitives_t,
    SnapshotSeries, synthetic_series, scale_density, bracketing_pair, get_frame, units,
    pgas_from_eint

const REPO = normpath(joinpath(@__DIR__, ".."))
# GPU kernels are only needed by the GPU slabs (a-gpu, d, e). Skip loading them —
# and CUDA — when BLP_SKIP_GPU is set, so the CPU slabs (a, b, c) run without a
# CUDA install.
if !haskey(ENV, "BLP_SKIP_GPU")
    include(joinpath(REPO, "src", "gpu_image.jl"))   # UniformFluidGrid + fused (+ slow) kernels
    include(joinpath(REPO, "src", "gpu_pol.jl"))     # polarized (+ slow)
    include(joinpath(REPO, "src", "gpu_amr.jl"))     # direct-AMR (+ slow)
end

const DATA = get(ENV, "BLP_TEST_GRMHD", get(ENV, "BLP_DATA",
    "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no/torus.mhd_w_bcc.00999.athdf"))
const RES = parse(Int, get(ENV, "BLP_RES", "48"))
const NG  = parse(Int, get(ENV, "BLP_NG", "128"))
const BOX = parse(Float64, get(ENV, "BLP_BOX", "40"))
const GPU_OK = !haskey(ENV, "BLP_SKIP_GPU") && CUDA.functional() && isfile(DATA)

corr(a, b) = begin
    va = vec(Float64.(a)); vb = vec(Float64.(b))
    ma = mean(va); mb = mean(vb)
    n = sum((va .- ma) .* (vb .- mb))
    d = sqrt(sum(abs2, va .- ma) * sum(abs2, vb .- mb))
    d == 0 ? 0.0 : n / d
end

# One-meshblock synthetic snapshot on [-4,4]^3 (spatially varying primitives) for the
# CPU sampler slabs — fast and independent of the multi-GB deck.
function tiny_snap(; rho0 = 2.0)
    ni = nj = nk = 4; nmb = 1
    data = Array{Float32}(undef, ni, nj, nk, nmb, 8)
    for k in 1:nk, j in 1:nj, i in 1:ni
        data[i, j, k, 1, 1] = rho0 + 0.1 * (i + j + k)      # dens
        data[i, j, k, 1, 2] = 0.01 * i                      # velx
        data[i, j, k, 1, 3] = 0.02 * j
        data[i, j, k, 1, 4] = 0.03 * k
        data[i, j, k, 1, 5] = 0.5 + 0.05 * i                # eint
        data[i, j, k, 1, 6] = 0.10; data[i, j, k, 1, 7] = 0.20; data[i, j, k, 1, 8] = 0.30
    end
    AthenaKSnapshot(data, [SVector(-4.0, -4.0, -4.0)], [SVector(4.0, 4.0, 4.0)],
                    [SVector(-3.0, -3.0, -3.0)], [SVector(2.0, 2.0, 2.0)], [0],
                    reshape([0, 0, 0], 3, 1),
                    ["dens", "velx", "vely", "velz", "eint", "bcc1", "bcc2", "bcc3"],
                    ni, nj, nk, nmb)
end

# CPU slow-light UNPOLARIZED march on two uniform host grids, blended with the SAME
# per-segment emission-time weight the GPU kernel uses:
#   w = clamp((t0 + X⁰)*inv_dt, 0, 1),  t0 = t_obs - t_a  (t_emit = t_obs + X⁰).
# For the GPU-vs-CPU same-fluid check (mirrors bench_gpu_pol :: cpu_march_uniform).
function cpu_fused_slow(cfg, hostA, hostB, ng, xo, inv_dx, t_a, inv_dt; t_obs)
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam)
    states, dls = trace_rays_sampled(st, rays; C = 80.0, escape = 1.1 * cfg.camera.distance)
    u = units(cfg.m_bh_cgs, cfg.mass_scale_cgs); p = cfg.plasma
    scc = SimCoeffConfig(bh_a = cfg.bh_a, d_unit = u.dens_unit,
                         e_unit = u.dens_unit * BlackLightPlus.Constants.CL^2, b_unit = u.B_unit,
                         plasma_mu = p.mu, plasma_ne_ni = p.ne_ni, plasma_use_p = p.use_p ? 1 : 0,
                         plasma_rat_low = p.rat_low, plasma_rat_high = p.rat_high, plasma_gamma = p.gamma,
                         plasma_gamma_i = p.gamma_i, plasma_gamma_e = p.gamma_e,
                         plasma_thermal_frac = p.thermal_frac, plasma_power_frac = p.power_frac,
                         plasma_kappa_frac = p.kappa_frac, cut_sigma_max = p.sigma_cut,
                         cut_theta_e_min = p.theta_e_min)
    γ = p.gamma; ν = cfg.frequency_hz; Lu = u.L_unit
    t0 = t_obs - t_a                                  # -X⁰ at which t_emit reaches t_a
    sampler = x -> begin
        w = clamp((t0 + x[1]) * inv_dt, 0.0, 1.0)     # w at emission time t_obs + X⁰
        sA = _gimg_sample(hostA, ng, xo, inv_dx, x[2], x[3], x[4])
        prim = w == 0.0 ? sA :
               (1.0 - w) .* sA .+ w .* _gimg_sample(hostB, ng, xo, inv_dx, x[2], x[3], x[4])
        rho = Float64(prim[1]); eint = Float64(prim[5])
        (rho = rho, pgas = pgas_from_eint(eint, γ),
         uu = (Float64(prim[2]), Float64(prim[3]), Float64(prim[4])),
         bb = (Float64(prim[6]), Float64(prim[7]), Float64(prim[8])), inside = rho > 0.0)
    end
    n = length(rays)
    img = Vector{Float64}(undef, n)
    Threads.@threads for k in 1:n
        img[k] = BlackLightPlus.Radiation.unpolarized_march(st, scc, sampler, states[k], dls[k], ν, Lu)
    end
    return reshape(img, cam.npix, cam.npix)
end

function bench_cfg()
    m_bh = 6.2e9 * 1.989e33
    L = BlackLightPlus.Constants.GNEWT * m_bh / BlackLightPlus.Constants.CL^2
    mass_scale = 1.0e-21 * L^3
    RunConfig(bh_a = 0.9, m_bh_cgs = m_bh, mass_scale_cgs = mass_scale, frequency_hz = 2.3e11,
              camera = CameraConfig(inclination_deg = 60.0, distance = 1000.0, fov = 30.0, resolution = RES),
              plasma = PlasmaConfig(rat_high = 10.0, rat_low = 1.0, gamma = 13.0 / 9.0,
                                    use_p = true, sigma_cut = 1.0, theta_e_min = -1.0))
end

@testset "slow light" begin

    # -------------------------------------------------------- (a) CPU fast-light limit
    @testset "a: fast-light limit (CPU, bit-identical)" begin
        snap = tiny_snap()
        x = SVector(0.5, -0.3, 1.1)
        ser = synthetic_series(snap; times = [0.0, 10.0, 25.0])   # identity evolve
        base = sample_primitives(snap, SVector(0.0, x...))
        for t in (-5.0, 0.0, 4.2, 10.0, 17.3, 25.0, 99.0)
            s = sample_primitives_t(ser, SVector(t, x...))
            @test s.rho === base.rho
            @test s.eint === base.eint
            @test s.uu === base.uu
            @test s.bb === base.bb
            @test s.inside === base.inside
        end
    end

    # -------------------------------------------------------- (b) linearity
    @testset "b: linearity (exact two-frame blend + analytic sine)" begin
        snap = tiny_snap()
        x = SVector(0.5, -0.3, 1.1)
        eps = 0.2; omega = 2π / 40; ts = [0.0, 10.0]
        ser = synthetic_series(snap; times = ts, evolve = (s, t) -> scale_density(s, 1 + eps * sin(omega * t)))
        A = get_frame(ser, 1); B = get_frame(ser, 2)
        rhoA = sample_primitives(A, SVector(0.0, x...)).rho
        rhoB = sample_primitives(B, SVector(0.0, x...)).rho
        rho0 = sample_primitives(snap, SVector(0.0, x...)).rho
        fA = 1 + eps * sin(omega * ts[1]); fB = 1 + eps * sin(omega * ts[2])
        maxrel = 0.0
        for t in range(0.0, -10.0; length = 9)
            ia, w = bracketing_pair(ser, ser.t_obs + t)   # dump bracket at the EMISSION time t_obs + X⁰
            @test ia == 1
            s = sample_primitives_t(ser, SVector(t, x...))
            @test s.rho == (1.0 - w) * rhoA + w * rhoB          # EXACT linear blend
            @test s.eint == sample_primitives(snap, SVector(0.0, x...)).eint  # only dens modulated
            approx = ((1.0 - w) * fA + w * fB) * rho0            # analytic sine form
            maxrel = max(maxrel, abs(s.rho - approx) / approx)
        end
        @test maxrel < 1e-6                                      # Float32 storage precision
        @printf("  [b] analytic-sine max rel err = %.2e (Float32 storage)\n", maxrel)
        # clamp behaviour
        @test bracketing_pair(ser, -100.0) == (1, 0.0)
        @test bracketing_pair(ser,  100.0) == (1, 1.0)
    end

    # -------------------------------------------------------- (c) causality / monotonic
    @testset "c: causality / monotonic bracket along a backward ray" begin
        st = KerrSchildCartesian(0.9)
        cam = PlaneParallelCamera(60.0, 1000.0, 30.0, 8)
        rays = camera_rays(st, cam)
        states, _ = trace_rays_sampled(st, rays[30:30]; C = 80.0, escape = 1100.0)
        ys = states[1]                                   # stored source->camera
        # kernel march order is camera->source == reverse(ys)
        x0_march = [ys[i][1] for i in length(ys):-1:1]
        @test x0_march[1] == 0.0                         # camera at X⁰ = 0
        dif = diff(x0_march)
        @test all(<=(0.0), dif)                          # X⁰ monotonic (decreases camera→source)
        @test x0_march[end] < -100.0                     # reaches the source in the past cone
        # a dump series spanning the ray's emission times t_obs + X⁰ (t_obs = 0): bracket
        # indices never advance along the march (deeper samples were emitted earlier)
        thi = x0_march[end]
        times = collect(range(thi - 10.0, 10.0; length = 6))
        ser = synthetic_series(tiny_snap(); times = times, t_obs = 0.0)
        ias = [bracketing_pair(ser, ser.t_obs + t)[1] for t in x0_march]
        @test issorted(ias; rev = true)                  # non-increasing camera→source (causal)
        @printf("  [c] X⁰: 0 → %.1f over %d samples; brackets %d → %d (monotonic)\n",
                thi, length(x0_march), ias[1], ias[end])
    end

    # -------------------------------------------------------- GPU slabs
    if !GPU_OK
        # `CUDA` is only loaded when the GPU includes ran (BLP_SKIP_GPU unset); guard it.
        @info "GPU slabs (a-gpu, d, e) SKIPPED (need a free CUDA device + BLP_TEST_GRMHD)" GPU=(!haskey(ENV, "BLP_SKIP_GPU") && CUDA.functional()) DATA=isfile(DATA)
    else
        cfg = bench_cfg()
        snap = AthenaKSnapshot(DATA)
        base_ug = resample_fluid_uniform(snap, BOX, NG)
        devs = [0]

        # ---------------------------------------------------- (a-gpu) fast-light limit (bit-identical)
        @testset "a-gpu: fast-light limit reproduces fast render bit-for-bit" begin
            fast = render_gpu_fused(cfg, base_ug; devices = devs)
            # a DIFFERENT frame B, but Δt→∞ (inv_dt=0) => w≡0 => B ignored, == fast render
            fgB = scale_grid_density(base_ug, 1.3)
            win0 = upload_window(base_ug, fgB, 0.0, Inf, devs)   # inv_dt = 0
            slow0 = render_gpu_fused_slow(cfg, win0; devices = devs)
            @test slow0 == fast                                  # BIT-IDENTICAL
            # polarized fast-light limit
            pfast = render_gpu_polarized(cfg, base_ug; devices = devs)
            pslow0 = render_gpu_polarized_slow(cfg, win0; devices = devs)
            @test pslow0.I == pfast.I && pslow0.Q == pfast.Q && pslow0.U == pfast.U && pslow0.V == pfast.V
            # direct-AMR fast-light limit
            dg = upload_amr(snap, devs; upload_bounds = false)
            afast = render_gpu_amr(cfg, dg; devices = devs)
            winA0 = DeviceAMRWindow(dg, dg, 0.0, 0.0)            # inv_dt = 0
            aslow0 = render_gpu_amr_slow(cfg, winA0; devices = devs)
            @test aslow0 == afast                                # BIT-IDENTICAL
            @printf("  [a-gpu] fused/pol/amr slow(inv_dt=0) == fast: bit-identical ✓\n")
        end

        # ---------------------------------------------------- (e) GPU vs CPU (same fluid)
        @testset "e: GPU slow-light vs CPU slow-light (same fluid)" begin
            # realistic ascending dump times bracketing the EMISSION times t_obs + X⁰
            # (emitting X⁰ ~ -(950-1100) at camera distance 1000, t_obs = t_b default)
            t_a, t_b = 12000.0, 13200.0
            fgA = scale_grid_density(base_ug, 1.10)
            fgB = scale_grid_density(base_ug, 0.90)
            win = upload_window(fgA, fgB, t_a, t_b, devs)
            gpu = render_gpu_fused_slow(cfg, win; devices = devs)
            cpu = cpu_fused_slow(cfg, fgA.host, fgB.host, base_ug.n, base_ug.xo, base_ug.inv_dx,
                                 t_a, win.inv_dt; t_obs = _win_t_obs(win))
            ci = corr(gpu, cpu)
            @printf("  [e] corr(I_gpu, I_cpu) = %.5f  flux_gpu=%.4e flux_cpu=%.4e\n",
                    ci, sum(gpu), sum(cpu))
            @test ci > 0.999
            @test isapprox(sum(gpu), sum(cpu); rtol = 1e-2)
        end

        # ---------------------------------------------------- (d) end-to-end movie + period average
        @testset "d: time-modulated flux; period-average == static" begin
            static = render_gpu_fused(cfg, base_ug; devices = devs)      # fast-light / base fluid
            t_a, t_b = 920.0, 1080.0                                     # window over emitting region
            # omega so A/B differ by a π/2 phase over Δt=160 (real along-ray gradient WITHOUT
            # the mid-window cancellation of an anti-phase choice): net flux modulation ~ eps.
            eps = 0.15; omega = (π / 2) / (t_b - t_a)
            K = 8                                                        # phases over one period
            imgs = Vector{Matrix{Float64}}(undef, K)
            fluxes = Vector{Float64}(undef, K)
            for jph in 0:K-1
                phi = 2π * jph / K
                fA = 1 + eps * sin(omega * t_a + phi)
                fB = 1 + eps * sin(omega * t_b + phi)
                win = upload_window(scale_grid_density(base_ug, fA),
                                    scale_grid_density(base_ug, fB), t_a, t_b, devs)
                img = render_gpu_fused_slow(cfg, win; devices = devs)
                imgs[jph + 1] = img
                fluxes[jph + 1] = sum(img)
            end
            fstatic = sum(static)
            fmean = mean(fluxes)
            fvar = (maximum(fluxes) - minimum(fluxes)) / fmean          # peak-to-peak / mean
            meanimg = reduce(.+, imgs) ./ K
            cmean = corr(meanimg, static)
            @printf("  [d] flux peak-to-peak = %.1f%% of mean; ⟨flux⟩/static = %.4f; corr(⟨img⟩,static)=%.5f\n",
                    100 * fvar, fmean / fstatic, cmean)
            @test fvar > 0.02                                           # frames genuinely time-modulated
            @test isapprox(fmean, fstatic; rtol = 0.03)                 # period-average ≈ static
            @test cmean > 0.999                                         # and pixelwise
        end
    end
end

# ------------------------------------------------------------------------------------
# (f) PHYSICAL DUMP-TIME AXIS (regression). Real AthenaK dump series carry large
# ascending coordinate times (e.g. t ~ 1.2e4 GM/c³) while the backward-traced ray's
# X⁰ starts at 0 at the camera and DECREASES toward the source; the physical emission
# time is t_obs + X⁰ (DECREASING camera→source). Before the fix sample_primitives_t
# used X⁰ directly as the dump-time axis, so real dump times silently clamped every
# sample to the EARLIEST dump (fast light on frame 1) and any shifted series was
# applied time-reversed along the line of sight.
@testset "f: physical dump-time axis (realistic ascending dump times)" begin
    snap = tiny_snap()
    x = SVector(0.5, -0.3, 1.1)
    ts = [12000.0, 12500.0, 13000.0]                 # realistic AthenaK dump times
    fac(t) = 1.0 + (t - ts[1]) / 1000.0              # dump factors 1.0 / 1.5 / 2.0
    ser = synthetic_series(snap; times = ts, evolve = (s, t) -> scale_density(s, fac(t)))
    @test ser.t_obs == ts[end]                       # default: observe at the last dump
    rho(i) = sample_primitives(get_frame(ser, i), SVector(0.0, x...)).rho
    # camera end of the ray (X⁰ = 0) emits at t_obs = 13000 -> the LATEST dump
    @test sample_primitives_t(ser, SVector(0.0, x...)).rho == rho(3)
    # X⁰ = -500 into the march emits at 12500 -> exactly the middle dump
    @test sample_primitives_t(ser, SVector(-500.0, x...)).rho == rho(2)
    # X⁰ = -750 emits at 12250 -> the exact half blend of dumps 1 and 2
    @test sample_primitives_t(ser, SVector(-750.0, x...)).rho == 0.5 * rho(1) + 0.5 * rho(2)
    # deeper than the window (X⁰ < -1000) clamps to the EARLIEST dump
    @test sample_primitives_t(ser, SVector(-2145.0, x...)).rho == rho(1)
    # chronology along a march: sampled emission epoch never ADVANCES as X⁰ decreases
    rhos = [sample_primitives_t(ser, SVector(t, x...)).rho for t in 0.0:-100.0:-1200.0]
    @test issorted(rhos; rev = true)                 # density factor grows with dump time
    # an observation epoch outside the dump window is refused loudly instead of
    # silently collapsing to one window end
    @test_throws ErrorException synthetic_series(snap; times = ts, t_obs = 0.0)
    ser2 = SnapshotSeries(["a.athdf", "b.athdf"], ts[1:2])   # path-backed (never loaded)
    @test ser2.t_obs == ts[2]
    @test_throws ErrorException SnapshotSeries(["a.athdf", "b.athdf"], ts[1:2]; t_obs = 0.0)
end

# ------------------------------------------------------------------------------------
# (g) HOT-PATH PURITY (regression): sampling a fully-materialised series must not
# mutate shared series state (the per-sample LRU _touch! raced under Threads.@threads
# and allocated on every sample).
@testset "g: materialised-series sampling is read-only on the hot path" begin
    snap = tiny_snap()
    ser = synthetic_series(snap; times = [0.0, 10.0, 25.0],
                           evolve = (s, t) -> scale_density(s, 1.0 + 0.01 * t))
    x = SVector(0.5, -0.3, 1.1)
    sample_primitives_t(ser, SVector(20.0, x...))    # warm a bracketing pair
    ord0 = copy(ser.order)
    for t in (0.0, 4.2, 12.0, 20.0, 24.9)
        sample_primitives_t(ser, SVector(t, x...))
    end
    @test ser.order == ord0                          # no LRU mutation while sampling
    # threaded march reproduces the serial samples exactly (shared-state race regression)
    tsamp = collect(range(0.0, 25.0; length = 512))
    serial = [sample_primitives_t(ser, SVector(t, x...)).rho for t in tsamp]
    threaded = similar(serial)
    Threads.@threads for i in eachindex(tsamp)
        threaded[i] = sample_primitives_t(ser, SVector(tsamp[i], x...)).rho
    end
    @test threaded == serial
end

# ------------------------------------------------------------------------------------
# (h) GPU-PATH PHYSICAL DUMP-TIME AXIS (KA CPU backend regression — the device mirror
# of slab f). The slow-light kernels carried the old X⁰-as-dump-time convention
# w = clamp((X⁰ - t_a)·inv_dt, 0, 1), so realistic ascending dump times (t ~ 1.2e4)
# silently clamped every sample to frame A (fast light on frame 1) and any in-window
# evolution ran time-reversed. They now blend at the emission time t_emit = t_obs + X⁰
# via t0 = t_obs - t_a. Runs entirely on the KernelAbstractions CPU backend (the gpu_*
# `using CUDA` loads without a device; nothing CUDA is initialized).
# Base.include(@__MODULE__, …): runtests.jl loads this file into a dynamically
# created module (Base.include(Module(:TestSlowLight), …)), which lacks the implicit
# one-arg `include` a `module` block defines; the qualified form works there AND in
# the standalone (Main) invocation.
@isdefined(_gimg_sample_t) || Base.include(@__MODULE__, joinpath(REPO, "src", "gpu_image.jl"))
@isdefined(_gpol_kernel!)  || Base.include(@__MODULE__, joinpath(REPO, "src", "gpu_pol.jl"))
@isdefined(_gamr_sample_t) || Base.include(@__MODULE__, joinpath(REPO, "src", "gpu_amr.jl"))
using KernelAbstractions
using BlackLightPlus.Radiation: unpolarized_march
using BlackLightPlus: SimCoeffConfig

@testset "h: GPU slow-light kernels blend at the emission time t_obs + X⁰ (KA CPU)" begin
    # synthetic torus on a uniform grid; frame A (density x2) at t_a, frame B (base) at t_b
    ngr = 40; box_half = 30.0
    hostB = zeros(Float32, 8, ngr, ngr, ngr)
    dxg = 2box_half / (ngr - 1)
    for k in 1:ngr, j in 1:ngr, i in 1:ngr
        x = -box_half + (i-1)*dxg; y = -box_half + (j-1)*dxg; z = -box_half + (k-1)*dxg
        r = sqrt(x^2 + y^2 + z^2)
        if 2.5 < r < 26.0
            hostB[1,i,j,k] = exp(-((r-8)/6)^2)
            hostB[2,i,j,k] = -0.2*y/r; hostB[3,i,j,k] = 0.2*x/r; hostB[4,i,j,k] = 0.01
            hostB[5,i,j,k] = 0.08*hostB[1,i,j,k]
            hostB[6,i,j,k] = 0.02; hostB[7,i,j,k] = 0.03*x/r; hostB[8,i,j,k] = 0.05
        end
    end
    hostA = copy(hostB); hostA[1, :, :, :] .*= 2.0f0
    m_bh = 6.2e9 * 1.989e33
    L = BlackLightPlus.Constants.GNEWT * m_bh / BlackLightPlus.Constants.CL^2
    cfg = RunConfig(bh_a = 0.9375, m_bh_cgs = m_bh, mass_scale_cgs = 1.0e-21 * L^3,
                    frequency_hz = 2.3e11,
                    camera = CameraConfig(inclination_deg = 30.0, distance = 500.0,
                                          fov = 30.0, resolution = 12))
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam)
    a = cfg.bh_a; r_h = 1 + sqrt(1 - a*a); escape = 1.1 * cfg.camera.distance
    R_em = sqrt(3.0) * box_half; nmax = 3000; C = 80.0
    xo = -box_half; inv_dx = (ngr - 1) / (2box_half); nrays = length(rays)
    backend = CPU()
    gp = _gimg_params(cfg)
    runfast(h) = begin
        img = Vector{Float64}(undef, nrays)
        _fused_kernel!(backend)(img, rays, h, gp, a, r_h, C, escape, R_em, nmax,
                                ngr, xo, inv_dx, nrays; ndrange = nrays)
        KernelAbstractions.synchronize(backend)
        img
    end
    runslow(t0, inv_dt) = begin
        img = Vector{Float64}(undef, nrays)
        _fused_kernel_sl!(backend)(img, rays, hostA, hostB, t0, inv_dt, gp, a, r_h, C,
                                   escape, R_em, nmax, ngr, xo, inv_dx, nrays; ndrange = nrays)
        KernelAbstractions.synchronize(backend)
        img
    end
    # `_fused_kernel_sl!` and `_fused_kernel!` are distinct kernels that agree to
    # the last bit only when they vectorize the same way. Forcing bounds checks
    # on (--check-bounds=yes, which `Pkg.test` does by default) blocks that
    # vectorization and shifts the reduction order, so the short-circuit
    # guarantee below is asserted exactly on a normal build and to round-off on
    # a bounds-checked one. Differences seen there are ~1 ulp.
    bitsame(x, y) = Base.JLOptions().check_bounds == 1 ?
                    isapprox(x, y; rtol = 1e-12) : x == y
    # realistic ascending dump times (as slab f): t_a = 12000, t_b = 13000
    t_a, t_b = 12000.0, 13000.0
    inv_dt = 1.0 / (t_b - t_a)
    fastA = runfast(hostA); fastB = runfast(hostB)
    # doubling rho at fixed eint HALVES theta_e, so frame A is far dimmer — all the
    # test needs is two clearly distinguishable frames
    @test sum(fastA) > 0.0 && sum(fastB) > 0.0 && sum(fastA) != sum(fastB)
    # t_obs = t_a: every emission time t_a + X⁰ <= t_a  =>  w == 0  =>  frame A, BIT-identical
    @test bitsame(runslow(t_a - t_a, inv_dt), fastA)
    # t_obs far past t_b: w == 1 everywhere  =>  frame B, BIT-identical (w==1 short-circuit)
    @test bitsame(runslow(t_b + 2000.0 - t_a, inv_dt), fastB)
    # t_obs = t_b (the SnapshotSeries default): the emitting X⁰ range (~ -500) maps to
    # t_emit = 13000 + X⁰ strictly INSIDE the window -> a genuine blend. The X⁰-as-dump-
    # time convention returned frame A exactly here (|X⁰| ~ 500 << t_a  =>  w = 0).
    slow = runslow(t_b - t_a, inv_dt)
    @test slow != fastA                                    # FAILS pre-fix (silent fast light)
    # blend flux sits strictly between the two frames' fluxes
    @test min(sum(fastA), sum(fastB)) < sum(slow) < max(sum(fastA), sum(fastB))
    # CPU reference march with the sample_primitives_t emission-time weight on the
    # identical grids (geodesics agree to ~1 ulp, so compare at correlation/flux level)
    u = units(cfg.m_bh_cgs, cfg.mass_scale_cgs); p = cfg.plasma
    scc = SimCoeffConfig(bh_a = cfg.bh_a, d_unit = u.dens_unit,
                         e_unit = u.dens_unit * BlackLightPlus.Constants.CL^2, b_unit = u.B_unit,
                         plasma_mu = p.mu, plasma_ne_ni = p.ne_ni,
                         plasma_use_p = p.use_p ? 1 : 0, plasma_rat_low = p.rat_low,
                         plasma_rat_high = p.rat_high, plasma_gamma = p.gamma,
                         plasma_gamma_i = p.gamma_i, plasma_gamma_e = p.gamma_e,
                         plasma_thermal_frac = p.thermal_frac,
                         cut_sigma_max = p.sigma_cut, cut_theta_e_min = p.theta_e_min)
    γ = p.gamma
    sampler = x -> begin
        w = clamp(((t_b + x[1]) - t_a) * inv_dt, 0.0, 1.0)   # t_emit = t_obs + X⁰, t_obs = t_b
        prim = _gimg_sample_t(hostA, hostB, ngr, xo, inv_dx, w, x[2], x[3], x[4])
        (rho = prim[1], pgas = pgas_from_eint(prim[5], γ),
         uu = (prim[2], prim[3], prim[4]), bb = (prim[6], prim[7], prim[8]),
         inside = prim[1] > 0.0)
    end
    states, dls = trace_rays_sampled(st, rays; C = C, escape = escape, nmax = nmax)
    cpu = Vector{Float64}(undef, nrays)
    Threads.@threads for i in 1:nrays
        cpu[i] = unpolarized_march(st, scc, sampler, states[i], dls[i],
                                   cfg.frequency_hz, u.L_unit)
    end
    @test corr(slow, cpu) > 0.9999
    @test isapprox(sum(slow), sum(cpu); rtol = 1e-6)
    # polarized slow kernel: same axis; t_obs = t_a  =>  frame A, BIT-identical to fast pol
    gpp = _gpol_params(cfg)
    runpolfast() = begin
        o = Vector{SVector{4,Float64}}(undef, nrays)
        _gpol_kernel!(backend)(o, rays, hostA, gpp, a, r_h, C, escape, R_em, nmax,
                               ngr, xo, inv_dx, nrays; ndrange = nrays)
        KernelAbstractions.synchronize(backend)
        o
    end
    runpolslow(t0) = begin
        o = Vector{SVector{4,Float64}}(undef, nrays)
        _gpol_kernel_sl!(backend)(o, rays, hostA, hostB, t0, inv_dt, gpp, a, r_h, C,
                                  escape, R_em, nmax, ngr, xo, inv_dx, nrays; ndrange = nrays)
        KernelAbstractions.synchronize(backend)
        o
    end
    @test runpolslow(0.0) == runpolfast()
    @test runpolslow(t_b - t_a) != runpolfast()            # in-window blend engages
    # _win_t_obs: driver default observation epoch = the window's upper end
    ugA = UniformFluidGrid(hostA, ngr, box_half, xo, inv_dx)
    ugB = UniformFluidGrid(hostB, ngr, box_half, xo, inv_dx)
    win = upload_window(ugA, ugB, t_a, t_b, Int[])         # no device touched
    @test _win_t_obs(win) == t_b
    win∞ = upload_window(ugA, ugB, t_a, Inf, Int[])        # fast light: epoch immaterial
    @test win∞.inv_dt == 0.0 && _win_t_obs(win∞) == t_a
end

# ------------------------------------------------------------------------------------
# (i) time-blend samplers: clamped window ends short-circuit to a single frame read,
# bit-identical to the two-frame blend at w == 0 / w == 1 (and the interior blend is
# unchanged). Covers _gimg_sample_t (uniform) and _gamr_sample_t (native AMR).
@testset "i: _gimg_sample_t/_gamr_sample_t w==0/w==1 short-circuits are bit-exact" begin
    ngr = 8; box_half = 4.0
    hA = rand(Float32, 8, ngr, ngr, ngr) .+ 0.5f0
    hB = rand(Float32, 8, ngr, ngr, ngr) .+ 0.5f0
    xo = -box_half; inv_dx = (ngr - 1) / (2box_half)
    blend(w, x, y, z) = (1.0 - w) .* _gimg_sample(hA, ngr, xo, inv_dx, x, y, z) .+
                        w .* _gimg_sample(hB, ngr, xo, inv_dx, x, y, z)
    for (x, y, z) in ((0.5, -0.3, 1.1), (-3.9, 3.9, 0.0), (2.2, 2.3, -2.4))
        @test _gimg_sample_t(hA, hB, ngr, xo, inv_dx, 0.0, x, y, z) ==
              _gimg_sample(hA, ngr, xo, inv_dx, x, y, z)
        @test _gimg_sample_t(hA, hB, ngr, xo, inv_dx, 1.0, x, y, z) ==
              _gimg_sample(hB, ngr, xo, inv_dx, x, y, z)
        @test _gimg_sample_t(hA, hB, ngr, xo, inv_dx, 1.0, x, y, z) == blend(1.0, x, y, z)
        @test _gimg_sample_t(hA, hB, ngr, xo, inv_dx, 0.25, x, y, z) == blend(0.25, x, y, z)
    end
    # native-AMR twin on the tiny one-block snapshot
    snapA = tiny_snap(); snapB = scale_density(snapA, 1.7)
    blkA = permutedims(snapA.data, (5, 1, 2, 3, 4))
    blkB = permutedims(snapB.data, (5, 1, 2, 3, 4))
    xlft = reduce(hcat, snapA.x_left); dxb = reduce(hcat, snapA.dx)
    leaf, amrp = _build_leaf_table(snapA)
    for (x, y, z) in ((0.5, -0.3, 1.1), (-2.9, 2.9, 0.0))
        sA = _gamr_sample(blkA, xlft, dxb, leaf, amrp, x, y, z)
        sB = _gamr_sample(blkB, xlft, dxb, leaf, amrp, x, y, z)
        @test _gamr_sample_t(blkA, blkB, xlft, dxb, leaf, amrp, 0.0, x, y, z) == sA
        @test _gamr_sample_t(blkA, blkB, xlft, dxb, leaf, amrp, 1.0, x, y, z) == sB
        @test _gamr_sample_t(blkA, blkB, xlft, dxb, leaf, amrp, 1.0, x, y, z) ==
              (1.0 - 1.0) .* sA .+ 1.0 .* sB
        @test _gamr_sample_t(blkA, blkB, xlft, dxb, leaf, amrp, 0.5, x, y, z) ==
              0.5 .* sA .+ 0.5 .* sB
    end
end

# ------------------------------------------------------------------------------------
# (j) consolidated march: the fast/slow AMR POLARIZED kernel pair shares ONE
# _gpol_march body (sampler swap only), so the slow kernel at inv_dt = 0 on the same
# frame must reproduce the fast kernel to floating-point contraction accuracy.
# (Bit-for-bit equality between the two kernel instantiations is compiler-dependent
# — fmuladd fusion may differ per specialization — so a tight relative bound is
# pinned instead; observed deviation is ~2e-11 of the image scale.)
@testset "j: _gpol_amr_kernel_sl! inv_dt=0 matches _gpol_amr_kernel! (fast-light limit)" begin
    snap = tiny_snap()
    blk = permutedims(snap.data, (5, 1, 2, 3, 4))
    xlft = reduce(hcat, snap.x_left); dxb = reduce(hcat, snap.dx)
    leaf, amrp = _build_leaf_table(snap)
    m_bh = 6.2e9 * 1.989e33
    L = BlackLightPlus.Constants.GNEWT * m_bh / BlackLightPlus.Constants.CL^2
    cfg = RunConfig(bh_a = 0.9375, m_bh_cgs = m_bh, mass_scale_cgs = 1.0e-21 * L^3,
                    frequency_hz = 2.3e11,
                    camera = CameraConfig(inclination_deg = 30.0, distance = 500.0,
                                          fov = 12.0, resolution = 6))
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam)
    a = cfg.bh_a; r_h = 1 + sqrt(1 - a*a); escape = 1.1 * cfg.camera.distance
    R_em = _gamr_domain_radius(amrp); nmax = 3000; C = 80.0
    nrays = length(rays)
    backend = CPU()
    gpp = _gpol_params(cfg)
    fast = Vector{SVector{4,Float64}}(undef, nrays)
    _gpol_amr_kernel!(backend)(fast, rays, blk, xlft, dxb, leaf, amrp, gpp, a, r_h, C,
                               escape, R_em, nmax, nrays; ndrange = nrays)
    KernelAbstractions.synchronize(backend)
    slow = Vector{SVector{4,Float64}}(undef, nrays)
    _gpol_amr_kernel_sl!(backend)(slow, rays, blk, blk, xlft, dxb, leaf, amrp, 0.0, 0.0,
                                  gpp, a, r_h, C, escape, R_em, nmax, nrays; ndrange = nrays)
    KernelAbstractions.synchronize(backend)
    dev = maximum(maximum(abs, s - f) for (s, f) in zip(slow, fast))
    scale = maximum(maximum(abs, f) for f in fast)
    @test dev <= 1e-9 * scale                       # fast-light limit, fp-contraction bound
    @test any(o -> any(!=(0.0), o), fast)           # the tiny block actually emits
end

# ------------------------------------------------------------------------------------
# (k) DRIVER-THROUGHPUT ENABLERS (KA CPU backend; checks the invariants the drivers
# rely on). Three host/driver knobs added to the render_gpu_* drivers:
#  * `order` — march rays in a caller-supplied permutation (e.g. `march_sort_order` by
#    predicted trip count, so a warp's lanes march similar-length rays). Each ray's march is
#    independent + deterministic, so gather-in / unpermute-out is BIT-EXACT per ray.
#  * `wg`    — override the kernel workgroupsize. Block size does not enter the march math
#    (one independent ray per thread, no cross-lane reductions), so the image is bitwise-
#    invariant to it — a sanity check that the plumbing changed nothing.
#  * advance_series_window / advance_amr_window — slide a slow-light window forward by ONE
#    dump, REUSING the uploaded upper frame as the next lower frame (`.a === old .b`), so a
#    movie window resamples+uploads only the single new frame.
# All drivers hardcode CUDABackend(), so we exercise the SAME kernels + host helpers on the
# KA CPU backend directly (the pattern of slabs h/j).
@testset "k: order / wg / rolling-window enablers are bit-exact (KA CPU)" begin
    # permutation algebra (the exact gather/scatter the drivers apply)
    v = collect(1:12) .* 1.0; ord = [3, 1, 2, 5, 4, 6, 10, 9, 8, 7, 12, 11]
    @test _undo_order(_apply_order(v, ord), ord) == v                 # round-trip identity
    @test _apply_order(v, nothing) === v && _undo_order(v, nothing) === v
    @test march_sort_order([5, 2, 8, 1]) == [4, 2, 1, 3]              # ascending trip-count sort
    @test_throws ErrorException _apply_order(v, vcat([1, 1], collect(3:12)))  # not a permutation
    @test_throws ErrorException _apply_order(v, collect(1:11))                # wrong length

    m_bh = 6.2e9 * 1.989e33; L = BlackLightPlus.Constants.GNEWT * m_bh / BlackLightPlus.Constants.CL^2
    cfgk = RunConfig(bh_a = 0.9375, m_bh_cgs = m_bh, mass_scale_cgs = 1.0e-21 * L^3,
                     frequency_hz = 2.3e11,
                     camera = CameraConfig(inclination_deg = 50.0, distance = 500.0,
                                           fov = 26.0, resolution = 10))
    ak = cfgk.bh_a; r_hk = 1 + sqrt(1 - ak * ak); escapek = 1.1 * cfgk.camera.distance
    nmaxk = 3000; Ck = 80.0; bk = CPU()
    # two arbitrary permutations, both fully deterministic (no RNG import)
    twoperms(m) = (reverse(1:m), sortperm([sin(3.0 * i) for i in 1:m]),
                   march_sort_order([mod(7i, m) + 1 for i in 1:m]))

    # ---- uniform-grid deck (slab-h torus): unpolarized + polarized kernels
    ngr = 32; box_half = 26.0
    hostk = zeros(Float32, 8, ngr, ngr, ngr); dxg = 2box_half / (ngr - 1)
    for k in 1:ngr, j in 1:ngr, i in 1:ngr
        x = -box_half + (i-1)*dxg; y = -box_half + (j-1)*dxg; z = -box_half + (k-1)*dxg
        r = sqrt(x^2 + y^2 + z^2)
        if 2.5 < r < 22.0
            hostk[1,i,j,k] = exp(-((r-8)/6)^2); hostk[2,i,j,k] = -0.2*y/r; hostk[3,i,j,k] = 0.2*x/r
            hostk[4,i,j,k] = 0.01; hostk[5,i,j,k] = 0.08*hostk[1,i,j,k]
            hostk[6,i,j,k] = 0.02; hostk[7,i,j,k] = 0.03*x/r; hostk[8,i,j,k] = 0.05
        end
    end
    stk = KerrSchildCartesian(cfgk.bh_a)
    camk = PlaneParallelCamera(cfgk.camera.inclination_deg, cfgk.camera.distance,
                               cfgk.camera.fov, cfgk.camera.resolution)
    raysk = camera_rays(stk, camk)
    R_emk = sqrt(3.0) * box_half; xok = -box_half; inv_dxk = (ngr - 1) / (2box_half); nrk = length(raysk)
    ufk = _fused_kernel!(bk); pfk = _gpol_kernel!(bk)
    ruf(wg, rr) = (img = Vector{Float64}(undef, length(rr));
        _kern_launch!(ufk, wg, length(rr), img, rr, hostk, _gimg_params(cfgk), ak, r_hk, Ck,
                      escapek, R_emk, nmaxk, ngr, xok, inv_dxk, length(rr));
        KernelAbstractions.synchronize(bk); img)
    rpf(wg, rr) = (o = Vector{SVector{4,Float64}}(undef, length(rr));
        _kern_launch!(pfk, wg, length(rr), o, rr, hostk, _gpol_params(cfgk), ak, r_hk, Ck,
                      escapek, R_emk, nmaxk, ngr, xok, inv_dxk, length(rr));
        KernelAbstractions.synchronize(bk); o)
    ubase = ruf(nothing, raysk); pbase = rpf(96, raysk)
    @test sum(ubase) > 0.0 && any(o -> any(!=(0.0), o), pbase)   # deck emits
    @test ruf(64, raysk) == ubase && ruf(128, raysk) == ubase    # wg bitwise-invariant (unpol)
    @test rpf(64, raysk) == pbase                                # wg bitwise-invariant (pol, vs pin 96)
    for order in twoperms(nrk)                                   # arbitrary-permutation reassembly
        @test _undo_order(ruf(nothing, _apply_order(raysk, order)), order) == ubase
        @test _undo_order(rpf(96,      _apply_order(raysk, order)), order) == pbase
    end
    @test _undo_order(ruf(64, _apply_order(raysk, reverse(1:nrk))), reverse(1:nrk)) == ubase  # order+wg combined

    # ---- native-AMR deck (tiny one-block snapshot): _famr_kernel! + _gpol_amr_kernel!
    snapk = tiny_snap()
    blkk = permutedims(snapk.data, (5, 1, 2, 3, 4)); xlftk = reduce(hcat, snapk.x_left); dxbk = reduce(hcat, snapk.dx)
    leafk, amrpk = _build_leaf_table(snapk)
    camA = PlaneParallelCamera(30.0, 500.0, 12.0, 8); raysA = camera_rays(KerrSchildCartesian(ak), camA)
    R_emA = _gamr_domain_radius(amrpk); nrA = length(raysA)
    fak = _famr_kernel!(bk); pak = _gpol_amr_kernel!(bk)
    rfa(wg, rr) = (img = Vector{Float64}(undef, length(rr));
        _kern_launch!(fak, wg, length(rr), img, rr, blkk, xlftk, dxbk, leafk, amrpk,
                      _gimg_params(cfgk), ak, r_hk, Ck, escapek, R_emA, nmaxk, length(rr));
        KernelAbstractions.synchronize(bk); img)
    rpa(wg, rr) = (o = Vector{SVector{4,Float64}}(undef, length(rr));
        _kern_launch!(pak, wg, length(rr), o, rr, blkk, xlftk, dxbk, leafk, amrpk,
                      _gpol_params(cfgk), ak, r_hk, Ck, escapek, R_emA, nmaxk, length(rr));
        KernelAbstractions.synchronize(bk); o)
    abase = rfa(nothing, raysA); apbase = rpa(96, raysA)
    @test any(!=(0.0), abase)                                    # AMR deck emits
    @test rfa(64, raysA) == abase && rpa(64, raysA) == apbase    # wg bitwise-invariant (AMR)
    for order in (reverse(1:nrA), sortperm([cos(2.0 * i) for i in 1:nrA]))
        @test _undo_order(rfa(nothing, _apply_order(raysA, order)), order) == abase
        @test _undo_order(rpa(96,      _apply_order(raysA, order)), order) == apbase
    end

    # ---- rolling window advance: promote the uploaded upper frame (device-buffer identity;
    # Int[] devices => host-only, no CUDA). The reused grid is the SAME buffer a fresh upload
    # would resample byte-for-byte, so the advanced frame renders bit-identically.
    ser = synthetic_series(tiny_snap(); times = [0.0, 10.0, 20.0, 30.0])
    winF = upload_series_window(ser, 1, Int[]; box_half = 4.0, n = 8)
    winF2 = advance_series_window(winF, ser, 2, Int[]; box_half = 4.0, n = 8)
    @test winF2.a === winF.b                                     # promoted, NOT re-resampled/uploaded
    @test winF2.t_a == ser.times[2] && winF2.inv_dt == 1.0 / (ser.times[3] - ser.times[2])
    @test (winF2.b.n, winF2.b.xo, winF2.b.inv_dx) == (winF.b.n, winF.b.xo, winF.b.inv_dx)
    @test_throws ErrorException advance_series_window(winF, ser, 2, Int[]; box_half = 4.0, n = 16)  # geom guard
    @test advance_series_window(winF, ser, 4, Int[]; box_half = 4.0, n = 8).inv_dt == 0.0           # series end
    winM = upload_amr_window(ser, 1, Int[])
    winM2 = advance_amr_window(winM, ser, 2, Int[])
    @test winM2.a === winM.b
    @test winM2.t_a == ser.times[2] && winM2.b.amrp == winM.b.amrp
    @test advance_amr_window(winM, ser, 4, Int[]).inv_dt == 0.0
    @printf("  [k] order round-trip + wg-invariance bitwise (unpol/pol/AMR); window advance reuses .b ✓\n")
end
