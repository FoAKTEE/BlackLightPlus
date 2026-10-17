# Oracle: the BLP products against the BlackLightPlus drivers on the torus decks (a = 0.9,
# i = 60°, fov 24, d = 1000, rat_high = 10, sigma_cut = 1, C = 80, 2500 steps): every
# full-frame driver on host devices (the old device files are evaluated into a scratch
# module, with CUDA bound by identity), the two-phase CPU render, the coordinate march,
# the binary, slow light, mixed lanes, trajectories, the ADM shadow, derivatives and the
# mass-scale ladder, the exact-Kerr disk, adaptive refinement, and the backend switch.
#
# Stokes I is compared bitwise. A comparison marked `contracted` (the polarized images,
# trajectories, shadow states, the `rat_high` derivative lane) is compared natively to a
# fraction of the image maximum, because LLVM's fusion of `muladd` pairs into FMA
# instructions depends on the code around the inlined march; the script then reruns
# itself on `-C generic`, a CPU target without FMA, where every such comparison must be
# bitwise. A `report`ed comparison (the two-phase CPU path, the derivatives) must agree to
# the same fraction in both processes and records whether it is bitwise: the stored-path
# polarized march transports the frame along tracer states, the fused one with the
# geodesic, which differ by round-off.
Base.include(@__MODULE__, joinpath(@__DIR__, "helpers.jl"))
Base.include(@__MODULE__, joinpath(@__DIR__, "..", "decks.jl"))

using BLP.Plans: RunConfig, CameraConfig, PlasmaConfig, Lanes, MixedLanes, prepare
using BLP.Products: render, render_amr, render_analytic, photon_ring_layers, trace, shadow,
                    sensitivity, variants
using BLP.Transfer: StokesI, IQUV
using BLP.Data: FrameWindow, scale_density, synthetic_series, window, with_locate, BlockScan
using BLP.Execution: PoolExecutor, HostDevice, BackendUnavailable, backend!, reset_backend!
using BLP.Matter: EquatorialDisk

const OI = BlackLightPlus.Inputs
const OS = BlackLightPlus.Simulation
const CC = BlackLightPlus.Constants
const bits = Decks.bits

"True in the child process on a CPU target without FMA."
const NOFMA = get(ENV, "BLP_ORACLE_NOFMA", "") == "1"

# ---------------------------------------------------------------- old device drivers
const OLD = old_device_module(:OldDevice, ("gpu_image.jl", "gpu_pol.jl", "gpu_amr.jl",
                                           "gpu_spacetime.jl", "gpu_image_ad.jl", "gpu_analytic.jl",
                                           joinpath("geodesics", "gpu_tracer.jl")))

# The reference pools take CUDA ordinals or its own `HostDevice`s.
old_hosts(nd) = Decks.DEVICE_KIND == "cuda" ? [d.ordinal for d in Decks.devices(nd)] :
                [OLD.HostDevice(k) for k in 1:nd]

# ---------------------------------------------------------------- decks
const M_BH = 6.2e9 * 1.989e33
const L_U = CC.GNEWT * M_BH / CC.CL^2
const MS = 1.0e-21 * L_U^3
const NU = 2.3e11
const RES = 12
"Observation epoch of the slow-light decks: the emission times t_obs + X⁰ (X⁰ ≈ -1018 at r ≈ 5) fall in [0, 30]."
const T_OBS = 1030.0

old_cfg(; res = RES, a = 0.9, d = 1000.0, fov = 24.0) =
    OI.RunConfig(bh_a = a, m_bh_cgs = M_BH, mass_scale_cgs = MS, frequency_hz = NU,
                 camera = OI.CameraConfig(inclination_deg = 60.0, distance = d, fov = fov,
                                          resolution = res),
                 plasma = OI.PlasmaConfig(rat_high = 10.0, rat_low = 1.0, gamma = 13.0 / 9.0,
                                          use_p = true, sigma_cut = 1.0, theta_e_min = -1.0))
new_cfg(; res = RES, d = 1000.0, fov = 24.0) =
    RunConfig(m_bh_cgs = M_BH, mass_scale_cgs = MS, frequency_hz = NU,
              camera = CameraConfig(inclination_deg = 60.0, distance = d, fov = fov, resolution = res),
              plasma = PlasmaConfig(rat_high = 10.0, rat_low = 1.0, gamma = 13.0 / 9.0, use_p = true,
                                    sigma_cut = 1.0, theta_e_min = -1.0),
              max_steps = 2500, C = 80.0)

const GRID = Decks.uniform_deck()
const SNAP = Decks.amr_deck()
const OGRID = OLD.UniformFluidGrid(GRID.data, GRID.n, GRID.box_half, GRID.xo, GRID.inv_dx)

"The old snapshot of the new one: the same blocks in the reader layout."
function old_snapshot(s)
    b = s.blocks
    cols(a) = [SVector{3,Float64}(a[:, m]) for m in 1:b.nmb]
    return OS.AthenaKSnapshot(permutedims(b.data, (2, 3, 4, 5, 1)), cols(b.lo), cols(b.hi),
                              cols(b.x_left), cols(b.dx), copy(s.levels), copy(s.logical_locations),
                              copy(s.names), b.ni, b.nj, b.nk, b.nmb)
end
const OSNAP = old_snapshot(SNAP)

kerr() = NS.KerrSchildCartesian(0.9)
plan(matter; mode = IQUV(), cfg = new_cfg(), kw...) = prepare(cfg; spacetime = kerr(), matter, mode, kw...)
planes(img) = (img.I, img.Q, img.U, img.V)
oplanes(nt) = (nt.I, nt.Q, nt.U, nt.V)

# ---------------------------------------------------------------- comparisons
const MARCH_RTOL = 1e-7
const DEVIATIONS = Dict{String,Float64}()

# Every number of a result, in order, as Float64.
floats(x::Real) = [Float64(x)]
floats(x::AbstractArray{<:Real}) = Float64.(vec(x))
floats(x::AbstractArray{<:SVector}) = reduce(vcat, (Float64.(collect(v)) for v in x))
floats(x::AbstractVector{<:AbstractArray}) = reduce(vcat, (floats(v) for v in x))
floats(x::Tuple) = reduce(vcat, (floats(v) for v in x))

"""
Largest absolute difference over every component, over the largest finite old magnitude;
a NaN on both sides agrees, a NaN on one side is an infinite difference.
"""
function deviation(a, b)
    fa = floats(a); fb = floats(b)
    length(fa) == length(fb) || return Inf
    d = maximum((isnan(x) && isnan(y)) ? 0.0 : (isnan(x) || isnan(y)) ? Inf : abs(x - y)
                for (x, y) in zip(fa, fb))
    return d / maximum(abs, filter(isfinite, fb); init = floatmin())
end

"""
Bitwise equality of `new` and `old`; for a `contracted` comparison in the native process,
a deviation within `MARCH_RTOL` (recorded under `label`).
"""
function agree(label, new, old; contracted = false)
    if NOFMA || !contracted
        ok = bits(new) == bits(old)
        ok || (DEVIATIONS[label] = deviation(new, old))
        return ok
    end
    d = deviation(new, old)
    DEVIATIONS[label] = d
    return d <= MARCH_RTOL
end

# Trajectories: equal lengths and states within MARCH_RTOL of each state's scale.
function agree_paths(label, (s1, d1), (s2, d2))
    NOFMA && return bits(s1) == bits(s2) && bits(d1) == bits(d2)
    length.(s1) == length.(s2) || return false
    dev = maximum(maximum(maximum(abs.(a .- b) ./ (1 .+ abs.(b))) for (a, b) in zip(x, y))
                  for (x, y) in zip(s1, s2))
    DEVIATIONS[label] = dev
    return dev <= MARCH_RTOL
end

const POOLS = Dict(nd => PoolExecutor(Decks.devices(nd); lanes = 2, chunk = 7) for nd in Decks.NDEV)
const OPOOLS = Dict(nd => OLD.DevicePool(old_hosts(nd); lanes = 2, chunk = 7) for nd in Decks.NDEV)

"""
The old default `stokes` closure of `render_amr` on an in-memory snapshot: camera rays,
fixed-control trajectories, and the stored-path marches with the sampled gas pressure.
"""
function two_phase_stokes(cfg, snap, polarized)
    st = OG.KerrSchildCartesian(cfg.bh_a)
    cam = BlackLightPlus.PlaneParallelCamera(60.0, cfg.camera.distance, cfg.camera.fov,
                                             cfg.camera.resolution)
    u = OS.units(cfg.m_bh_cgs, cfg.mass_scale_cgs)
    scc = BlackLightPlus._prepared_scc(cfg, u)
    γ = cfg.plasma.gamma
    sampler = x -> (s = OS.sample_primitives(snap, x);
                    (rho = s.rho, pgas = OS.pgas_from_eint(s.eint, γ), uu = s.uu, bb = s.bb,
                     inside = s.inside))
    up = BlackLightPlus._camera_up(cam)
    return coords -> begin
        rays = BlackLightPlus.camera_rays_at(st, cam, coords)
        states, dls = BlackLightPlus.trace_rays_sampled(st, rays; C = 80.0, nmax = cfg.max_steps,
                                                        escape = 1.1 * cfg.camera.distance)
        out = zeros(4, length(rays))
        for i in eachindex(rays)
            if polarized
                out[:, i] .= OR.polarized_march(st, scc, sampler, states[i], dls[i],
                                                cfg.frequency_hz, u.L_unit; cam_up = up)
            else
                out[1, i] = OR.unpolarized_march(st, scc, sampler, states[i], dls[i],
                                                 cfg.frequency_hz, u.L_unit)
            end
        end
        out
    end
end

"A comparison that is reported, not required bitwise: within MARCH_RTOL, bits recorded."
function report(label, new, old)
    d = deviation(new, old)
    DEVIATIONS[label] = bits(new) == bits(old) ? 0.0 : d
    return d <= MARCH_RTOL
end

@testset "oracle: Products$(NOFMA ? " (no FMA)" : "")$(Decks.DEVICE_KIND == "cuda" ? " (CUDA)" : "")" begin
    oc = old_cfg()
    dev1 = old_hosts(1)

    @testset "one program on every executor: $name, $(nameof(typeof(mode)))" for
            (name, matter) in (("uniform grid", GRID), ("blocks", SNAP)), mode in (StokesI(), IQUV())
        p = plan(matter; mode)
        ref = bits(planes(render(p; backend = :cpu)))
        @test bits(planes(render(p; backend = Decks.POOL_BACKEND))) == ref
        @test all(bits(planes(render(p; backend = POOLS[nd]))) == ref for nd in Decks.NDEV)
        @test count(!iszero, render(p; backend = :cpu).I) > RES^2 ÷ 4
    end

    @testset "Stokes I and IQUV against the fused drivers" begin
        for (matter, ofused, opol) in ((GRID, () -> OLD.render_gpu_fused(oc, OGRID; devices = dev1),
                                        () -> OLD.render_gpu_polarized(oc, OGRID; devices = dev1)),
                                       (SNAP, () -> OLD.render_gpu_amr(oc, OSNAP; devices = dev1),
                                        () -> OLD.render_gpu_amr_polarized(oc, OSNAP; devices = dev1)))
            lab = matter === GRID ? "uniform" : "blocks"
            @test agree("$lab I", render(plan(matter; mode = StokesI()); backend = Decks.POOL_BACKEND).I, ofused())
            @test agree("$lab IQUV", planes(render(plan(matter); backend = Decks.POOL_BACKEND)), oplanes(opol());
                        contracted = true)
        end
        # the old pools over 1, 2, 3, 5 host devices render the same image
        ref = render(plan(SNAP; mode = StokesI()); backend = POOLS[5]).I
        @test all(agree("blocks I, $nd devices", ref, OLD.render_gpu_amr(oc, OSNAP; devices = OPOOLS[nd]))
                  for nd in Decks.NDEV[2:end])
    end

    @testset "the two-phase CPU render" begin
        pI = plan(SNAP; mode = StokesI()); pQ = plan(SNAP)
        oI = BlackLightPlus.render(oc, OSNAP; polarized = false)
        oQ = BlackLightPlus.render(oc, OSNAP; polarized = true)
        @test report("two-phase CPU I vs fused", render(pI; backend = :cpu).I, oI.I)
        @test report("two-phase CPU IQUV vs fused", planes(render(pQ; backend = :cpu)), oplanes(oQ))
        t = trace(pI; backend = :cpu)
        @test report("two-phase CPU I vs stored path", render(pI; path = t, backend = :cpu).I, oI.I)
        @test report("two-phase CPU IQUV vs stored path", planes(render(pQ; path = t, backend = :cpu)),
                     oplanes(oQ))
        @test agree("stored path vs fused I", render(pI; path = t, backend = :cpu).I,
                    render(pI; backend = :cpu).I; contracted = true)
    end

    @testset "coordinate lists" begin
        coords = [(-7.3, 2.1), (0.4, -5.5), (3.3, 3.3), (10.0, -1.0), (-2.2, -9.9), (0.05, 6.0)]
        ocam = BlackLightPlus.PlaneParallelCamera(60.0, 1000.0, 24.0, RES)
        rays = BlackLightPlus.camera_rays_at(OG.KerrSchildCartesian(0.9), ocam, coords)
        @test agree("coords I", render(plan(SNAP; mode = StokesI()), coords; backend = Decks.POOL_BACKEND),
                    OLD.march_rays_gpu_amr(oc, OSNAP, rays; devices = dev1))
        @test agree("coords IQUV", render(plan(SNAP), coords; backend = POOLS[2]),
                    OLD.march_rays_gpu_amr(oc, OSNAP, rays; polarized = true, devices = dev1);
                    contracted = true)
    end

    @testset "superposed binary" begin
        m, s, c = [0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)]
        stn = NS.SuperposedKerrSchild(m, s, c); sto = OG.SuperposedKerrSchild(m, s, c)
        scan = with_locate(SNAP.blocks, BlockScan())
        pb(mode) = prepare(new_cfg(); spacetime = stn, matter = scan, mode)
        @test agree("binary I", render(pb(StokesI()); backend = Decks.POOL_BACKEND).I,
                    OLD.render_gpu_spacetime(oc, sto, OSNAP; devices = dev1))
        @test agree("binary IQUV", planes(render(pb(IQUV()); backend = Decks.POOL_BACKEND)),
                    oplanes(OLD.render_gpu_spacetime(oc, sto, OSNAP; polarized = true, devices = dev1));
                    contracted = true)
    end

    @testset "slow light" begin
        win = FrameWindow(GRID, scale_density(GRID, 1.3), 0.0, 30.0; t_obs = T_OBS)
        owin = OLD.upload_window(OGRID, OLD.scale_grid_density(OGRID, 1.3), 0.0, 30.0, dev1)
        slowI = render(plan(win; mode = StokesI()); backend = Decks.POOL_BACKEND).I
        @test agree("slow uniform I", slowI, OLD.render_gpu_fused_slow(oc, owin; devices = dev1, t_obs = T_OBS))
        fastI = render(plan(GRID; mode = StokesI()); backend = :cpu).I
        @test maximum(abs.(slowI .- fastI)) > 1e-3 * maximum(fastI)
        @test agree("slow uniform IQUV", planes(render(plan(win); backend = Decks.POOL_BACKEND)),
                    oplanes(OLD.render_gpu_polarized_slow(oc, owin; devices = dev1, t_obs = T_OBS));
                    contracted = true)
        series = synthetic_series(SNAP; times = [0.0, 30.0], t_obs = 15.0,
                                  evolve = (s, t) -> t == 0.0 ? s : scale_density(s, 1.3))
        awin = window(series, 1; t_obs = T_OBS)
        oawin = OLD.DeviceAMRWindow(OLD.upload_amr(OSNAP, dev1; upload_bounds = false),
                                    OLD.upload_amr(old_snapshot(scale_density(SNAP, 1.3)), dev1;
                                                   upload_bounds = false), 0.0, 1 / 30)
        @test agree("slow blocks I", render(plan(awin; mode = StokesI()); backend = Decks.POOL_BACKEND).I,
                    OLD.render_gpu_amr_slow(oc, oawin; devices = dev1, t_obs = T_OBS))
        @test agree("slow blocks IQUV", planes(render(plan(awin); backend = Decks.POOL_BACKEND)),
                    oplanes(OLD.render_gpu_amr_polarized_slow(oc, oawin; devices = dev1, t_obs = T_OBS));
                    contracted = true)
    end

    @testset "mixed lanes" begin
        for (matter, old) in ((GRID, () -> OLD.render_gpu_fused(oc, OGRID; devices = dev1, precision = :mixed)),
                              (SNAP, () -> OLD.render_gpu_amr(oc, OSNAP; devices = dev1, precision = :mixed)))
            p = plan(matter; mode = StokesI(), lanes = MixedLanes())
            @test !isempty(p.rescue)
            img = render(p; backend = Decks.POOL_BACKEND).I
            @test agree("mixed I", img, old())
            @test bits(render(p; backend = :cpu).I) == bits(img)
        end
        p = plan(SNAP; lanes = MixedLanes())
        @test agree("mixed IQUV", planes(render(p; backend = Decks.POOL_BACKEND)),
                    oplanes(OLD.render_gpu_amr_polarized(oc, OSNAP; devices = dev1, precision = :mixed));
                    contracted = true)
    end

    @testset "trajectories" begin
        p = plan(GRID; mode = StokesI())
        sto = OG.KerrSchildCartesian(0.9)
        rays = BlackLightPlus.camera_rays(sto, BlackLightPlus.PlaneParallelCamera(60.0, 1000.0, 24.0, RES))
        t = trace(p; backend = :cpu)
        tp = trace(p; backend = POOLS[3])
        @test bits(t.states) == bits(tp.states) && bits(t.dls) == bits(tp.dls)
        @test agree_paths("trace vs sampled tracer", (t.states, t.dls),
                          BlackLightPlus.trace_rays_sampled(sto, rays; C = 80.0, escape = 1.1 * 1000.0,
                                                            nmax = 2500, R_em = p.R_em))
        @test agree_paths("trace vs device tracer", (t.states, t.dls),
                          OLD.trace_rays_gpu(sto, rays; C = 80.0, escape = 1.1 * 1000.0, R_em = p.R_em,
                                             nmax = 2500, devices = old_hosts(2), chunk = 9))
        @test maximum(length, t.states) > 500
    end

    @testset "shadows: Kerr and the ADM grid" begin
        xs = -20.0:0.5:20.0
        kn = NS.KerrSchildCartesian(0.9); ko = OG.KerrSchildCartesian(0.9)
        gn = NS.adm_grid_from_metric(x -> NS.metric(kn, x), xs, xs, xs; alpha_min = 0.1,
                                     excise = x -> NS.ks_radius(kn, x) < NS.horizon_radius(kn))
        go = OG.adm_grid_from_metric(x -> OG.metric(ko, x), xs, xs, xs; alpha_min = 0.1,
                                     excise = x -> OG.ks_radius(ko, x) < OG.horizon_radius(ko))
        for (stn, sto, d, fov) in ((kn, ko, 1000.0, 24.0), (gn, go, 15.0, 16.0))
            p = prepare(new_cfg(res = 10, d = d, fov = fov); spacetime = stn, matter = nothing)
            sh = shadow(p; backend = Decks.POOL_BACKEND)
            @test bits(sh.null_error) == bits(shadow(p; backend = :cpu).null_error)
            rays = BlackLightPlus.camera_rays(sto, BlackLightPlus.PlaneParallelCamera(60.0, d, fov, 10))
            st, ns, mn, rf, yf = OLD.trace_shadow_gpu(sto, rays; C = 80.0, escape = 1.1 * d,
                                                      nmax = 2500, devices = old_hosts(2))
            @test vec(sh.fate) == Int.(st) && vec(sh.steps) == Int.(ns)
            @test agree("shadow $(nameof(typeof(stn)))", (vec(sh.null_error), vec(sh.radius), vec(sh.final)),
                        (mn, rf, yf); contracted = true)
            @test 0 < count(==(1), sh.fate) < 100 && count(==(2), sh.fate) > 0
        end
    end

    @testset "sensitivity and variants" begin
        p = plan(GRID; mode = StokesI())
        for param in (:mass_scale, :rat_high)
            s = sensitivity(p, param; backend = Decks.POOL_BACKEND)
            img, dimg, flux, dflux = OLD.render_gpu_fused_grad(oc, OGRID, param; devices = dev1)
            # the seeded electron model is Dual in every field (only `rat_high` carries a
            # partial), so its value lane runs other code than the plain march: contracted
            @test agree("sensitivity $param: image", s.image.I, img; contracted = param === :rat_high)
            @test agree("sensitivity $param: flux", s.flux, flux; contracted = param === :rat_high)
            @test report("sensitivity $param: derivative", s.derivative, dimg)
            @test report("sensitivity $param: flux derivative", s.dflux, dflux)
            @test count(!iszero, s.derivative) > RES^2 ÷ 4
        end
        ms = [MS, 2.5 * MS]
        v = variants(p, ms; backend = Decks.POOL_BACKEND)
        @test agree("variants", v.images, OLD.render_gpu_fused_kvariant(oc, OGRID, ms; devices = dev1))
        @test bits(v.images) == bits(variants(p, ms; backend = POOLS[5]).images)
    end

    @testset "exact-Kerr equatorial disk" begin
        p = plan(nothing; mode = StokesI())
        disk = EquatorialDisk(0.9); odisk = OR.EquatorialDisk(0.9)
        fields = (:I, :Ib, :rs, :ts, :phis, :redshift_sign)
        ra = render_analytic(p, disk; backend = :cpu)
        oa = BlackLightPlus.render_analytic(oc; source = odisk)
        og = OLD.render_analytic_gpu(oc; devices = old_hosts(2), source = odisk)
        @test all(agree("analytic $f vs CPU", getfield(ra, f), getfield(oa, f)) for f in fields)
        @test all(agree("analytic $f vs device", getfield(ra, f), getfield(og, f)) for f in fields)
        @test ra.bands == oa.bands
        rp = render_analytic(p, disk; backend = POOLS[3])
        @test all(bits(getfield(rp, f)) == bits(getfield(ra, f)) for f in fields)
        @test count(>(0), ra.Ib[2]) > 0
        pl = photon_ring_layers(p, disk; dx = 0.2, backend = :cpu)
        opl = BlackLightPlus.photon_ring_layers(oc; disk = odisk, dx = 0.2)
        @test pl.N == opl.N && pl.grid == opl.grid && pl.masks == opl.masks
        @test agree("photon ring layers", (pl.I, pl.Ib), (opl.I, opl.Ib))
        @test bits(photon_ring_layers(p, disk; dx = 0.2, backend = POOLS[2]).Ib) == bits(pl.Ib)
    end

    @testset "adaptive refinement: $(nameof(typeof(mode)))" for mode in (StokesI(), IQUV())
        cfg16 = old_cfg(res = 16)
        p = plan(SNAP; mode, cfg = new_cfg(res = 16))
        pol = mode isa IQUV
        new = render_amr(p; block = 4, max_level = 2, backend = Decks.POOL_BACKEND)
        stk = OLD.AMRStokes(cfg16, OSNAP; polarized = pol, devices = old_hosts(2))
        old = BlackLightPlus.render_amr(cfg16, ""; block = 4, max_level = 2, stokes = stk)
        @test new.levels == old.levels && new.nrays == old.nrays && maximum(new.levels) >= 1
        @test agree("render_amr $(nameof(typeof(mode))) vs streamed device marcher",
                    planes(new.image), oplanes(old); contracted = pol)
        cpu = BlackLightPlus.render_amr(cfg16, ""; block = 4, max_level = 2,
                                        stokes = two_phase_stokes(cfg16, OSNAP, pol))
        @test new.levels == cpu.levels && new.nrays == cpu.nrays
        @test report("render_amr $(nameof(typeof(mode))) vs two-phase CPU", planes(new.image), oplanes(cpu))
        @test bits(planes(render_amr(p; block = 4, max_level = 2, backend = :cpu).image)) ==
              bits(planes(new.image))
    end

    @testset "backend switch through the products" begin
        p = plan(GRID; mode = StokesI(), cfg = new_cfg(res = 4))
        @test_throws BackendUnavailable render(p; backend = :gpu)
        reset_backend!()
        withenv("BLP_BACKEND" => "host_pool:2") do
            r = render(p)
            @test r.run.executor.kind === :pool && length(r.run.executor.devices) == 2
            backend!(:cpu)
            @test render(p).run.executor.kind === :threads
            @test render(p; backend = Decks.POOL_BACKEND).run.executor.kind === :pool
            backend!(nothing)
            @test render(p).run.executor.kind === :pool
        end
        reset_backend!()
    end

    @info "Products oracle$(NOFMA ? " (no FMA)" : ""): deviations / image maximum (0 = bitwise)\n" *
          join(("$k: $v" for (k, v) in sort(collect(DEVIATIONS))), "\n")

    # The no-FMA rerun concerns the host target; on CUDA devices the script runs once.
    if !NOFMA && Decks.DEVICE_KIND != "cuda"
        @testset "every comparison bitwise without FMA" begin
            cmd = `$(Base.julia_cmd()) -C generic -t 8 --project=$(Base.active_project())
                   -e "using Test; include($(repr(@__FILE__)))"`
            @test success(pipeline(addenv(cmd, "BLP_ORACLE_NOFMA" => "1"); stdout, stderr))
        end
    end
end
