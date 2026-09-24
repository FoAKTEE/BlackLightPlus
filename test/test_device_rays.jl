# Camera rays made on the device (src/gpu_camera.jl; results/blp/multigpu/dag.md, MG08).
# `_camray_kernel!` calls the host's `camera_ray` on each item's image-plane coordinates,
# so on the KA CPU backend its rays equal `camera_rays` / `camera_rays_at` bit for bit —
# for the camera pixel grid and AMR blocks, fp64 and the narrowed fp32 lane, Kerr, the
# superposed binary and an ADM grid spacetime — and every full-frame driver renders the
# same image as the host-ray path.
using BlackLightPlus: camera_rays_at, metric, adm_grid_from_metric
using BlackLightPlus.Geodesics: CameraFrame
@isdefined(DI_NDEV) || include(joinpath(@__DIR__, "device_decks.jl"))

# Rays of `src` for items 1:n, made on one HostDevice worker in chunks of 13.
function device_rays(src::CameraRays{T}, n) where {T}
    pool = DevicePool([HostDevice(1)]; lanes = 2, chunk = 13)
    out = Vector{SVector{8,T}}(undef, n)
    foreach_chunk(pool, collect(Iterators.partition(1:n, 13))) do lane, r
        y = rays_on_device!(lane, src, r)
        copyto!(out, first(r), y, 1, length(r))
    end
    close(pool)
    return out
end

@testset "device camera rays (KA CPU backend)" begin

@testset "pixel grid == camera_rays, bitwise" begin
    kerr = KerrSchildCartesian(0.9)
    bbh = SuperposedKerrSchild([0.5, 0.5], [0.3, -0.2], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
    for st in (kerr, bbh), inc in (0.0, 17.0, 60.0, 90.0, 163.0)
        cam = PlaneParallelCamera(inc, 1000.0, 24.0, 11)
        host = camera_rays(st, cam)
        @test isequal(device_rays(CameraRays(st, cam), 121), host)
        @test isequal(device_rays(CameraRays(st, cam, Float32), 121),
                      [SVector{8,Float32}(r) for r in host])
    end
    # ADM grid: the camera sits on the grid, so the rays use the interpolated metric; the
    # device copy (DevADMParams over the uploaded data) runs the same body
    xs = range(-8.0, 8.0; length = 17)
    adm = adm_grid_from_metric(x -> metric(kerr, x), xs, xs, xs; alpha_min = 0.1)
    cam = PlaneParallelCamera(60.0, 5.0, 4.0, 6)
    dev = Dict{Any,Any}(HostDevice(1) => DevADMParams(adm, copy(adm.data)))
    @test isequal(device_rays(CameraRays(dev, cam), 36), camera_rays(adm, cam))
end

@testset "AMR block grid == camera_rays_at, bitwise" begin
    st = KerrSchildCartesian(0.9)
    cam = PlaneParallelCamera(60.0, 1000.0, 24.0, 16)
    bs = 4
    blocks = [SVector(-12.0 + 1.5i, -12.0 + 3.0i, 3.0 / 2^(i % 3) / bs) for i in 0:6]
    coords = Tuple{Float64,Float64}[]
    for b in blocks, i in 1:bs, j in 1:bs          # the render_amr coordinate loop
        push!(coords, (b[1] + (i - 0.5) * b[3], b[2] + (j - 0.5) * b[3]))
    end
    src = CameraRays(st, CameraFrame(cam), BlockGrid(blocks, bs))
    @test isequal(device_rays(src, length(coords)), camera_rays_at(st, cam, coords))
end

@testset "no positive-energy root: an error, as on the host" begin
    st = KerrSchildCartesian(0.99)
    cam = PlaneParallelCamera(90.0, 1.2, 0.1, 2)     # the camera inside the ergosphere
    @test_throws ErrorException camera_rays(st, cam)
    err = try
        device_rays(CameraRays(st, cam), 4); nothing
    catch e
        e
    end
    @test err isa CapturedException && occursin("null root", sprint(showerror, err.ex))
end

@testset "full frames: device rays render the host-ray image" begin
    cfg = di_cfg()
    fg = di_uniform()
    snap = di_amr()
    st = KerrSchildCartesian(cfg.bh_a)
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st, cam)
    devs = di_hosts(2)
    pool = device_pool(devs)
    dg = upload_grid(fg, pool)
    a = cfg.bh_a; r_h = 1 + sqrt(1 - a * a); R_em = sqrt(3.0) * abs(dg.xo)
    host(launch, T) = (o = Vector{T}(undef, length(rays));
                       march!((o,), pool, [MarchPart(1:length(rays), HostRays(rays), launch)]); o)
    @test isequal(vec(render_gpu_fused(cfg, dg; devices = devs)),
                  host(_fused_launch(dg, _gimg_params(cfg), a, r_h, 80.0, 1100.0, R_em,
                                     cfg.max_steps, nothing), Float64))
    pol = render_gpu_polarized(cfg, dg; devices = devs)
    hp = host(_gpol_launch(dg, _gpol_params(cfg), a, r_h, 80.0, 1100.0, R_em, cfg.max_steps),
              SVector{4,Float64})
    @test all(isequal(vec(getfield(pol, k)), getindex.(hp, k)) for k in 1:4)
    @test isequal(vec(render_gpu_amr(cfg, snap; devices = devs)),
                  march_rays_gpu_amr(cfg, snap, rays; devices = devs)[1, :])
    ap = render_gpu_amr_polarized(cfg, snap; devices = devs)
    M = march_rays_gpu_amr(cfg, snap, rays; devices = devs, polarized = true)
    @test all(isequal(vec(getfield(ap, k)), M[k, :]) for k in 1:4)
    bbh = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
    @test isequal(vec(render_gpu_spacetime(cfg, bbh, snap; devices = devs)),
                  march_rays_gpu_spacetime(cfg, bbh, snap, camera_rays(bbh, cam); devices = devs))
end

end
