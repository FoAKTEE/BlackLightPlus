# Oracle: camera rays of BLP.Spacetime against the old host `camera_rays`,
# `camera_rays_at` and the old device ray kernel (run on the KernelAbstractions CPU
# backend), bit for bit, for several cameras and spacetimes, in fp64 and narrowed to fp32.
Base.include(@__MODULE__, joinpath(@__DIR__, "helpers.jl"))

const CAM = scratch_module(:OldCamera)
Core.eval(CAM, :(using BlackLightPlus.Geodesics: CameraFrame, camera_ray, pixel_centres))
load_old!(CAM, "gpu_camera.jl", [:PixelGrid, :image_coords, :BlockGrid, :_camray_kernel!])

# The new rays of every pixel, in camera order.
function new_rays(st, cam, ::Type{T} = Float64) where {T}
    fr = NS.CameraFrame(cam)
    centres = NS.pixel_centres(cam)
    out = Vector{SVector{8,T}}(undef, NS.npixels(cam))
    ok = true
    for k in eachindex(out)
        α, β = NS.pixel_coords(centres, cam.npix, k)
        y, okk = NS.camera_ray(st, fr, α, β)
        out[k] = SVector{8,T}(y)
        ok &= okk
    end
    return out, ok
end

function old_kernel_rays(st, cam, ::Type{T} = Float64) where {T}
    n = cam.npix^2
    y0s = Vector{SVector{8,T}}(undef, n); bad = zeros(Int32, 1)
    gen = CAM.PixelGrid(OG.pixel_centres(cam), cam.npix)
    CAM._camray_kernel!(KA.CPU())(y0s, bad, Int32.(1:n), gen, st, OG.CameraFrame(cam), n;
                                  ndrange = n)
    KA.synchronize(KA.CPU())
    return y0s, bad[1] == 0
end

@noinline callf(f::F, args...) where {F} = f(args...)
const nself = Ref(0)
const ncoord = Ref(0)

const PHI0 = -0.2233022
const C1 = 12.5 .* (cos(PHI0), sin(PHI0), 0.0)

@testset "oracle: camera rays" begin
    rng = Xoshiro(161016)
    kerr(a) = (NS.KerrSchildCartesian(a), OG.KerrSchildCartesian(a))
    pairs = Any[kerr(0.0), kerr(0.9), kerr(0.998),
                (NS.SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [C1, (-C1[1], -C1[2], 0.0)]),
                 OG.SuperposedKerrSchild([0.5, 0.5], [0.0, 0.0], [C1, (-C1[1], -C1[2], 0.0)]))]
    kn, ko = kerr(0.9)
    xs = -15.0:1.0:15.0
    push!(pairs, (NS.adm_grid_from_metric(x -> NS.metric(kn, x), xs, xs, xs; alpha_min = 0.1),
                  OG.adm_grid_from_metric(x -> OG.metric(ko, x), xs, xs, xs; alpha_min = 0.1)))
    push!(pairs, (NS.UserMetric(x -> NS.metric(kn, x)), OG.UserMetric(x -> OG.metric(ko, x))))
    cams = [NS.PlaneParallelCamera(60.0, 1000.0, 40.0, 32), NS.PlaneParallelCamera(17.0, 1.0e6, 40.0, 16),
            NS.PlaneParallelCamera(90.0, 300.0, 40.0, 12), NS.PlaneParallelCamera(120.0, 1000.0, 24.0, 16),
            NS.PlaneParallelCamera(163.0, 120.0, 12.0, 9), NS.PlaneParallelCamera(0.0, 500.0, 10.0, 7)]
    for (stn, sto) in pairs, cn in cams
        co = OG.PlaneParallelCamera(cn.inc_deg, cn.d, cn.fov, cn.npix)
        @test reinterpret(UInt64, NS.pixel_centres(cn)) == reinterpret(UInt64, OG.pixel_centres(co))
        fr = NS.CameraFrame(cn); fo = OG.CameraFrame(co)
        @test (fr.sin_i, fr.cos_i, fr.d) === (fo.sin_i, fo.cos_i, fo.d)
        @test fr.up === BlackLightPlus._camera_up(co)
        rn, okn = new_rays(stn, cn)
        ro = OG.camera_rays(sto, co)
        @test okn && all(same.(rn, ro))
        rk, okk = old_kernel_rays(sto, co)
        @test okk && all(same.(rn, rk))
        r32, _ = new_rays(stn, cn, Float32)
        k32, _ = old_kernel_rays(sto, co, Float32)
        @test all(same.(r32, k32))
        # arbitrary image coordinates: the new ray equals the reference `camera_ray`
        # compiled in the same context; the reference `camera_rays_at` loop itself
        # differs from its `camera_ray` in the last bit for a few coordinates
        coords = [(4.0 * randn(rng), 4.0 * randn(rng)) for _ in 1:500]
        @test all(same(callf(NS.camera_ray, stn, fr, c...), callf(OG.camera_ray, sto, fo, c...))
                  for c in coords)
        at = OG.camera_rays_at(sto, co, coords)
        nself[] += count(!same(OG.camera_ray(sto, fo, c...)[1], y) for (c, y) in zip(coords, at))
        ncoord[] += length(coords)
    end
    @info "reference camera_rays_at vs reference camera_ray: coordinates with different bits" nself[] ncoord[]
end
