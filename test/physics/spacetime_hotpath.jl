# The hot-path contract: per-step functions allocate nothing in Float64 and Float32, the
# spacetime records are isbits, Dual states and spins flow through the integrator, the
# lanes narrow from the stored Float64 values, and the ADM grid adapts its data.
using Test, BlackLightPlus, StaticArrays, ForwardDiff, Adapt
using BlackLightPlus.Numerics: lerp8, xsum_add!, xsum_atomic_add!, xsum_zeros, mc_uniform, MCRng, primary_rng
using BlackLightPlus.Spacetime: KerrSchildCartesian, SuperposedKerrSchild, ADMGrid, Minkowski, UserMetric,
                     MarchBudget, MarchStep, CameraFrame, PlaneParallelCamera, metric_and_inverse,
                     geodesic_rhs, geodesic_transport_rhs, rk4_step, march_control, camera_ray,
                     lane, metric, inverse_metric, adm_grid_from_metric, ks_radius, horizon_radius,
                     raise_and_norm

# Bytes allocated by one call, after a warm-up call compiles it.
allocs(f::F, args::Vararg{Any,N}) where {F,N} = (f(args...); @allocated f(args...))
# Fixed-arity callers of the Vararg-indexed accumulators (the harness above would
# otherwise forward through an unspecialized Vararg call).
add_cell(acc, x) = xsum_add!(acc, x, 2, 3)
atomic_add_cell(acc, x) = xsum_atomic_add!(acc, x, 1, 1)

@testset "hot path" begin
    st = KerrSchildCartesian(0.9)
    st32 = lane(st, Float32)
    bin = SuperposedKerrSchild([0.5, 0.5], [0.0, 0.3], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)])
    xs = -8.0:1.0:8.0
    grid = adm_grid_from_metric(x -> metric(st, x), xs, xs, xs; alpha_min = 0.1)
    y = SVector(0.0, 7.0, 3.0, 2.0, 1.0, -0.9, 0.05, 0.02)
    y32 = SVector{8,Float32}(y)
    x = SVector(y[1], y[2], y[3], y[4]); x32 = SVector{4,Float32}(x)
    b = MarchBudget(80.0, 1100.0, Inf, 6000); b32 = lane(b, Float32)
    fr = CameraFrame(PlaneParallelCamera()); fr32 = CameraFrame(Float32(fr.sin_i), Float32(fr.cos_i),
                                                                Float32(fr.d), SVector{4,Float32}(fr.up))
    vs = (SVector(0.0, 1.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0, 0.0)); vs32 = map(SVector{4,Float32}, vs)

    @testset "zero allocations" begin
        for (s, yy, xx, bb, vv, frm) in ((st, y, x, b, vs, fr), (st32, y32, x32, b32, vs32, fr32),
                                         (bin, y, x, b, vs, fr), (grid, y, x, b, vs, fr))
            T = eltype(yy)
            @test allocs(metric_and_inverse, s, xx) == 0
            @test allocs(geodesic_rhs, s, yy) == 0
            @test allocs(geodesic_transport_rhs, s, yy, vv) == 0
            @test allocs(rk4_step, s, yy, T(0.1)) == 0
            @test allocs(march_control, s, yy, T(Inf), bb) == 0
            @test allocs(camera_ray, s, frm, T(3.0), T(-2.0)) == 0
        end
        c = ntuple(k -> SVector{8,Float64}(ntuple(i -> k + i, 8)), 8)
        @test allocs(lerp8, c, 0.3, 0.6, 0.9) == 0
        @test allocs(lerp8, map(v -> SVector{8,Float32}(v), c), 0.3f0, 0.6f0, 0.9f0) == 0
        acc = xsum_zeros(4, 3)
        @test allocs(add_cell, acc, 1.5e-3) == 0
        @test allocs(atomic_add_cell, acc, -7.25) == 0
        @test allocs(mc_uniform, primary_rng(7, 3)) == 0
    end

    @testset "isbits records" begin
        @test isbits(KerrSchildCartesian(0.9)) && isbits(st32)
        @test isbits(bin) && isbits(lane(bin, Float32))
        @test isbits(MarchBudget(80.0, 1100.0, Inf, 6000)) && isbits(MarchStep{Float64}(false, 1.0, 2.0))
        @test isbits(fr) && isbits(CameraFrame(PlaneParallelCamera(inc_deg = 17.0)))
        @test isbits(Minkowski()) && isbits(MCRng(1, 2))
        @test !isbits(grid)                                   # host data; isbits after adapt to a device
    end

    @testset "lanes" begin
        @test st.r_h === 1 + sqrt(1 - 0.9 * 0.9) && horizon_radius(st) === st.r_h
        @test st32.a === 0.9f0 && st32.r_h === Float32(st.r_h)
        @test lane(st, Float64) === st
        @test lane(b, Float32) === MarchBudget{Float32}(80.0f0, 1100.0f0, Inf32, 6000)
        b32b = lane(bin, Float32)
        @test b32b.r_h == Float32.(bin.r_h) && b32b.center[1] == SVector{3,Float32}(3, 0, 0)
        @test lane(grid, Float32) === grid                    # data untouched
        @test lane(Minkowski(), Float32) === Minkowski{Float32}()
        @test march_control(st32, y32, Inf32, b32) isa MarchStep{Float32}
        @test_throws MethodError march_control(st32, y32, Inf, b32)    # no silent promotion
        @test_throws MethodError rk4_step(st32, y32, 0.1)
        @test rk4_step(st32, y32, 0.1f0) isa SVector{8,Float32}
    end

    @testset "Dual states and spins" begin
        h = 0.05
        J = ForwardDiff.jacobian(yy -> rk4_step(st, yy, convert(eltype(yy), h)), y)
        for k in (2, 6)
            e = SVector{8}(ntuple(i -> i == k ? 1e-6 : 0.0, 8))
            fd = (rk4_step(st, y + e, h) - rk4_step(st, y - e, h)) / 2e-6
            @test maximum(abs.(J[:, k] - fd)) < 1e-6
        end
        f(a) = rk4_step(KerrSchildCartesian(a), SVector{8,typeof(a)}(y), typeof(a)(h))[6]
        dA = ForwardDiff.derivative(f, 0.9)
        @test dA ≈ (f(0.9 + 1e-6) - f(0.9 - 1e-6)) / 2e-6 rtol = 1e-5
        @test ForwardDiff.derivative(a -> KerrSchildCartesian(a).r_h, 0.9) ≈ -0.9 / sqrt(1 - 0.81)
        # a Dual position on the stationary closed forms: nested seeding stays consistent
        g(z) = raise_and_norm(st, SVector(0.0, 7.0, 3.0, z), SVector(1.0, -0.9, 0.05, 0.02))[2]
        @test ForwardDiff.derivative(g, 2.0) ≈ (g(2.0 + 1e-6) - g(2.0 - 1e-6)) / 2e-6 rtol = 1e-6
        @test ForwardDiff.derivative(z -> geodesic_rhs(st, SVector(0.0, 7.0, 3.0, z, 1.0, -0.9, 0.05, 0.02))[8], 2.0) ≈
              (geodesic_rhs(st, y + SVector(0, 0, 0, 1e-6, 0, 0, 0, 0))[8] -
               geodesic_rhs(st, y - SVector(0, 0, 0, 1e-6, 0, 0, 0, 0))[8]) / 2e-6 rtol = 1e-5
    end

    @testset "ADM grid on another array type" begin
        # the sampling body reads any AbstractArray: a view stands in for a device array
        gv = ADMGrid(grid.x0, grid.y0, grid.z0, grid.dx, grid.dy, grid.dz, grid.nx, grid.ny, grid.nz,
                     view(grid.data, :, :, :, :), grid.alpha_min, grid.offgrid_eta)
        ys = [SVector(0.0, 7.0 * sin(k), 6.0 * cos(k), 3.0 * sin(2k), 1.0, -0.5, 0.3, 0.1) for k in 1:50]
        @test all(metric_and_inverse(gv, SVector(yy[1], yy[2], yy[3], yy[4])) ===
                  metric_and_inverse(grid, SVector(yy[1], yy[2], yy[3], yy[4])) for yy in ys)
        @test all(rk4_step(gv, yy, 0.1) === rk4_step(grid, yy, 0.1) for yy in ys)
        @test all(march_control(gv, yy, Inf, b) === march_control(grid, yy, Inf, b) for yy in ys)
        ga = Adapt.adapt(Array, grid)
        @test ga isa ADMGrid{Float64,Array{Float64,4}} && ga.data == grid.data
        @test ADMGrid(grid; alpha_min = 0.65).alpha_min == 0.65
    end
end
