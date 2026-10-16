# Oracle: the metric layer, the geodesic right-hand sides, the RK4 step and the march
# control of BLP.Spacetime against BlackLightPlus.Geodesics/Radiation, bit for bit, on
# random states of Kerr (four spins, fp64 and the fp32 lane), two superposed binaries,
# an ADM grid built from Kerr, a user metric and flat space.
Base.include(@__MODULE__, joinpath(@__DIR__, "helpers.jl"))

# Each call is compiled as its own non-inlined specialization, so new and old bodies
# are compiled in the same context.
@noinline callf(f::F, args...) where {F} = f(args...)
# The result, or the exception type (a negative discriminant throws in the real sqrt).
tryf(f, args...) = try callf(f, args...) catch err; typeof(err) end

const GIMG = scratch_module(:GimgStop)
Base.include(GIMG, joinpath(OLD_SRC, "gpu_metric.jl"))

# Spacetime pairs (new, old).
kerr_pair(a) = (NS.KerrSchildCartesian(a), OG.KerrSchildCartesian(a))
function binary_pair(m, s, c)
    return (NS.SuperposedKerrSchild(m, s, c), OG.SuperposedKerrSchild(m, s, c))
end
const BB_PHI0 = -0.2233022
const BB_C1 = 12.5 .* (cos(BB_PHI0), sin(BB_PHI0), 0.0)
const BINARIES = (([0.5, 0.5], [0.0, 0.0], [(3.0, 0.0, 0.0), (-3.0, 0.0, 0.0)]),
                  ([0.5, 0.5], [0.0, 0.0], [BB_C1, (-BB_C1[1], -BB_C1[2], 0.0)]),
                  ([0.55, 0.45], [0.4, -0.3], [(-6.0, 0.0, 0.0), (6.0, 0.0, 0.0)]))

randbudget(rng) = (C = rand(rng, (20.0, 40.0, 80.0)), escape = rand(rng, (1100.0, 30.0, 10.0)),
                   R_em = rand(rng, (Inf, 12.0, 25.0)))

# The metric layer and the right-hand sides at the states `ys`.
function check_metric_layer(stn, sto, ys, rng; transport = true, gradients = 10)
    nbad = 0
    for (k, y) in enumerate(ys)
        x = SVector(y[1], y[2], y[3], y[4]); p = SVector(y[5], y[6], y[7], y[8])
        v = randn(rng, SVector{4,Float64})
        ok = same(callf(NS.metric, stn, x), callf(OG.metric, sto, x)) &&
             same(callf(NS.inverse_metric, stn, x), callf(OG.inverse_metric, sto, x)) &&
             same(callf(NS.metric_and_inverse, stn, x), callf(OG.metric_and_inverse, sto, x)) &&
             same(callf(NS.lower_index, stn, x, v), callf(OG.lower_index, sto, x, v)) &&
             same(callf(NS.raise_index, stn, x, p), callf(OG.raise_index, sto, x, p)) &&
             same(callf(NS.raise_and_norm, stn, x, p), callf(OG.raise_and_norm, sto, x, p)) &&
             same(callf(NS.null_error, stn, y), callf(OG.null_error, sto, y)) &&
             same(callf(NS.geodesic_rhs, stn, y), callf(OG.geodesic_rhs, sto, y))
        h = 10.0^(-3 + 3 * rand(rng))
        ok &= same(callf(NS.rk4_step, stn, y, h), callf(OG.rk4_step_fixed, sto, y, h))
        ps = SVector(p[2], p[3], p[4])
        ok &= same(tryf(NS.null_covariant, stn, x, ps), tryf(OG.null_covariant, sto, x, ps))
        if transport
            vs = (randn(rng, SVector{4,Float64}), randn(rng, SVector{4,Float64}))
            ok &= same(callf(NS.geodesic_transport_rhs, stn, y, vs),
                       callf(OG.geodesic_transport_rhs, sto, y, vs))
        end
        if k % gradients == 0
            ok &= same(callf(NS.inverse_metric_gradient, stn, x),
                       callf(OG.inverse_metric_gradient, sto, x))
            ok &= same(callf(NS.metric_gradient, stn, x), callf(OG.metric_gradient, sto, x))
            ok &= same(callf(NS.christoffel, stn, x), callf(OG.christoffel, sto, x))
        end
        nbad += !ok
    end
    return nbad
end

const nthrow = Ref(0)

# march_control against an old control `oldctl(b) -> control`, at states `ys`.
function check_march(stn, ys, rng, oldctl, radius; T = Float64)
    nbad = 0
    for y in ys
        b = randbudget(rng)
        R = radius(y)
        r_prev = rand(rng, (Inf, R * (1 + 1e-3), R * (1 - 1e-3), R))
        bn = NS.lane(NS.MarchBudget(b.C, b.escape, b.R_em, 6000), T)
        m = tryf(NS.march_control, stn, SVector{8,T}(y), T(r_prev), bn)
        o = tryf(OR.march_control, oldctl(b), SVector{8,T}(y), T(r_prev), 1)
        nbad += !(m isa Type ? m === o : same((m.stop, m.h, m.R), o))
        nthrow[] += m isa Type
    end
    return nbad
end


# The fixed-control sampled tracer of the old package, written with the new
# `march_control` and `rk4_step` in the same loop shape (threaded closure, push!,
# source → camera reversal), so the two compile alike.
function new_sampled(st, rays::Vector{SVector{8,Float64}}; C, escape, nmax, R_em = Inf)
    n = length(rays)
    states = Vector{Vector{SVector{8,Float64}}}(undef, n)
    dls = Vector{Vector{Float64}}(undef, n)
    b = NS.MarchBudget(C, escape, R_em, nmax)
    Threads.@threads for i in 1:n
        ys = SVector{8,Float64}[rays[i]]
        hs = Float64[]
        sizehint!(ys, 1024)
        sizehint!(hs, 1024)
        y = rays[i]
        r_prev = Inf
        for _ in 1:nmax
            m = NS.march_control(st, y, r_prev, b)
            m.stop && break
            r_prev = m.R
            h = min(m.h, Inf)
            y = NS.rk4_step(st, y, h)
            push!(ys, y)
            push!(hs, h)
        end
        k = length(ys)
        rev = Vector{SVector{8,Float64}}(undef, k)
        dl = Vector{Float64}(undef, k)
        for j in 1:k
            rev[j] = ys[k+1-j]
        end
        for j in 1:k-1
            dl[j] = hs[k-j]
        end
        dl[k] = 0.0
        states[i] = rev
        dls[i] = dl
    end
    return states, dls
end

# fp32-lane trajectories under the Kerr march control, reference and new, in one loop shape.
function lane_trajectory(step, control, y0::SVector{8,Float32}, nmax)
    ys = [y0]; y = y0; r_prev = Inf32
    for it in 1:nmax
        stop, h, r = control(y, r_prev)
        stop && break
        r_prev = r
        y = step(y, h)
        push!(ys, y)
    end
    return ys
end

ksr(a) = y -> first(OG.ks_radius_energy(y, a))
coordr(y) = sqrt(y[2]^2 + y[3]^2 + y[4]^2)

@testset "oracle: Spacetime" begin
    rng = Xoshiro(20261016)

    @testset "Kerr a = $a" for a in (0.0, 0.5, 0.9, 0.998)
        stn, sto = kerr_pair(a)
        r_h = OG.horizon_radius(sto)
        @test stn.r_h === r_h
        ys = random_states(rng, sto, [SVector(0.0, 0.0, 0.0)], 3000; rlo = 0.8, rhi = 40.0)
        append!(ys, random_states(rng, sto, [SVector(0.0, 0.0, 0.0)], 300; rlo = 900.0, rhi = 1300.0))
        @test check_metric_layer(stn, sto, ys, rng) == 0
        @test all(same(callf(NS.ks_radius, stn, SVector(y[1], y[2], y[3], y[4])),
                       callf(OG.ks_radius, sto, SVector(y[1], y[2], y[3], y[4]))) for y in ys)
        @test all(same(NS.ks_radius_energy(y, a), OG.ks_radius_energy(y, a)) for y in ys)
        @test all(same(NS.normal_frame_energy(stn, y), OG.normal_frame_energy(sto, y)) for y in ys)
        # the closed-form Kerr rule, against the Kerr march control and the device stop
        @test check_march(stn, ys, rng, b -> OR.KerrMarch(a, r_h, b.C, b.escape, b.R_em), ksr(a)) == 0
        # the same rule through the generic spacetime control
        @test check_march(stn, ys, rng, b -> OR.SpacetimeMarch(sto, b.C, b.escape, b.R_em), ksr(a)) == 0
        nstop = 0
        for y in ys
            b = randbudget(rng); R = ksr(a)(y)
            r_prev = rand(rng, (Inf, R * (1 + 1e-3), R * (1 - 1e-3)))
            m = NS.march_control(stn, y, r_prev, NS.MarchBudget(b.C, b.escape, b.R_em, 6000))
            s, r = GIMG._gimg_stop(y, r_prev, r_h, b.escape, b.R_em, a)
            nstop += !(m.stop === s && m.R === r && m.h === (r - r_h) / b.C)
        end
        @test nstop == 0
    end

    @testset "Kerr fp32 lane a = $a" for a in (0.0, 0.5, 0.9, 0.998)
        stn = NS.lane(NS.KerrSchildCartesian(a), Float32)
        sto = OG.KerrSchildCartesian(Float32(a))
        r_h = 1 + sqrt(1 - a * a)
        @test stn.a === Float32(a) && stn.r_h === Float32(r_h)
        ys64 = random_states(rng, OG.KerrSchildCartesian(a), [SVector(0.0, 0.0, 0.0)], 2000;
                             rlo = 0.8, rhi = 40.0)
        nbad = 0
        for y64 in ys64
            y = SVector{8,Float32}(y64); h = Float32(10.0^(-3 + 3 * rand(rng)))
            vs = (randn(rng, SVector{4,Float32}), randn(rng, SVector{4,Float32}))
            nbad += !(same(callf(NS.geodesic_rhs, stn, y), callf(OG.geodesic_rhs, sto, y)) &&
                      same(callf(NS.rk4_step, stn, y, h), callf(OG.rk4_step_fixed, sto, y, h)) &&
                      same(callf(NS.geodesic_transport_rhs, stn, y, vs),
                           callf(OG.geodesic_transport_rhs, sto, y, vs)))
        end
        @test nbad == 0
        @test check_march(stn, ys64, rng,
                          b -> OR.KerrMarch(Float32(a), Float32(r_h), Float32(b.C),
                                            Float32(b.escape), Float32(b.R_em)),
                          y -> Float64(first(OG.ks_radius_energy(SVector{8,Float32}(y), Float32(a))));
                          T = Float32) == 0
    end

    @testset "superposed binary $i" for (i, (m, s, c)) in enumerate(BINARIES)
        stn, sto = binary_pair(m, s, c)
        @test all(stn.r_h[A] === OG.horizon_radius_hole(sto, A) for A in 1:2)
        ctrs = [SVector(cc...) for cc in c]
        ys = random_states(rng, sto, ctrs, 3000; rlo = 0.4, rhi = 30.0)
        @test check_metric_layer(stn, sto, ys, rng) == 0
        @test all(same(NS.ks_radius(stn, SVector(y[1], y[2], y[3], y[4]), A),
                       OG.ks_radius_hole(sto, SVector(y[1], y[2], y[3], y[4]), A))
                  for y in ys, A in 1:2)
        @test check_march(stn, ys, rng, b -> OR.SpacetimeMarch(sto, b.C, b.escape, b.R_em),
                          coordr) == 0
        @info "binary $i: march_control states where both implementations throw" nthrow[]
        nthrow[] = 0
    end

    @testset "ADM grid from Kerr" begin
        kn, ko = kerr_pair(0.9)
        xs = -20.0:0.5:20.0
        gn = NS.adm_grid_from_metric(x -> NS.metric(kn, x), xs, xs, xs; alpha_min = 0.1,
                                     excise = x -> NS.ks_radius(kn, x) < NS.horizon_radius(kn))
        go = OG.adm_grid_from_metric(x -> OG.metric(ko, x), xs, xs, xs; alpha_min = 0.1,
                                     excise = x -> OG.ks_radius(ko, x) < OG.horizon_radius(ko))
        @test reinterpret(UInt64, vec(gn.data)) == reinterpret(UInt64, vec(go.data))
        @test (gn.x0, gn.dx, gn.nx, gn.alpha_min, gn.offgrid_eta) ===
              (go.x0, go.dx, go.nx, go.alpha_min, go.offgrid_eta)
        ys = random_states(rng, ko, [SVector(0.0, 0.0, 0.0)], 2000; rlo = 0.5, rhi = 19.0)
        append!(ys, random_states(rng, ko, [SVector(0.0, 0.0, 0.0)], 200; rlo = 34.0, rhi = 60.0))
        @test check_metric_layer(gn, go, ys, rng; gradients = 20) == 0
        @test all(same(callf(NS.sample_adm, gn, y[2], y[3], y[4]),
                       callf(OG.sample_adm, go, y[2], y[3], y[4])) for y in ys)
        @test check_march(gn, ys, rng, b -> OR.SpacetimeMarch(go, b.C, b.escape, b.R_em),
                          coordr) == 0
        # a grid whose capture lapse differs from its build lapse
        gn2 = NS.ADMGrid(gn; alpha_min = 0.65)
        @test check_march(gn2, ys, rng,
                          b -> OR.SpacetimeMarch(OG.ADMGridSpacetime(xs, xs, xs, go.data;
                                                                    alpha_min = 0.65),
                                                 b.C, b.escape, b.R_em), coordr) == 0
    end



    @testset "fp32 trajectories" begin
        a = 0.9375; r_h = 1 + sqrt(1 - a * a)
        sto = OG.KerrSchildCartesian(Float32(a)); stn = NS.lane(NS.KerrSchildCartesian(a), Float32)
        km = OR.KerrMarch(Float32(a), Float32(r_h), 80.0f0, 1100.0f0, Inf32)
        bn = NS.lane(NS.MarchBudget(80.0, 1100.0, Inf, 6000), Float32)
        rays = OG.camera_rays(OG.KerrSchildCartesian(a), OG.PlaneParallelCamera(60.0, 1000.0, 30.0, 16))
        nbad = 0
        for y64 in rays
            y0 = SVector{8,Float32}(y64)
            to = lane_trajectory((y, h) -> OG.rk4_step_fixed(sto, y, h),
                                 (y, rp) -> OR.march_control(km, y, rp, 1), y0, 6000)
            tn = lane_trajectory((y, h) -> NS.rk4_step(stn, y, h),
                                 (y, rp) -> (m = NS.march_control(stn, y, rp, bn); (m.stop, m.h, m.R)),
                                 y0, 6000)
            nbad += !(length(to) == length(tn) && all(same.(to, tn)))
        end
        @test nbad == 0
    end

    @testset "sampled trajectories" begin
        for (stn, sto, cam, kw) in
                ((kerr_pair(0.9375)..., OG.PlaneParallelCamera(60.0, 1000.0, 40.0, 32),
                  (C = 40.0, escape = 1.1e3, nmax = 6000)),
                 (kerr_pair(0.5)..., OG.PlaneParallelCamera(17.0, 1000.0, 24.0, 24),
                  (C = 80.0, escape = 1.1e3, nmax = 2500, R_em = 30.0)),
                 (binary_pair(BINARIES[2]...)..., OG.PlaneParallelCamera(90.0, 1000.0, 40.0, 24),
                  (C = 80.0, escape = 1100.0, nmax = 2500)),
                 (binary_pair(BINARIES[1]...)..., OG.PlaneParallelCamera(60.0, 1000.0, 16.0, 16),
                  (C = 80.0, escape = 1100.0, nmax = 6000, R_em = 12.0)))
            rays = OG.camera_rays(sto, cam)
            so, do_ = OG.trace_rays_sampled(sto, rays; kw...)
            sn, dn = new_sampled(stn, rays; kw...)
            @test all(sn[i] == so[i] && all(same.(sn[i], so[i])) for i in eachindex(rays))
            @test all(reinterpret(UInt64, dn[i]) == reinterpret(UInt64, do_[i]) for i in eachindex(rays))
            @test maximum(length, so) > 50
        end
    end


    @testset "adaptive solves" begin
        kn, ko = kerr_pair(0.9)
        xs = -20.0:0.5:20.0
        gn = NS.adm_grid_from_metric(x -> NS.metric(kn, x), xs, xs, xs; alpha_min = 0.1,
                                     excise = x -> NS.ks_radius(kn, x) < NS.horizon_radius(kn))
        go = OG.adm_grid_from_metric(x -> OG.metric(ko, x), xs, xs, xs; alpha_min = 0.1,
                                     excise = x -> OG.ks_radius(ko, x) < OG.horizon_radius(ko))
        # (new, reference, start radius, first impact parameter, keywords); the user
        # metric has no horizon, so its rays pass outside the capture radius
        cases = Any[(kn, ko, 100.0, -6.0, (escape = 1100.0,)),
                    (binary_pair(BINARIES[2]...)..., 100.0, -6.0, (escape = 400.0,)),
                    (gn, go, 15.0, -6.0, (escape = 19.0,)),
                    (NS.UserMetric(x -> NS.metric(kn, x)), OG.UserMetric(x -> OG.metric(ko, x)), 100.0,
                     5.0, (escape = 110.0,))]
        for (stn, sto, d, b0, kw) in cases
            nbad = 0
            for k in 1:6
                x0 = SVector(0.0, d, b0 + 2.0 * k, 1.3)
                p = OG.metric(sto, x0) * SVector(1.0, -1.0, 0.0, 0.0)
                y0 = vcat(x0, OG.nullify_covariant(sto, x0, SVector(p[2], p[3], p[4])))
                sn = NS.trace_geodesic(stn, y0; kw..., save_everystep = true)
                so = OG.trace_geodesic(sto, y0; kw..., save_everystep = true)
                nbad += !(sn.t == so.t && all(same.(sn.u, so.u)) && sn.retcode == so.retcode)
            end
            @test nbad == 0
        end
    end

    @testset "user metric and flat space" begin
        kn, ko = kerr_pair(0.5)
        un = NS.UserMetric(x -> NS.metric(kn, x)); uo = OG.UserMetric(x -> OG.metric(ko, x))
        ys = random_states(rng, ko, [SVector(0.0, 0.0, 0.0)], 1000; rlo = 2.5, rhi = 40.0)
        @test check_metric_layer(un, uo, ys, rng; transport = false) == 0
        mn = NS.Minkowski(); mo = OG.Minkowski()
        @test check_metric_layer(mn, mo, ys, rng; transport = false) == 0
        # the transport of flat space reads partials of the constant η: both throw
        vs = (randn(rng, SVector{4,Float64}), randn(rng, SVector{4,Float64}))
        @test tryf(NS.geodesic_transport_rhs, mn, ys[1], vs) ===
              tryf(OG.geodesic_transport_rhs, mo, ys[1], vs) === BoundsError
    end
end
