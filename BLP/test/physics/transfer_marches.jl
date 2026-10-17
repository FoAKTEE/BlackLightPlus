# The transfer marches on a torus over Kerr: Stokes I of both modes, the stored-path and
# fused marches, the fp32 geodesic lane (type flow and accuracy), mass-scale and
# temperature-ratio derivatives against finite differences, the mass-scale ladder,
# analytic providers, vacuum, and the hot-path contract.
using Test, BLP, StaticArrays, LinearAlgebra, Statistics, ForwardDiff
using BLP.Transfer: StokesI, IQUV, PowerLaw, TransferParams, SpacetimeMarch, StoredPath, march,
                    output_type
using BLP.Numerics: Units, MSUN, GNEWT, CL
using BLP.Spacetime: KerrSchildCartesian, MarchBudget, PlaneParallelCamera, CameraFrame, lane
using BLP.Matter: GridProvider, AnalyticProvider, PlasmaPrimitives, ElectronModel, PlasmaCuts, Bound,
                  decode
import BLP.Matter: primitives, plasma_state, domain_radius
const Decks = Main.Decks
const Prog = Decks.MarchProgram
dual(x) = ForwardDiff.Dual{Nothing}(x, 1.0)

const M_BH = 6.2e9 * MSUN
const L_U = GNEWT * M_BH / CL^2
const MS = 3.0e-18 * L_U^3
const NU = 230.0e9

params(; ms = MS, electrons = ElectronModel(), powerlaw = nothing) =
    TransferParams(Units(M_BH, ms), electrons, PlasmaCuts(sigma = Bound(-Inf, 1.0)), NU; powerlaw = powerlaw)

"The torus of `Decks.TorusDeck` supplied as a cgs state."
struct AnalyticTorus <: AnalyticProvider end
plasma_state(::AnalyticTorus, st, x, g, p) = decode(primitives(Decks.TorusDeck(), x), g, p.units, p.electrons, p.cuts)
domain_radius(::AnalyticTorus) = 11.0

"A provider without plasma."
struct Empty <: GridProvider end
primitives(::Empty, x::SVector{4}) = PlasmaPrimitives{Float64}()
domain_radius(::Empty) = 0.0

"A uniform ball of radius 20 (the fixed-geometry derivative deck)."
struct UniformBall <: GridProvider end
primitives(::UniformBall, x::SVector{4}) =
    sqrt(x[2] * x[2] + x[3] * x[3] + x[4] * x[4]) < 20.0 ?
    PlasmaPrimitives{Float64}(1.0, SVector(0.1, 0.0, 0.0), 0.1 / (5.0 / 3.0 - 1.0), SVector(0.5, 0.1, 0.0), true) :
    PlasmaPrimitives{Float64}()
domain_radius(::UniformBall) = 20.0

# Bytes allocated by one march, after a warm-up call compiles it.
allocs(mode, y0, prog) = (march(mode, y0, prog); @allocated march(mode, y0, prog))
image(mode, rays, prog) = [march(mode, y, prog) for y in rays]

@testset "transfer marches" begin
    a = 0.9375
    st = KerrSchildCartesian(a)
    cam = PlaneParallelCamera(inc_deg = 30.0, d = 500.0, fov = 24.0, npix = 12)
    frame = CameraFrame(cam)
    rays = Decks.camera_rays(st, cam)
    b = MarchBudget(80.0, 550.0, 12.0, 3000)
    p = params()
    prog = Prog(st, SpacetimeMarch(st, b), Decks.TorusDeck(), p, frame)
    I = image(StokesI(), rays, prog)
    S = image(IQUV(), rays, prog)

    @testset "Stokes I of both modes" begin
        @test output_type(StokesI(), p) === Float64 && eltype(I) === Float64
        @test eltype(S) === SVector{4,Float64}
        @test count(>(0), I) > 60 && all(>=(0), I)
        @test all(s -> norm(s[2:4]) <= s[1] * (1 + 1e-12), S)
        @test sum(abs(s[2]) + abs(s[3]) for s in S) > 0
        @test sum(s[1] for s in S) ≈ sum(I) rtol = 0.05
    end

    @testset "stored path and fused march" begin
        states, dls = Decks.sampled_trace(st, rays; C = b.C, escape = b.escape, nmax = b.nmax, R_em = b.R_em)
        sp(i) = Prog(st, StoredPath(st, states[i], dls[i]), Decks.TorusDeck(), p, frame)
        @test [march(StokesI(), states[i][end], sp(i)) for i in eachindex(rays)] ≈ I rtol = 1e-12
        Ss = [march(IQUV(), states[i][end], sp(i)) for i in eachindex(rays)]
        @test maximum(maximum(abs.(Ss[i] - S[i])) for i in eachindex(rays)) < 1e-9 * maximum(I)
    end

    @testset "fp32 geodesic lane" begin
        ctl32 = SpacetimeMarch(lane(st, Float32), lane(b, Float32))
        prog32 = Prog(st, ctl32, Decks.TorusDeck(), p, frame)
        rays32 = [SVector{8,Float32}(y) for y in rays]
        # the lane is set by the control: a Float64 state on the fp32 control is a MethodError
        @test_throws MethodError march(StokesI(), rays[1], prog32)
        @test_throws MethodError march(IQUV(), rays32[1], prog)
        # the transfer and the accumulator stay Float64
        I32 = image(StokesI(), rays32, prog32)
        S32 = image(IQUV(), rays32, prog32)
        @test eltype(I32) === Float64 && eltype(S32) === SVector{4,Float64}
        @test abs(sum(I32) - sum(I)) / sum(I) < 1e-4
        @test cor(I32, I) > 0.9999
        @test sum(abs2, I32 .- I) / sum(abs2, I) < 1e-6
        for k in 1:4
            @test cor([s[k] for s in S32], [s[k] for s in S]) > 0.999
        end
        @test abs(sum(s[1] for s in S32) - sum(s[1] for s in S)) / sum(s[1] for s in S) < 1e-4
    end

    @testset "mass-scale derivative (fixed geometry)" begin
        # an optically thin uniform ball: d_unit ∝ s, e_unit ∝ s, B_unit ∝ √s
        stb = KerrSchildCartesian(0.9)
        camb = PlaneParallelCamera(inc_deg = 60.0, d = 100.0, fov = 10.0, npix = 2)
        raysb = Decks.camera_rays(stb, camb)
        states, dls = Decks.sampled_trace(stb, raysb; C = 20.0, escape = 110.0, nmax = 6000)
        L = Units(1.0e3 * MSUN, 1.0).L_unit
        function F(s)
            q = TransferParams(Units(1.0e3 * MSUN, s * 1.0e-16 * L^3), ElectronModel(), PlasmaCuts(), NU)
            return sum(march(StokesI(), states[i][end],
                             Prog(stb, StoredPath(stb, states[i], dls[i]), UniformBall(), q, CameraFrame(camb)))
                       for i in eachindex(raysb))
        end
        d_ad = ForwardDiff.derivative(F, 1.0)
        @test F(1.0) > 0.0
        @test isfinite(d_ad) && d_ad != 0.0
        @test d_ad ≈ (F(1.0 + 1.0e-5) - F(1.0 - 1.0e-5)) / 2.0e-5 rtol = 1e-5
    end

    @testset "mass-scale and temperature-ratio lanes on the torus" begin
        D = typeof(dual(1.0))
        pd = params(ms = dual(MS))
        progd = Prog(st, SpacetimeMarch(st, b), Decks.TorusDeck(), pd, frame)
        Id = image(StokesI(), rays, progd)
        @test eltype(Id) === D
        @test ForwardDiff.value.(Id) ≈ I rtol = 1e-12
        h = 1e-6 * MS
        Ip = image(StokesI(), rays, Prog(st, SpacetimeMarch(st, b), Decks.TorusDeck(), params(ms = MS + h), frame))
        Im = image(StokesI(), rays, Prog(st, SpacetimeMarch(st, b), Decks.TorusDeck(), params(ms = MS - h), frame))
        @test sum(ForwardDiff.partials.(Id, 1)) ≈ (sum(Ip) - sum(Im)) / 2h rtol = 1e-5
        # the temperature ratio R_high: the value lane is the plain image and the thermal
        # fraction scales the emission
        er(f) = ElectronModel{D}(rat_high = dual(10.0), thermal_frac = f)
        img(f, e) = image(StokesI(), rays, Prog(st, SpacetimeMarch(st, b), Decks.TorusDeck(), params(electrons = e), frame))
        I5 = img(0.5, er(0.5)); I1 = img(1.0, er(1.0))
        @test ForwardDiff.value.(I1) ≈ I rtol = 1e-12
        @test ForwardDiff.value.(I5) ≈ img(0.5, ElectronModel(thermal_frac = 0.5)) rtol = 1e-12
        @test sum(ForwardDiff.value.(I5)) < sum(ForwardDiff.value.(I1))
        @test all(isfinite, ForwardDiff.partials.(I5, 1)) && any(!=(0.0), ForwardDiff.partials.(I5, 1))
    end

    @testset "mass-scale ladder" begin
        scales = (MS, 2.5 * MS, 0.4 * MS)
        ps = ntuple(k -> params(ms = scales[k]), 3)
        progl = Prog(st, SpacetimeMarch(st, b), Decks.TorusDeck(), ps, frame)
        @test output_type(StokesI(), ps) === SVector{3,Float64}
        Il = image(StokesI(), rays, progl)
        @test eltype(Il) === SVector{3,Float64}
        for k in 1:3
            Ik = image(StokesI(), rays, Prog(st, SpacetimeMarch(st, b), Decks.TorusDeck(), ps[k], frame))
            @test [v[k] for v in Il] ≈ Ik rtol = 1e-12
        end
        # the ladder decodes one sample of primitives per step: grid providers only
        @test_throws MethodError march(StokesI(), rays[1], Prog(st, SpacetimeMarch(st, b), AnalyticTorus(), ps, frame))
    end

    @testset "analytic provider and vacuum" begin
        proga = Prog(st, SpacetimeMarch(st, b), AnalyticTorus(), p, frame)
        @test image(StokesI(), rays, proga) ≈ I rtol = 1e-12
        @test maximum(maximum(abs.(x - y)) for (x, y) in zip(image(IQUV(), rays, proga), S)) < 1e-9 * maximum(I)
        proge = Prog(st, SpacetimeMarch(st, b), Empty(), p, frame)
        @test all(==(0.0), image(StokesI(), rays, proge))
        @test all(==(SVector(0.0, 0.0, 0.0, 0.0)), image(IQUV(), rays, proge))
    end

    @testset "hot path" begin
        y0 = rays[50]; y32 = SVector{8,Float32}(y0)
        prog32 = Prog(st, SpacetimeMarch(lane(st, Float32), lane(b, Float32)), Decks.TorusDeck(), p, frame)
        states, dls = Decks.sampled_trace(st, [y0]; C = b.C, escape = b.escape, nmax = b.nmax, R_em = b.R_em)
        progs = Prog(st, StoredPath(st, states[1], dls[1]), Decks.TorusDeck(), p, frame)
        D = typeof(dual(1.0))
        progd = Prog(st, SpacetimeMarch(st, b), Decks.TorusDeck(), params(ms = dual(MS)), frame)
        progl = Prog(st, SpacetimeMarch(st, b), Decks.TorusDeck(), (p, params(ms = 2MS)), frame)
        for mode in (StokesI(), IQUV())
            @test allocs(mode, y0, prog) == 0
            @test allocs(mode, y32, prog32) == 0
            @test allocs(mode, states[1][end], progs) == 0
        end
        @test allocs(StokesI(), y0, progd) == 0
        @test allocs(StokesI(), y0, progl) == 0
        for pr in (prog, prog32, progd, progl)
            @test isbits(pr) && Decks.device_isbits(pr)
        end
        @test !isbits(progs) && Decks.device_isbits(progs)
    end
end
