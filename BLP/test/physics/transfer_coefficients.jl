# Transfer coefficients of decoded samples: the unpolarized and polarized functions agree on
# Stokes I, invalid samples are no-ops of the documented types, the populations add, the
# kinematics narrow exactly out of a Dual state, and mass-scale derivatives match finite
# differences.
using Test, BLP, StaticArrays, LinearAlgebra, Random, ForwardDiff
using BLP.Transfer: StokesI, IQUV, PowerLaw, Kappa, TransferParams, coefficients, kinematic, couple
using BLP.Numerics: Units, MSUN, GNEWT, CL
using BLP.Spacetime: KerrSchildCartesian, metric_and_inverse, null_covariant, ks_radius
using BLP.Matter: PlasmaPrimitives, PlasmaState, ElectronModel, PlasmaCuts, Bound, decode

const M_BH = 6.2e9 * MSUN
const L_U = GNEWT * M_BH / CL^2
const MS = 3.0e-18 * L_U^3
const NU = 230.0e9

params(; ms = MS, thermal_frac = 1.0, powerlaw = nothing, kappa = nothing, cuts = PlasmaCuts()) =
    TransferParams(Units(M_BH, ms), ElectronModel(thermal_frac = thermal_frac), cuts, NU;
                   powerlaw = powerlaw, kappa = kappa)

function sample(rng, st)
    while true
        x = SVector(0.0, 15.0 .* (2.0 .* rand(rng, SVector{3,Float64}) .- 1.0)...)
        ks_radius(st, x) > 2.0 || continue
        g = metric_and_inverse(st, x)
        k, ok = null_covariant(st, x, randn(rng, SVector{3,Float64}))
        ok || continue
        kcov = k / (-k[1])
        rho = exp(randn(rng))
        prim = PlasmaPrimitives{Float64}(rho, 0.3randn(rng, SVector{3,Float64}), 0.1rho,
                                         0.3sqrt(rho) * randn(rng, SVector{3,Float64}), true)
        return (; prim, g, kcov, kcon = g[2] * kcov)
    end
end

@testset "transfer coefficients" begin
    rng = Xoshiro(11)
    st = KerrSchildCartesian(0.9)
    p = params()

    @testset "Stokes I of both modes" begin
        worst = 0.0
        for _ in 1:300
            s = sample(rng, st)
            state = decode(s.prim, s.g, p.units, p.electrons, p.cuts)
            state.valid || continue
            j, a = coefficients(StokesI(), p, state, s.kcov, NU)
            jp, ap, rp, tet, valid = coefficients(IQUV(), p, state, s.kcov, s.kcon, s.g, NU)
            @test valid && j > 0 && a > 0
            @test jp[3] == 0.0 && ap[3] == 0.0 && rp[1] == 0.0 && rp[3] == 0.0
            worst = max(worst, abs(jp[1] / j - 1), abs(ap[1] / a - 1))
            # renormalising e_(1) on single Kerr-Schild leaves the coefficients unchanged
            jr, ar, rr, _, _ = coefficients(IQUV(), p, state, s.kcov, s.kcon, s.g, NU, true)
            @test jr ≈ jp rtol = 1e-10
            @test rr ≈ rp rtol = 1e-10
        end
        @test worst < 1e-6
    end

    @testset "invalid samples" begin
        s = sample(rng, st)
        for prim in (PlasmaPrimitives{Float64}(), PlasmaPrimitives{Float64}(1.0, s.prim.uu, 0.1, zero(s.prim.bb), true),
                     PlasmaPrimitives{Float64}(1.0, s.prim.uu, -0.1, s.prim.bb, true))
            state = decode(prim, s.g, p.units, p.electrons, p.cuts)
            @test !state.valid
            @test coefficients(StokesI(), p, state, s.kcov, NU) === (0.0, 0.0)
            j, a, r, tet, valid = coefficients(IQUV(), p, state, s.kcov, s.kcon, s.g, NU)
            @test !valid && j === a === r === SVector(0.0, 0.0, 0.0, 0.0) && tet === zero(SMatrix{4,4,Float64})
        end
        pc = params(cuts = PlasmaCuts(sigma = Bound(-Inf, 0.0)))     # every magnetised sample is cut
        state = decode(s.prim, s.g, pc.units, pc.electrons, pc.cuts)
        @test !state.valid && coefficients(StokesI(), pc, state, s.kcov, NU) === (0.0, 0.0)
        pd = params(ms = ForwardDiff.Dual{Nothing}(MS, 1.0))
        sd = decode(PlasmaPrimitives{Float64}(), s.g, pd.units, pd.electrons, pd.cuts)
        @test coefficients(StokesI(), pd, sd, s.kcov, NU) isa NTuple{2,typeof(pd.units.dens_unit)}
    end

    @testset "field along the photon" begin
        g = SMatrix{4,4}(Diagonal(SVector(-1.0, 1.0, 1.0, 1.0)))
        u = SVector(1.0, 0.0, 0.0, 0.0); k = SVector(1.0, 1.0, 0.0, 0.0); bv = SVector(0.0, 1.0, 0.0, 0.0)
        state = PlasmaState{Float64}(1.0e6, 10.0, 30.0, u, g * u, bv, g * bv, 1.0, 1.0, 1.0, true)
        j, α, ρ, tet, valid = coefficients(IQUV(), p, state, g * k, k, (g, g), NU)
        @test valid && all(isfinite, tet)
        @test all(isfinite, j) && all(isfinite, α) && all(isfinite, ρ)
        @test maximum(abs, j) < 1e-25                                 # sin θ_B = 0: no emission
        @test all(isfinite, couple(IQUV(), SVector(1.0, 0.1, 0.0, 0.0), j, α, ρ, 1.0e10))
        @test coefficients(StokesI(), p, state, g * k, NU)[1] == 0.0
    end

    @testset "populations add" begin
        pl = PowerLaw(0.3, 3.5, 25.0, 1.0e7); kp = Kappa(0.3, 4.2, 0.5)
        pt = params(); pp = params(thermal_frac = 0.0, powerlaw = pl); pk = params(thermal_frac = 0.0, kappa = kp)
        pall = params(powerlaw = pl, kappa = kp)
        pz = params(powerlaw = PowerLaw(0.0, 3.5, 25.0, 1.0e7))
        for _ in 1:100
            s = sample(rng, st)
            dec(q) = decode(s.prim, s.g, q.units, q.electrons, q.cuts)
            dec(pt).valid || continue
            jt, at = coefficients(StokesI(), pt, dec(pt), s.kcov, NU)
            jl, al = coefficients(StokesI(), pp, dec(pp), s.kcov, NU)
            jk, ak = coefficients(StokesI(), pk, dec(pk), s.kcov, NU)
            @test coefficients(StokesI(), pall, dec(pall), s.kcov, NU) === ((jt + jl) + jk, (at + al) + ak)
            @test coefficients(StokesI(), pz, dec(pz), s.kcov, NU) === (jt, at)
            P(q) = coefficients(IQUV(), q, dec(q), s.kcov, s.kcon, s.g, NU)
            @test P(pall)[1] ≈ P(pt)[1] + P(pp)[1] + P(pk)[1] rtol = 1e-14
            @test P(pz)[1:3] === P(pt)[1:3]
        end
    end

    @testset "mass-scale derivatives" begin
        f(m, q) = (u = Units(M_BH, m); TransferParams(u, q.electrons, q.cuts, NU; powerlaw = q.powerlaw))
        q = params(powerlaw = PowerLaw(0.3, 3.5, 25.0, 1.0e7))
        n = 0
        for _ in 1:60
            s = sample(rng, st)
            jfun(m) = (pm = f(m, q); coefficients(StokesI(), pm, decode(s.prim, s.g, pm.units, pm.electrons, pm.cuts), s.kcov, NU))
            j0 = jfun(MS)
            j0[1] > 0 || continue
            jd = jfun(ForwardDiff.Dual{Nothing}(MS, 1.0))
            @test all(ForwardDiff.value.(jd) .≈ j0)
            h = 1e-6 * MS
            fd = (jfun(MS + h)[1] - jfun(MS - h)[1]) / 2h
            @test ForwardDiff.partials(jd[1], 1) ≈ fd rtol = 1e-5
            # the kinematics of the Dual state are the Float64 ones
            pm = f(ForwardDiff.Dual{Nothing}(MS, 1.0), q)
            sd = decode(s.prim, s.g, pm.units, pm.electrons, pm.cuts)
            s64 = decode(s.prim, s.g, q.units, q.electrons, q.cuts)
            @test kinematic(Float64, sd.ucon) == s64.ucon && kinematic(Float64, sd.b_sq) == s64.b_sq
            @test kinematic(Float64, s64.bcov) === s64.bcov
            n += 1
        end
        @test n > 20
    end
end
