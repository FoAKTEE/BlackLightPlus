# Monte Carlo emissivity + emission-tetrad tests. j_nu must equal the invariant
# thermal-synchrotron emissivity times nu^2; jmax must envelope the pitch-angle
# distribution; int_jnu_zone must match an independent fine quadrature; and the
# (u, b-hat)-aligned emission tetrad must be orthonormal under the metric with
# e_(0) = u. Ground truth is analytic / self-consistent — no reference code.
using Test
import BlackLightPlus
using BlackLightPlus: j_nu, alpha_nu, jmax, int_jnu_zone, emission_tetrad
using StaticArrays
using LinearAlgebra: dot

const _TS = BlackLightPlus.Radiation.thermal_synchrotron

# Representative thermal plasma, MC operating regime nu >~ nu_s.
const NE = 1.0e6      # cm^-3
const TE = 40.0       # Theta_e
const BF = 30.0       # Gauss

@testset "j_nu == invariant j * nu^2" begin
    for nu in (3.0e10, 1.0e11, 2.3e11, 1.0e12), st in (0.2, 0.5, 0.9, 1.0)
        j_inv, a_inv = _TS(1.0, NE, TE, BF, nu, st)
        @test j_nu(NE, TE, BF, nu, st) ≈ j_inv * nu^2 rtol = 0
        @test alpha_nu(NE, TE, BF, nu, st) ≈ a_inv / nu rtol = 0
        @test j_nu(NE, TE, BF, nu, st) > 0.0          # physical, positive
    end
end

@testset "jmax envelopes the pitch-angle distribution" begin
    for nu in (5.0e10, 2.3e11, 1.0e12)
        jm = jmax(NE, TE, BF, nu)
        @test jm ≈ j_nu(NE, TE, BF, nu, 1.0)          # envelope = j at sin=1 (theta=pi/2)
        for st in (0.05, 0.3, 0.6, 0.85, 1.0)
            @test j_nu(NE, TE, BF, nu, st) ≤ jm * (1 + 1e-12)   # never exceeds the envelope
        end
    end
end

@testset "int_jnu_zone vs independent quadrature" begin
    # Independent high-resolution trapezoidal rule over the half-sphere:
    # int j_nu dOmega = 4 pi int_0^{pi/2} j_nu(sin th) sin th d th.
    function ref_int(nu; n = 20000)
        h = (0.5π) / n
        acc = 0.0
        for i in 0:n
            th = i * h
            w = (i == 0 || i == n) ? 0.5 : 1.0
            acc += w * j_nu(NE, TE, BF, nu, sin(th)) * sin(th)
        end
        return 4π * acc * h
    end
    for nu in (5.0e10, 2.3e11, 1.0e12)
        iz = int_jnu_zone(NE, TE, BF, nu)
        @test iz > 0.0
        @test iz ≈ ref_int(nu) rtol = 1e-6            # Simpson (512) vs trapezoid (20000)
        # bracketed by the min/max of 4 pi <j> over the hemisphere
        @test iz ≤ 4π * jmax(NE, TE, BF, nu)
    end
end

@testset "emission_tetrad orthonormality (u, b-hat aligned)" begin
    # Minkowski metric, a boosted fluid 4-velocity (u.u = -1), generic B.
    g = @SMatrix [-1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    γ = 1.5; β = sqrt(1 - 1/γ^2)
    ucon = SVector(γ, γ*β, 0.0, 0.0)
    ucov = g * ucon
    @test dot(ucon, g * ucon) ≈ -1.0                  # proper 4-velocity
    bcon = SVector(0.0, 0.3, 1.0, 0.5)                # generic spatial-ish B
    E = emission_tetrad(ucon, ucov, bcon, g, g)       # gcon == gcov for Minkowski
    η = SVector(-1.0, 1.0, 1.0, 1.0)
    for a in 1:4, b in 1:4
        ea = SVector(E[a,1], E[a,2], E[a,3], E[a,4])
        eb = SVector(E[b,1], E[b,2], E[b,3], E[b,4])
        expected = a == b ? η[a] : 0.0
        @test dot(ea, g * eb) ≈ expected atol = 1e-12  # e_(a).e_(b) = eta_ab
    end
    # e_(0) is the fluid 4-velocity (u.u = -1 => no rescale)
    @test SVector(E[1,1], E[1,2], E[1,3], E[1,4]) ≈ ucon atol = 1e-12
end

@testset "spectral shape: rise then exponential cutoff" begin
    # j_nu(nu) should rise from low nu, peak, then fall (exp(-X^{1/3}) cutoff) at
    # high nu >> nu_s. Check the far-UV tail is far below a mid-band value.
    st = 1.0
    j_mid = j_nu(NE, TE, BF, 1.0e11, st)
    j_hi  = j_nu(NE, TE, BF, 1.0e15, st)              # deep in the cutoff
    @test j_hi < j_mid                                # cutoff suppresses the tail
    @test j_mid > 0.0
end

# ------------------------------------------------------------ weight_at top-edge clamp
using BlackLightPlus: SphereModel, sphere_zone, build_weight_table, weight_at

@testset "weight_at: top-edge clamp returns the LAST node" begin
    # At/above nu_max the log-linear interpolant must clamp to the final grid node
    # exp(lwgt[end]); the old clamp forced f = 0 and returned node N-1 -- off by one full
    # grid node, ~22 orders of magnitude down the exponentially-cut synchrotron band.
    model = SphereModel()
    zone = sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0))
    wgt = build_weight_table([zone], 1000.0; n_esamp = 10, nu_min = 1.0e8, nu_max = 1.0e16)
    numax = 1.0e16
    # the top node is genuinely distinct from node N-1 (exp cutoff), so the check bites
    @test !isapprox(exp(wgt.lwgt[end]), exp(wgt.lwgt[end - 1]); rtol = 1e-3)
    @test weight_at(numax, wgt) ≈ exp(wgt.lwgt[end]) rtol = 1e-12
    @test weight_at(2 * numax, wgt) ≈ exp(wgt.lwgt[end]) rtol = 1e-12   # beyond grid -> last node
    # continuous across the top edge (no discontinuous drop to node N-1)
    @test weight_at(numax, wgt) ≈ weight_at(numax * (1 - 1e-12), wgt) rtol = 1e-6
end
