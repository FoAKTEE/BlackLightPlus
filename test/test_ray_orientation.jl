# Ray time-orientation and polarization-frame tests. A camera ray must be the physical
# photon that reaches the camera, traced backward: along the integration x^t decreases,
# its conserved ratio L_z/E is Bardeen's λ = -α sin i at image coordinate α (so the
# image is not mirrored), and a fluid element moving toward the camera is blueshifted,
# ν_fluid/ν_obs = γ(1 - v). The polarized checks fix the camera Stokes convention on
# optically thin slabs in near-flat space (a = 0, r ≈ 750): IAU/IEEE, north = +β,
# east = -α, EVPA χ = ½ atan2(U, Q) east of north. Synchrotron E-vectors are
# perpendicular to the sky-projected field; thermal electrons emit V > 0 when B points
# toward the observer (Marszewski+2021 J_V ∝ cos θ, IEEE); a Faraday screen whose field
# points toward the observer (RM > 0) rotates χ toward east. Every check runs on the
# CPU march and on the fused kernels (KernelAbstractions CPU backend).
using Test, StaticArrays, LinearAlgebra, KernelAbstractions
using BlackLightPlus
using BlackLightPlus: KerrSchildCartesian, PlaneParallelCamera, camera_rays_at,
                      trace_rays_sampled, SimCoeffConfig, geodesic_rhs,
                      angular_momentum_z, energy
using BlackLightPlus.Inputs: RunConfig, CameraConfig, PlasmaConfig
const RO = BlackLightPlus.Radiation

@isdefined(_gimg_thermal) || include(joinpath(@__DIR__, "..", "src", "gpu_image.jl"))
@isdefined(_gpol_thermal) || include(joinpath(@__DIR__, "..", "src", "gpu_pol.jl"))

const RO_CL = 2.99792458e10
const RO_L = 6.674e-8 * 6.5e9 * 1.989e33 / RO_CL^2    # r_g of a 6.5e9 M_sun hole [cm]
const RO_NU = 1.0e9                                    # observing frequency [Hz]
const RO_B = 100.0                                     # field strength [G] at |bb| = 0.01

@testset "camera ray is the backward-traced physical photon" begin
    for a in (0.0, 0.9), inc in (30.0, 60.0, 90.0)
        st = KerrSchildCartesian(a)
        cam = PlaneParallelCamera(inc_deg = inc, d = 1000.0, fov = 10.0, npix = 1)
        for (α, β) in ((4.0, 1.0), (-3.0, 2.5), (0.0, -5.0))
            y = camera_rays_at(st, cam, [(α, β)])[1]
            @test geodesic_rhs(st, y)[1] < 0                  # x^t decreases along the trace
            λ = angular_momentum_z(y) / energy(y)             # invariant under p -> -p
            @test λ ≈ -α * sind(inc) atol = 1e-9 * max(1.0, abs(α))   # Bardeen λ = -α sin i
        end
    end
end

# Slab geometry shared by every check below: camera at i = 90° (on +x, north = +z,
# west = +y = +α), one ray at image (α, β) = (500, 500), i.e. the line y = z = 500
# travelling toward -x. The emitting slab fills 400 < x < 600 around that line; the
# optional Faraday screen fills 650 < x < 700 (between slab and camera).
const RO_INC = 90.0
ro_st() = KerrSchildCartesian(0.0)
ro_cam() = PlaneParallelCamera(inc_deg = RO_INC, d = 1000.0, fov = 10.0, npix = 1)
ro_ray() = camera_rays_at(ro_st(), ro_cam(), [(500.0, 500.0)])
ro_inyz(x) = 400.0 < x[3] < 600.0 && 400.0 < x[4] < 600.0
ro_inslab(x) = ro_inyz(x) && 400.0 < x[2] < 600.0
ro_inscreen(x) = ro_inyz(x) && 650.0 < x[2] < 700.0
const RO_UP = SVector(0.0, -cosd(RO_INC), 0.0, sind(RO_INC))   # camera vertical (image β)

# rat_low = rat_high = 1 makes Θ_e independent of β; p/ρ = 1e-3 gives Θ_e ≈ 0.9.
ro_cfg(d_unit) = SimCoeffConfig(d_unit = d_unit, e_unit = d_unit * RO_CL^2,
                                b_unit = RO_B / 0.01, plasma_rat_low = 1.0,
                                plasma_rat_high = 1.0)

# CPU sampler: slab (density 1, velocity uu, field bb) plus optional screen.
function ro_sampler(uu, bb; screen_bb = nothing, screen_rho = 200.0)
    z3 = SVector(0.0, 0.0, 0.0)
    return x -> begin
        if ro_inslab(x)
            (rho = 1.0, pgas = 1.0e-3, uu = uu, bb = bb, inside = true)
        elseif screen_bb !== nothing && ro_inscreen(x)
            (rho = screen_rho, pgas = 1.0e-3 * screen_rho, uu = z3, bb = screen_bb, inside = true)
        else
            (rho = 0.0, pgas = 0.0, uu = z3, bb = z3, inside = false)
        end
    end
end

function ro_cpu_I(uu, bb; d_unit)
    states, dls = trace_rays_sampled(ro_st(), ro_ray(); C = 80.0, escape = 1.1e3)
    return RO.unpolarized_march(ro_st(), ro_cfg(d_unit), ro_sampler(uu, bb),
                                states[1], dls[1], RO_NU, RO_L)
end

function ro_cpu_S(bb; d_unit = 1.0e-31, screen_bb = nothing)
    states, dls = trace_rays_sampled(ro_st(), ro_ray(); C = 80.0, escape = 1.1e3)
    return RO.polarized_march(ro_st(), ro_cfg(d_unit), ro_sampler(SVector(0.0, 0.0, 0.0), bb;
                              screen_bb = screen_bb), states[1], dls[1], RO_NU, RO_L;
                              cam_up = RO_UP)
end

# Fused-kernel twin: the same slab/screen on a uniform grid over the cube [400, 720]³.
const RO_NG = 33
const RO_XO = 400.0
const RO_INVDX = (RO_NG - 1) / 320.0
function ro_grid(uu, bb; screen_bb = nothing, screen_rho = 200.0, gamma = 5 / 3)
    host = zeros(Float32, 8, RO_NG, RO_NG, RO_NG)
    for i in 1:RO_NG, j in 1:RO_NG, k in 1:RO_NG
        x = SVector(0.0, RO_XO + (i - 1) / RO_INVDX, RO_XO + (j - 1) / RO_INVDX,
                    RO_XO + (k - 1) / RO_INVDX)
        rho, u, b = if 400.0 <= x[2] <= 600.0
            1.0, uu, bb
        elseif screen_bb !== nothing && 650.0 <= x[2] <= 700.0
            screen_rho, SVector(0.0, 0.0, 0.0), screen_bb
        else
            0.0, SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0)
        end
        rho == 0.0 && continue
        host[:, i, j, k] .= Float32.((rho, u[1], u[2], u[3], 1.0e-3 * rho / (gamma - 1),
                                      b[1], b[2], b[3]))
    end
    return host
end

function ro_gp(params, d_unit)
    cfg = RunConfig(bh_a = 0.0, frequency_hz = RO_NU,
                    camera = CameraConfig(inclination_deg = RO_INC, distance = 1000.0,
                                          fov = 10.0, resolution = 1),
                    plasma = PlasmaConfig(rat_low = 1.0, rat_high = 1.0))
    return merge(params(cfg), (d_unit = d_unit, e_unit = d_unit * RO_CL^2,
                               b_unit = RO_B / 0.01, L_unit = RO_L, nu_obs = RO_NU))
end

function ro_kernel(kern, out, host, gp)
    rays = ro_ray()
    kern(CPU())(out, rays, host, gp, 0.0, 2.0, 80.0, 1.1e3, Inf, 6000, RO_NG, RO_XO,
                RO_INVDX, 1; ndrange = 1)
    KernelAbstractions.synchronize(CPU())
    return out[1]
end
ro_gpu_I(uu, bb; d_unit) =
    ro_kernel(_fused_kernel!, Vector{Float64}(undef, 1), ro_grid(uu, bb),
              ro_gp(_gimg_params, d_unit))
ro_gpu_S(bb; d_unit = 1.0e-31, screen_bb = nothing) =
    ro_kernel(_gpol_kernel!, Vector{SVector{4,Float64}}(undef, 1),
              ro_grid(SVector(0.0, 0.0, 0.0), bb; screen_bb = screen_bb),
              ro_gp(_gpol_params, d_unit))

const RO_BVERT = SVector(0.0, 0.0, 0.01)                   # field along north
const RO_BNW = SVector(0.0, 0.01, 0.01) / sqrt(2.0)        # along north-west (+y = west)
const RO_BTOWARD = SVector(0.01, 0.0, 0.01) / sqrt(2.0)    # north, tilted toward the camera
const RO_BAWAY = SVector(-0.01, 0.0, 0.01) / sqrt(2.0)     # north, tilted away from it

for (label, fI, fS) in (("CPU march", ro_cpu_I, ro_cpu_S),
                        ("fused kernel", ro_gpu_I, ro_gpu_S))
    @testset "Doppler: gas moving toward the camera is blueshifted ($label)" begin
        # Optically thick (τ ≫ 1) slab in the Rayleigh-Jeans limit: I_obs = ν_obs³ B_ν/ν³
        # at the fluid-frame ν, so I_obs ∝ ν_obs/ν = 1/[γ(1 - v)] for velocity v toward the
        # camera (+x); the ratio to the static slab removes the r ≈ 750 redshift.
        I0 = fI(SVector(0.0, 0.0, 0.0), RO_BVERT; d_unit = 1.0e-18)
        v = 0.5; γ = 1 / sqrt(1 - v^2)
        Iapp = fI(SVector(γ * v, 0.0, 0.0), RO_BVERT; d_unit = 1.0e-18)
        Irec = fI(SVector(-γ * v, 0.0, 0.0), RO_BVERT; d_unit = 1.0e-18)
        @test I0 > 0
        @test Iapp / I0 ≈ γ * (1 + v) rtol = 2e-2               # 1.732 (bug: 0.577)
        @test Irec / I0 ≈ γ * (1 - v) rtol = 2e-2
    end

    @testset "EVPA ⟂ projected B, IAU east of north ($label)" begin
        S = fS(RO_BVERT)                          # B → north: E-vector east-west, χ = 90°
        @test S[1] > 0
        @test S[2] / S[1] < -0.5                  # Q < 0
        @test abs(S[3]) < 1e-2 * abs(S[2])
        @test abs(S[4]) < 1e-2 * S[1]             # cos θ_B = 0: no circular emission
        S = fS(RO_BNW)                            # B → north-west: E-vector NE, χ = +45°
        @test S[3] / S[1] > 0.5                   # U > 0
        @test abs(S[2]) < 1e-2 * abs(S[3])
    end

    @testset "circular polarization sign, IEEE ($label)" begin
        @test fS(RO_BTOWARD)[4] > 0               # B toward the observer: V > 0
        @test fS(RO_BAWAY)[4] < 0
    end

    @testset "Faraday rotation sense ($label)" begin
        # Emitter with B → north (χ₀ = 90°) behind a screen with B nearly along the line
        # of sight (sin θ_B = 0.02: negligible emission and conversion). RM > 0 (field
        # toward the observer) rotates χ east of 90°, i.e. 2χ past 180°: U < 0.
        bscr(s) = SVector(s * 0.01, 0.0, 2.0e-4)
        Sp = fS(RO_BVERT; d_unit = 1.0e-31, screen_bb = bscr(1.0))
        Sm = fS(RO_BVERT; d_unit = 1.0e-31, screen_bb = bscr(-1.0))
        @test Sp[2] < 0 && Sm[2] < 0              # still mostly east-west
        @test Sp[3] < -0.05 * Sp[1]               # rotated toward east
        @test Sm[3] > 0.05 * Sm[1]                # opposite field: toward west
    end
end
