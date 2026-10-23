# Ray orientation and the camera Stokes convention through the products, on optically
# thin and thick slabs in near-flat space (a = 0, r ≈ 750): a camera ray is the physical
# photon traced backward, so gas moving toward the camera is blueshifted; IAU/IEEE
# polarization with north = +β, east = -α and EVPA χ = ½ atan2(U, Q) east of north:
# synchrotron E-vectors are perpendicular to the sky-projected field, thermal electrons
# emit V > 0 for a field toward the observer, and a Faraday screen whose field points
# toward the observer rotates χ toward east.
using Test, BlackLightPlus, StaticArrays
using BlackLightPlus.Plans: CameraConfig, PlasmaConfig, RunConfig, prepare
using BlackLightPlus.Products: render
using BlackLightPlus.Spacetime: KerrSchildCartesian
using BlackLightPlus.Matter: GridProvider, PlasmaPrimitives
using BlackLightPlus.Transfer: StokesI, IQUV
using BlackLightPlus.Numerics: CL, GNEWT, MSUN
import BlackLightPlus.Matter: primitives, domain_radius

const M_HOLE = 6.5e9 * MSUN
const L_HOLE = GNEWT * M_HOLE / CL^2
const GAMMA = 13.0 / 9.0
const Z3 = SVector(0.0, 0.0, 0.0)

"""
Camera at i = 90° (on +x, north = +z, west = +y = +α) and one ray at (α, β) = (500, 500),
the line y = z = 500 toward -x: a slab fills 400 < x < 600 around it (density 1,
velocity `uu`, field `bb`), an optional screen 650 < x < 700 (between slab and camera).
`p/ρ = 10⁻³` and `rat_low = rat_high = 1` give Θₑ ≈ 0.9 independent of β.
"""
struct Slab <: GridProvider
    uu::SVector{3,Float64}
    bb::SVector{3,Float64}
    screen_bb::SVector{3,Float64}
    screen::Bool
end
inyz(x) = 400.0 < x[3] < 600.0 && 400.0 < x[4] < 600.0
function primitives(s::Slab, x::SVector{4})
    eint = 1.0e-3 / (GAMMA - 1)
    inyz(x) && 400.0 < x[2] < 600.0 && return PlasmaPrimitives{Float64}(1.0, s.uu, eint, s.bb, true)
    s.screen && inyz(x) && 650.0 < x[2] < 700.0 &&
        return PlasmaPrimitives{Float64}(200.0, Z3, 200.0 * eint, s.screen_bb, true)
    return PlasmaPrimitives{Float64}()
end
domain_radius(::Slab) = Inf

# The Stokes vector of the ray through the slab, with the field `bb` in units of 10⁴ G at
# the density unit `d_unit` (g cm⁻³).
function slab_stokes(mode, uu, bb; d_unit = 1.0e-31, screen_bb = nothing)
    scale = 1.0e4 / (CL * sqrt(4π * d_unit))
    cfg = RunConfig(m_bh_cgs = M_HOLE, mass_scale_cgs = d_unit * L_HOLE^3, frequency_hz = 1.0e9,
                    camera = CameraConfig(inclination_deg = 90.0, distance = 1000.0, fov = 10.0,
                                          resolution = 1),
                    plasma = PlasmaConfig(rat_low = 1.0, rat_high = 1.0, gamma = GAMMA, sigma_cut = -1.0))
    slab = Slab(uu, scale * bb, screen_bb === nothing ? Z3 : scale * screen_bb, screen_bb !== nothing)
    p = prepare(cfg; spacetime = KerrSchildCartesian(0.0), matter = slab, mode)
    return render(p, [(500.0, 500.0)]; backend = :cpu)[:, 1]
end

const BVERT = SVector(0.0, 0.0, 0.01)                   # field along north
const BNW = SVector(0.0, 0.01, 0.01) / sqrt(2.0)        # along north-west (+y = west)
const BTOWARD = SVector(0.01, 0.0, 0.01) / sqrt(2.0)    # north, tilted toward the camera
const BAWAY = SVector(-0.01, 0.0, 0.01) / sqrt(2.0)     # north, tilted away from it

@testset "ray orientation and the Stokes convention" begin
    @testset "Doppler: gas moving toward the camera is blueshifted ($(nameof(typeof(mode))))" for
            mode in (StokesI(), IQUV())
        # optically thick Rayleigh-Jeans slab: I ∝ ν_obs/ν = 1/[γ(1 - v)] toward the camera
        I0 = slab_stokes(mode, Z3, BVERT; d_unit = 1.0e-18)[1]
        v = 0.5; γ = 1 / sqrt(1 - v^2)
        Iapp = slab_stokes(mode, SVector(γ * v, 0.0, 0.0), BVERT; d_unit = 1.0e-18)[1]
        Irec = slab_stokes(mode, SVector(-γ * v, 0.0, 0.0), BVERT; d_unit = 1.0e-18)[1]
        @test I0 > 0
        @test Iapp / I0 ≈ γ * (1 + v) rtol = 2e-2
        @test Irec / I0 ≈ γ * (1 - v) rtol = 2e-2
    end

    @testset "EVPA ⟂ projected B, IAU east of north" begin
        S = slab_stokes(IQUV(), Z3, BVERT)          # B north: E-vector east-west, χ = 90°
        @test S[1] > 0 && S[2] / S[1] < -0.5
        @test abs(S[3]) < 1e-2 * abs(S[2]) && abs(S[4]) < 1e-2 * S[1]
        S = slab_stokes(IQUV(), Z3, BNW)            # B north-west: E-vector north-east, χ = 45°
        @test S[3] / S[1] > 0.5 && abs(S[2]) < 1e-2 * abs(S[3])
    end

    @testset "circular polarization sign, IEEE" begin
        @test slab_stokes(IQUV(), Z3, BTOWARD)[4] > 0
        @test slab_stokes(IQUV(), Z3, BAWAY)[4] < 0
    end

    @testset "Faraday rotation sense" begin
        # emitter with B north behind a screen with B nearly along the line of sight: a
        # field toward the observer (RM > 0) rotates χ east of 90°, so U < 0
        bscr(s) = SVector(s * 0.01, 0.0, 2.0e-4)
        Sp = slab_stokes(IQUV(), Z3, BVERT; screen_bb = bscr(1.0))
        Sm = slab_stokes(IQUV(), Z3, BVERT; screen_bb = bscr(-1.0))
        @test Sp[2] < 0 && Sm[2] < 0
        @test Sp[3] < -0.05 * Sp[1]
        @test Sm[3] > 0.05 * Sm[1]
    end
end
