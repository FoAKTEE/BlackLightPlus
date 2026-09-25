# r02_orientation.jl — R02 tests 1 and 2 (dag.md R02): Stokes orientation of RAPTOR
# (fork libraptor_rt_pol.so, scenes of model/blp_analytic) against BlackLightPlus.
#
# Test 1, thin slabs (test/test_ray_orientation.jl, a = 0, camera i = 90 deg on +x, one ray
#   at image (alpha, beta) = (500, 500)): B north -> Q < 0; B north-west -> U > 0; B toward
#   the observer -> V > 0 (away -> V < 0); a Faraday screen with RM > 0 rotates chi east
#   (U < 0), RM < 0 west (U > 0).
# Test 2, the figure driver's orientation check (fig_eht_bbh.jl orientation_check: Kerr
#   a = 0.9, prograde ring, vertical field, i = 60, FOV 30, 96^2, M = 1.3e10 M_sun,
#   mass scale 1e24 g, 230 GHz): east/west I ratio, Q/I, U/I and net EVPA.
#
# Usage: julia --project=. -t N r02_orientation.jl <libraptor_rt_pol.so> <out.json>
# Exit 0 iff every test-1 sign agrees between the codes and with the expectation, and
# test 2 agrees within 1% on the ratios and 0.5 deg on the EVPA.

using BlackLightPlus, StaticArrays, LinearAlgebra, Printf
using BlackLightPlus.Geodesics: KerrSchildCartesian, SuperposedKerrSchild, PlaneParallelCamera,
                                camera_rays, camera_rays_at, trace_rays_sampled
using BlackLightPlus.Inputs: RunConfig, PlasmaConfig
const RAD = BlackLightPlus.Radiation
const SIM = BlackLightPlus.Simulation
include(joinpath(@__DIR__, "jsonw.jl"))

const LIB = ARGS[1]
const OUT = ARGS[2]
const CL = 2.99792458e10
const G_R = 6.6743e-8          # model/blp_analytic constants (M9)
const MSUN_R = 1.98847e33
const TMP = mktempdir()

h = Libc.Libdl.dlopen(LIB)
const f_setup = Libc.Libdl.dlsym(h, :raptor_rt_setup)
const f_pixel = Libc.Libdl.dlsym(h, :raptor_rt_pixel)

function write_file(name, lines)
    p = joinpath(TMP, name)
    open(io -> foreach(l -> println(io, l), lines), p, "w")
    return p
end
v3(v) = join([repr(x) for x in v], " ")

"Configure the RAPTOR library: model.in, bbh.in, scene file."
function raptor_setup(; mbh_msun, m_unit, inc, freq, fov, npix, bbh, scene)
    mi = write_file("model.in", [
        "MBH (Msun) $(repr(mbh_msun))", "DISTANCE (kpc) 16800", "M_UNIT (g) $(repr(m_unit))",
        "R_LOW (-) 1", "R_HIGH (-) 1", "INCLINATION (deg) $(repr(inc))",
        "IMG_WIDTH (pixels) $npix", "IMG_HEIGHT (pixels) $npix",
        "CAM_SIZE_X (Rg) $(repr(fov))", "CAM_SIZE_Y (Rg) $(repr(fov))",
        "FREQS_PER_DEC (-) 1", "FREQ_MIN (Hz) $(repr(freq))", "STEPSIZE (-) 0.01",
        "MAX_LEVEL (-) 1"])
    ENV["RAPTOR_BBH"] = write_file("bbh.in", bbh)
    ENV["RAPTOR_SCENE"] = write_file("scene.in", scene)
    ccall(f_setup, Cvoid, (Cstring,), mi)
end
function raptor_pixels(coords)
    out = Vector{SVector{4,Float64}}(undef, length(coords))
    Threads.@threads :dynamic for i in eachindex(coords)
        s = zeros(4)
        ccall(f_pixel, Cint, (Cdouble, Cdouble, Ptr{Float64}), coords[i][1], coords[i][2], s)
        out[i] = SVector{4}(s)
    end
    return out
end

bbh1(; spin, C = 80.0) = ["nholes 1", "mass 1.0", "spin $(repr(spin))", "center1 0.0 0.0 0.0",
                          "C $(repr(C))", "nmax 4000", "escape 1100.0", "horizon_eps 0.001",
                          "capture_redshift 100.0", "cam_dist 1000.0"]

# ---------------------------------------------------------------- test 1: slabs
const RO_L = 6.674e-8 * 6.5e9 * 1.989e33 / CL^2
const RO_NU = 1.0e9
const RO_B = 100.0
const BVERT = SVector(0.0, 0.0, 0.01)
const BNW = SVector(0.0, 0.01, 0.01) / sqrt(2.0)
const BTOWARD = SVector(0.01, 0.0, 0.01) / sqrt(2.0)
const BAWAY = SVector(-0.01, 0.0, 0.01) / sqrt(2.0)
bscr(s) = SVector(s * 0.01, 0.0, 2.0e-4)

ro_st() = KerrSchildCartesian(0.0)
ro_cfg(d_unit) = SimCoeffConfig(d_unit = d_unit, e_unit = d_unit * CL^2, b_unit = RO_B / 0.01,
                                plasma_rat_low = 1.0, plasma_rat_high = 1.0)
ro_inyz(x) = 400.0 < x[3] < 600.0 && 400.0 < x[4] < 600.0
function ro_sampler(bb; screen_bb = nothing, screen_rho = 200.0)
    z3 = SVector(0.0, 0.0, 0.0)
    return x -> begin
        if ro_inyz(x) && 400.0 < x[2] < 600.0
            (rho = 1.0, pgas = 1.0e-3, uu = z3, bb = bb, inside = true)
        elseif screen_bb !== nothing && ro_inyz(x) && 650.0 < x[2] < 700.0
            (rho = screen_rho, pgas = 1.0e-3 * screen_rho, uu = z3, bb = screen_bb, inside = true)
        else
            (rho = 0.0, pgas = 0.0, uu = z3, bb = z3, inside = false)
        end
    end
end
function blp_slab(bb; d_unit = 1.0e-31, screen_bb = nothing)
    cam = PlaneParallelCamera(inc_deg = 90.0, d = 1000.0, fov = 10.0, npix = 1)
    ray = camera_rays_at(ro_st(), cam, [(500.0, 500.0)])
    states, dls = trace_rays_sampled(ro_st(), ray; C = 80.0, escape = 1.1e3)
    return RAD.polarized_march(ro_st(), ro_cfg(d_unit), ro_sampler(bb; screen_bb = screen_bb),
                               states[1], dls[1], RO_NU, RO_L;
                               cam_up = SVector(0.0, -cosd(90.0), 0.0, sind(90.0)))
end
function raptor_slab(bb; d_unit = 1.0e-31, screen_bb = nothing)
    raptor_setup(mbh_msun = RO_L * CL^2 / (G_R * MSUN_R), m_unit = 1.0, inc = 90.0,
                 freq = RO_NU, fov = 10.0, npix = 10, bbh = bbh1(spin = 0.0),
                 scene = ["scene slab", "d_unit $(repr(d_unit))", "b_unit $(repr(RO_B / 0.01))",
                          "gamma $(repr(5 / 3))", "rat_low 1.0", "rat_high 1.0", "sigma_cut -1",
                          "slab_uu 0.0 0.0 0.0", "slab_bb $(v3(bb))",
                          "screen $(screen_bb === nothing ? 0 : 1)",
                          "screen_bb $(v3(screen_bb === nothing ? zeros(3) : screen_bb))",
                          "screen_rho 200.0"])
    return raptor_pixels([(500.0, 500.0)])[1]
end

evpa(S) = 0.5 * atand(S[3], S[2])
function slab_case(name, expect, bb; kw...)
    B = blp_slab(bb; kw...); R = raptor_slab(bb; kw...)
    e_b = expect(B); e_r = expect(R)
    same = all(sign(R[k]) == sign(B[k]) || (abs(R[k]) < 1e-2 * R[1] && abs(B[k]) < 1e-2 * B[1]) for k in 2:4)
    @printf("  %-22s BLP  I %.4e Q/I %+.4f U/I %+.4f V/I %+.4f | RAPTOR I %.4e Q/I %+.4f U/I %+.4f V/I %+.4f | expect B %s R %s\n",
            name, B[1], B[2] / B[1], B[3] / B[1], B[4] / B[1], R[1], R[2] / R[1], R[3] / R[1],
            R[4] / R[1], e_b, e_r)
    return Dict("blp" => collect(B), "raptor" => collect(R), "blp_expected" => e_b,
                "raptor_expected" => e_r, "signs_agree" => same,
                "I_ratio" => R[1] / B[1], "evpa_blp_deg" => evpa(B), "evpa_raptor_deg" => evpa(R),
                "pass" => e_b && e_r && same)
end

println("test 1: thin slabs (a = 0, i = 90, ray at (500, 500), 1 GHz)")
t1 = Dict{String,Any}()
t1["B_north_Q_negative"] = slab_case("B north: Q<0", S -> S[1] > 0 && S[2] / S[1] < -0.5 &&
                                     abs(S[3]) < 1e-2 * abs(S[2]) && abs(S[4]) < 1e-2 * S[1], BVERT)
t1["B_northwest_U_positive"] = slab_case("B north-west: U>0", S -> S[3] / S[1] > 0.5 &&
                                         abs(S[2]) < 1e-2 * abs(S[3]), BNW)
t1["B_toward_V_positive"] = slab_case("B toward: V>0", S -> S[4] > 0, BTOWARD)
t1["B_away_V_negative"] = slab_case("B away: V<0", S -> S[4] < 0, BAWAY)
t1["faraday_RM_positive_east"] = slab_case("screen RM>0: chi east", S -> S[2] < 0 && S[3] < -0.05 * S[1],
                                           BVERT; screen_bb = bscr(1.0))
t1["faraday_RM_negative_west"] = slab_case("screen RM<0: chi west", S -> S[2] < 0 && S[3] > 0.05 * S[1],
                                           BVERT; screen_bb = bscr(-1.0))
pass1 = all(v["pass"] for v in values(t1))

# ---------------------------------------------------------------- test 2: ring
const MSUN_D = 1.98847e33
const M_BH = 2 * 6.5e9 * MSUN_D
const NU = 230.0e9
const MS = 1.0e24
const INC, FOV, RES = 60.0, 30.0, 96
println("test 2: orientation check (Kerr a = 0.9, prograde ring, i = 60, 96^2)")
st1 = SuperposedKerrSchild([1.0], [0.9], [(0.0, 0.0, 0.0)])
z3 = zero(SVector{3,Float64})
samp = x -> begin
    R = hypot(x[2], x[3])
    if 6.0 < R < 12.0 && abs(x[4]) < 1.0
        v = 1 / sqrt(R); W = 1 / sqrt(1 - v^2)
        (rho = 1.0, pgas = 0.1, uu = SVector(-W * v * x[3] / R, W * v * x[2] / R, 0.0),
         bb = SVector(0.0, 0.0, 0.3), inside = true)
    else
        (rho = 0.0, pgas = 0.0, uu = z3, bb = z3, inside = false)
    end
end
cam = PlaneParallelCamera(INC, 1000.0, FOV, RES)
rays = camera_rays(st1, cam)
states, dls = trace_rays_sampled(st1, rays; C = 80.0, nmax = 4000, escape = 1100.0)
cfg = RunConfig(bh_a = 0.0, m_bh_cgs = M_BH, mass_scale_cgs = MS, frequency_hz = NU,
                polarized = true,
                plasma = PlasmaConfig(gamma = 13 / 9, rat_low = 1.0, rat_high = 1.0, mu = 0.5,
                                      sigma_cut = 1.0), max_steps = 4000)
scc = BlackLightPlus._prepared_scc(cfg, SIM.units(M_BH, MS))
L_UNIT = SIM.units(M_BH, 1.0).L_unit
up = BlackLightPlus._camera_up(cam)
SB = Vector{SVector{4,Float64}}(undef, length(rays))
Threads.@threads :dynamic for i in eachindex(rays)
    SB[i] = RAD.polarized_march(st1, scc, samp, states[i], dls[i], NU, L_UNIT; cam_up = up)
end
e = range(-FOV / 2, FOV / 2; length = RES + 1); c = collect((e[1:end-1] .+ e[2:end]) ./ 2)
coords = [(a, b) for a in c for b in c]
raptor_setup(mbh_msun = 2 * 6.5e9, m_unit = MS, inc = INC, freq = NU, fov = FOV, npix = 100,
             bbh = bbh1(spin = 0.9),
             scene = ["scene ring", "gamma $(repr(13 / 9))", "rat_low 1.0", "rat_high 1.0",
                      "sigma_cut 1.0"])
SR = raptor_pixels(coords)
function ring_stats(S)
    img = [S[(ia-1)*RES+ib][k] for k in 1:4, ib in 1:RES, ia in 1:RES]   # [s, iβ, iα]
    I = img[1, :, :]; hh = RES ÷ 2
    east = sum(I[:, 1:hh]); west = sum(I[:, hh+1:end])
    Q = sum(img[2, :, :]); U = sum(img[3, :, :]); V = sum(img[4, :, :])
    return Dict("east_over_west_I" => east / west, "Q_over_I" => Q / sum(I),
                "U_over_I" => U / sum(I), "V_over_I" => V / sum(I),
                "net_evpa_deg" => 0.5 * atand(U, Q), "I_sum" => sum(I))
end
rb = ring_stats(SB); rr = ring_stats(SR)
rel(k) = abs(rr[k] / rb[k] - 1)
t2 = Dict("blp" => rb, "raptor" => rr,
          "rel_diff" => Dict(k => rel(k) for k in ("east_over_west_I", "Q_over_I", "U_over_I", "I_sum")),
          "evpa_diff_deg" => abs(rr["net_evpa_deg"] - rb["net_evpa_deg"]),
          "blp_recorded_index_json" => Dict("east_over_west_I" => 2.543172193, "Q_over_I" => -0.65036495,
                                            "U_over_I" => 0.2401844574, "net_evpa_deg" => 79.86522448))
signs2 = sign(rr["Q_over_I"]) == sign(rb["Q_over_I"]) && sign(rr["U_over_I"]) == sign(rb["U_over_I"]) &&
         (rr["east_over_west_I"] > 1) == (rb["east_over_west_I"] > 1)
pass2 = signs2 && rel("east_over_west_I") <= 0.01 && rel("Q_over_I") <= 0.01 &&
        rel("U_over_I") <= 0.01 && t2["evpa_diff_deg"] <= 0.5
t2["signs_agree"] = signs2
t2["pass"] = pass2
for (nm, r) in (("BLP", rb), ("RAPTOR", rr))
    @printf("  %-6s east/west %.4f  Q/I %+.4f  U/I %+.4f  V/I %+.5f  EVPA %.2f deg  sum I %.5e\n", nm,
            r["east_over_west_I"], r["Q_over_I"], r["U_over_I"], r["V_over_I"], r["net_evpa_deg"], r["I_sum"])
end
@printf("  rel diffs: e/w %.2e Q/I %.2e U/I %.2e I %.2e; EVPA %.3f deg -> %s\n", rel("east_over_west_I"),
        rel("Q_over_I"), rel("U_over_I"), rel("I_sum"), t2["evpa_diff_deg"], pass2 ? "PASS" : "FAIL")

res = Dict("test1_slabs" => t1, "test1_pass" => pass1, "test2_orientation_check" => t2,
           "test2_pass" => pass2, "pass" => pass1 && pass2)
mkpath(dirname(OUT))
write_json(OUT, res)
println(pass1 && pass2 ? "R02 orientation PASS" : "R02 orientation FAIL")
exit(pass1 && pass2 ? 0 : 1)
