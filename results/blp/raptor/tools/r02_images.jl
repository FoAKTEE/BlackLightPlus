# r02_images.jl — R02 test 3 (dag.md R02): image-level comparison of RAPTOR (fork,
# single-pixel libraries) and BlackLightPlus on analytic scenes, same camera (M4),
# metric (M1, one hole), march (M3, C = 160) and coefficients (M7).
#   gold : Gold et al. 2020 analytic emitter, tests T1-T5 (Table 1), unpolarized
#          (RAPTOR unpolarized transfer; BlackLightPlus: its exact segment operator
#          `unpolarized_segment` along trace_rays_sampled states, the reverse march of
#          reverse_unpolarized_march with the Gold coefficients)
#   ring : the orientation-check ring (thermal, polarized), for |P|, V and EVPA
# Camera: d = 1000, FOV 30 M, i = 17/60/90 deg, NPIX^2 (default 480).
# Metrics: total flux, NMSE(I) = sum (I_R - I_B)^2 / sum I_B^2, normalized L1 of I (and
# |P|, V for the ring), net EVPA difference.
#
# Usage: julia --project=. -t N r02_images.jl <bindir> <out.json> [npix] [C] [legs]
# Exit 0 iff every image meets NMSE(I) <= 1e-4, flux within 0.1%, L1(I) <= 1%, and for
# the ring L1(|P|), L1(V) <= 3% and EVPA within 0.5 deg.

using BlackLightPlus, StaticArrays, LinearAlgebra, Printf
using BlackLightPlus.Geodesics: SuperposedKerrSchild, PlaneParallelCamera, camera_rays,
                                trace_rays_sampled, ks_radius_hole, horizon_radius_hole
using BlackLightPlus.Inputs: RunConfig, PlasmaConfig
const RAD = BlackLightPlus.Radiation
const SIM = BlackLightPlus.Simulation
include(joinpath(@__DIR__, "jsonw.jl"))

const BINDIR = ARGS[1]
const OUT = ARGS[2]
const NPIX = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 480
const CVAL = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 160.0
const LEGS = length(ARGS) >= 5 ? split(ARGS[5], ",") : ["gold", "ring"]
const INCS = (17.0, 60.0, 90.0)
const FOV = 30.0
const DIST = 1000.0
const NMAX = 8000
const CL = 2.99792458e10
const G_R = 6.6743e-8
const MSUN_R = 1.98847e33
const KPC = 3.0856775814913673e21
const NU = 230.0e9
const TMP = mktempdir()

# Gold et al. 2020 Table 1: (A, alpha, h, l0, a)
const GOLD = Dict("T1" => (0.0, -3.0, 0.0, 0.0, 0.9), "T2" => (0.0, -2.0, 0.0, 1.0, 0.0),
                  "T3" => (0.0, 0.0, 10 / 3, 1.0, 0.9), "T4" => (1.0e5, 0.0, 10 / 3, 1.0, 0.9),
                  "T5" => (1.0e6, 0.0, 100 / 3, 1.0, 0.9))
const GOLD_L = 6.0e11            # GM/c^2 [cm]
const GOLD_D = 2.4e22            # distance [cm]
const CN0 = 3.0e-18
const BETA = 2.5

# ---------------------------------------------------------------- RAPTOR libraries
struct PixLib
    setup::Ptr{Cvoid}
    pixel::Ptr{Cvoid}
end
function PixLib(path)
    h = Libc.Libdl.dlopen(path)
    PixLib(Libc.Libdl.dlsym(h, :raptor_rt_setup), Libc.Libdl.dlsym(h, :raptor_rt_pixel))
end
const LPOL = PixLib(joinpath(BINDIR, "libraptor_rt_pol.so"))
const LUNP = PixLib(joinpath(BINDIR, "libraptor_rt_unpol.so"))

function write_file(name, lines)
    p = joinpath(TMP, name)
    open(io -> foreach(l -> println(io, l), lines), p, "w")
    return p
end
function raptor_image(L::PixLib, coords; mbh_msun, m_unit, dist_kpc, inc, spin, scene)
    mi = write_file("model.in", [
        "MBH (Msun) $(repr(mbh_msun))", "DISTANCE (kpc) $(repr(dist_kpc))",
        "M_UNIT (g) $(repr(m_unit))", "R_LOW (-) 1", "R_HIGH (-) 1",
        "INCLINATION (deg) $(repr(inc))", "IMG_WIDTH (pixels) 10", "IMG_HEIGHT (pixels) 10",
        "CAM_SIZE_X (Rg) $(repr(FOV))", "CAM_SIZE_Y (Rg) $(repr(FOV))", "FREQS_PER_DEC (-) 1",
        "FREQ_MIN (Hz) $(repr(NU))", "STEPSIZE (-) 0.01", "MAX_LEVEL (-) 1"])
    ENV["RAPTOR_BBH"] = write_file("bbh.in", [
        "nholes 1", "mass 1.0", "spin $(repr(spin))", "center1 0.0 0.0 0.0",
        "C $(repr(CVAL))", "nmax $NMAX", "escape $(repr(1.1 * DIST))", "horizon_eps 0.001",
        "capture_redshift 100.0", "cam_dist $(repr(DIST))"])
    ENV["RAPTOR_SCENE"] = write_file("scene.in", scene)
    redirect_stdout(devnull) do
        ccall(L.setup, Cvoid, (Cstring,), mi)
    end
    out = zeros(4, length(coords))
    Threads.@threads :dynamic for i in eachindex(coords)
        s = zeros(4)
        ccall(L.pixel, Cint, (Cdouble, Cdouble, Ptr{Float64}), coords[i][1], coords[i][2], s)
        out[:, i] .= s
    end
    return out
end

# ---------------------------------------------------------------- BlackLightPlus
"Gold fluid at x: (n/n0, u^mu) or nothing (the C scene of model/blp_analytic, mirrored)."
function gold_fluid(st, x, h, l0)
    a = st.spin[1]
    r = ks_radius_hole(st, x, 1)
    r > horizon_radius_hole(st, 1) || return nothing
    cth = x[4] / r; th = acos(cth); R = r * sin(th); z = h * cth
    n = exp(-0.5 * ((r / 10.0)^2 + z^2))
    l = l0 / (1.0 + R) * R^1.5
    s = sin(th); c = cos(th)
    sig = r^2 + a^2 * c^2; del = r^2 - 2r + a^2
    A_ = (r^2 + a^2)^2 - a^2 * del * s^2
    gtt = -A_ / (sig * del); gtp = -2a * r / (sig * del); gpp = (del - a^2 * s^2) / (sig * del * s^2)
    den = gtt - 2gtp * l + gpp * l^2
    den < 0.0 || return nothing
    ub = sqrt(-1.0 / den)
    ut = gtt * (-ub) + gtp * (ub * l); uph = gtp * (-ub) + gpp * (ub * l)
    return n, SVector(ut, -x[3] * uph, x[2] * uph, 0.0)
end

function gold_ray(st, ys, dls, T, L)
    A, α, h, l0, _ = T
    Iinv = 0.0; trans = 1.0
    m = length(ys)
    for k in m:-1:1
        ds = (k > 1 ? dls[k-1] : 0.0) * L / NU
        y = ys[k]; x = SVector(y[1], y[2], y[3], y[4])
        f = gold_fluid(st, x, h, l0)
        f === nothing && continue
        n, u = f
        kcov = -SVector(y[5], y[6], y[7], y[8])
        ν = -dot(kcov, u) * NU
        r = ν / NU
        j = CN0 * n * r^(-α) / ν^2
        a = A * CN0 * n * r^(-(BETA + α)) * ν
        Aseg, Bseg = RAD.unpolarized_segment(j, a, ds)
        Iinv += trans * Bseg; trans *= Aseg
    end
    return Iinv * NU^3
end

"Trace in chunks (bounded memory) and apply `f(states_i, dls_i)` per ray."
function blp_map(st, rays, f; chunk = 16384)
    out = Vector{Any}(undef, length(rays))
    for c0 in 1:chunk:length(rays)
        idx = c0:min(c0 + chunk - 1, length(rays))
        states, dls = trace_rays_sampled(st, rays[idx]; C = CVAL, nmax = NMAX, escape = 1.1 * DIST)
        Threads.@threads :dynamic for q in eachindex(idx)
            out[idx[q]] = f(states[q], dls[q])
        end
    end
    return out
end

# ring (orientation check) plasma
const M_BH = 2 * 6.5e9 * MSUN_R
const MS = 1.0e24
ring_samp = x -> begin
    R = hypot(x[2], x[3]); z3 = zero(SVector{3,Float64})
    if 6.0 < R < 12.0 && abs(x[4]) < 1.0
        v = 1 / sqrt(R); W = 1 / sqrt(1 - v^2)
        (rho = 1.0, pgas = 0.1, uu = SVector(-W * v * x[3] / R, W * v * x[2] / R, 0.0),
         bb = SVector(0.0, 0.0, 0.3), inside = true)
    else
        (rho = 0.0, pgas = 0.0, uu = z3, bb = z3, inside = false)
    end
end

# ---------------------------------------------------------------- metrics
function img_metrics(R, B, pixsr)
    IR = R[1, :]; IB = B[1, :]
    m = Dict{String,Any}(
        "flux_raptor" => sum(IR) * pixsr / 1e-23, "flux_blp" => sum(IB) * pixsr / 1e-23,
        "flux_rel_diff" => sum(IR) / sum(IB) - 1,
        "nmse_I" => sum(abs2, IR .- IB) / sum(abs2, IB),
        "l1_I" => sum(abs, IR .- IB) / sum(IB),
        "max_pixel_rel_diff_I" => maximum(abs.(IR .- IB)) / maximum(IB))
    if any(!iszero, B[2:4, :])
        PR = hypot.(R[2, :], R[3, :]); PB = hypot.(B[2, :], B[3, :])
        m["l1_P"] = sum(abs, PR .- PB) / sum(PB)
        m["l1_V"] = sum(abs, R[4, :] .- B[4, :]) / sum(abs, B[4, :])
        m["nmse_P"] = sum(abs2, PR .- PB) / sum(abs2, PB)
        evpa(M) = 0.5 * atand(sum(M[3, :]), sum(M[2, :]))
        d = evpa(R) - evpa(B); d = mod(d + 90, 180) - 90
        m["evpa_net_raptor_deg"] = evpa(R); m["evpa_net_blp_deg"] = evpa(B); m["evpa_diff_deg"] = d
        m["m_net_raptor"] = hypot(sum(R[2, :]), sum(R[3, :])) / sum(IR)
        m["m_net_blp"] = hypot(sum(B[2, :]), sum(B[3, :])) / sum(IB)
        m["v_net_raptor"] = sum(R[4, :]) / sum(IR); m["v_net_blp"] = sum(B[4, :]) / sum(IB)
    end
    return m
end
function passes(m)
    ok = m["nmse_I"] <= 1e-4 && abs(m["flux_rel_diff"]) <= 1e-3 && m["l1_I"] <= 1e-2
    if haskey(m, "l1_P")
        ok &= m["l1_P"] <= 0.03 && m["l1_V"] <= 0.03 && abs(m["evpa_diff_deg"]) <= 0.5
    end
    return ok
end

# ---------------------------------------------------------------- main
res = Dict{String,Any}("npix" => NPIX, "C" => CVAL, "fov" => FOV, "dist" => DIST, "nmax" => NMAX)
allpass = true
e = range(-FOV / 2, FOV / 2; length = NPIX + 1); c = collect((e[1:end-1] .+ e[2:end]) ./ 2)
coords = [(a, b) for a in c for b in c]
for inc in INCS
    cam = PlaneParallelCamera(inc_deg = inc, d = DIST, fov = FOV, npix = NPIX)
    if "gold" in LEGS
        for spin in (0.9, 0.0)
            st = SuperposedKerrSchild([1.0], [spin], [(0.0, 0.0, 0.0)])
            rays = camera_rays(st, cam)
            tests = sort([k for (k, T) in GOLD if T[5] == spin])
            t0 = time()
            Bimgs = Dict(k => zeros(4, length(rays)) for k in tests)
            vals = blp_map(st, rays, (ys, dl) -> [gold_ray(st, ys, dl, GOLD[k], GOLD_L) for k in tests])
            for i in eachindex(vals), q in eachindex(tests)
                Bimgs[tests[q]][1, i] = vals[i][q]
            end
            tb = time() - t0
            for k in tests
                A, α, h, l0, _ = GOLD[k]
                t1 = time()
                R = raptor_image(LUNP, coords; mbh_msun = GOLD_L * CL^2 / (G_R * MSUN_R),
                                 m_unit = 1.0, dist_kpc = GOLD_D / KPC, inc = inc, spin = spin,
                                 scene = ["scene gold", "gold_A $(repr(A))", "gold_alpha $(repr(α))",
                                          "gold_h $(repr(h))", "gold_l0 $(repr(l0))",
                                          "gold_beta $(repr(BETA))", "gold_Cn0 $(repr(CN0))",
                                          "gold_nup $(repr(NU))"])
                pixsr = (FOV / NPIX * GOLD_L / GOLD_D)^2
                m = img_metrics(R, Bimgs[k], pixsr)
                m["pass"] = passes(m); m["wall_raptor_s"] = time() - t1; m["wall_blp_trace_s"] = tb
                global allpass &= m["pass"]
                res["gold_$(k)_i$(Int(inc))"] = m
                @printf("gold %s i=%2d  F_R %.5f F_B %.5f Jy  dF %+.2e  NMSE %.2e  L1 %.2e  %s\n", k, inc,
                        m["flux_raptor"], m["flux_blp"], m["flux_rel_diff"], m["nmse_I"], m["l1_I"],
                        m["pass"] ? "PASS" : "FAIL")
            end
        end
    end
    if "ring" in LEGS
        st = SuperposedKerrSchild([1.0], [0.9], [(0.0, 0.0, 0.0)])
        rays = camera_rays(st, cam)
        cfg = RunConfig(bh_a = 0.0, m_bh_cgs = M_BH, mass_scale_cgs = MS, frequency_hz = NU,
                        polarized = true,
                        plasma = PlasmaConfig(gamma = 13 / 9, rat_low = 1.0, rat_high = 1.0,
                                              mu = 0.5, sigma_cut = 1.0))
        scc = BlackLightPlus._prepared_scc(cfg, SIM.units(M_BH, MS))
        Lr = SIM.units(M_BH, 1.0).L_unit
        up = BlackLightPlus._camera_up(cam)
        v = blp_map(st, rays, (ys, dl) -> RAD.polarized_march(st, scc, ring_samp, ys, dl, NU, Lr; cam_up = up))
        B = reduce(hcat, [Vector(x) for x in v])
        R = raptor_image(LPOL, coords; mbh_msun = 2 * 6.5e9, m_unit = MS, dist_kpc = 16800.0,
                         inc = inc, spin = 0.9,
                         scene = ["scene ring", "gamma $(repr(13 / 9))", "rat_low 1.0",
                                  "rat_high 1.0", "sigma_cut 1.0"])
        pixsr = (FOV / NPIX * Lr / (16.8e6 * 3.0856775814913673e18))^2
        m = img_metrics(R, B, pixsr); m["pass"] = passes(m)
        global allpass &= m["pass"]
        res["ring_i$(Int(inc))"] = m
        @printf("ring    i=%2d  dF %+.2e  NMSE %.2e  L1 I %.2e P %.2e V %.2e  EVPA %+.3f deg (m %.4f/%.4f, v %+.5f/%+.5f)  %s\n",
                inc, m["flux_rel_diff"], m["nmse_I"], m["l1_I"], m["l1_P"], m["l1_V"], m["evpa_diff_deg"],
                m["m_net_raptor"], m["m_net_blp"], m["v_net_raptor"], m["v_net_blp"], m["pass"] ? "PASS" : "FAIL")
    end
end
res["pass"] = allpass
res["thresholds"] = Dict("nmse_I" => 1e-4, "flux_rel_diff" => 1e-3, "l1_I" => 1e-2, "l1_P" => 0.03,
                         "l1_V" => 0.03, "evpa_diff_deg" => 0.5)
mkpath(dirname(OUT))
write_json(OUT, res)
println(allpass ? "R02 images PASS" : "R02 images FAIL")
exit(allpass ? 0 : 1)
