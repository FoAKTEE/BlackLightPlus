# m87_calibrate.jl — the density scale of every GRMHD run of the paper's figures, from
# `m87_calibrate` (m87_scaling.jl): the Kerr accretion flow (AthenaK snapshot 00999 on Kerr
# a = 0.9, the configuration `pol_config` of fig_pol_hires.jl) and the two binaries (00455
# and 00122: the static superposed metric of the dump, the plasma at R_high = 40 and the
# camera of fig_eht_bbh.jl), each on its unpolarized 230 GHz image seen from i = 17°, over
# the full field of view of its figures (30 and 56 GM/c²), at C = 320 and BLP_RES_<run>
# pixels per side.
#
# Checks at the calibrated scale, on the same image unless stated:
#   pixels   the flux with half and twice the pixels per side, or at the pixel counts of
#            BLP_PIXELS_<run> (the binaries of M87_DENSITY_SCALE: 480,640,720,800,1120,1280,1440,1920);
#   C        the flux with C = 80 and C = 640 at half the pixels, against C = 320 there;
#   wide     a field four times wider at BLP_RES_<run> pixels: the share of its flux outside
#            the central field;
#   pol      the Stokes I flux of the polarized (IQUV) render;
#   old      (00999) the flux at 17° of the scaling before (6.2e9 Msun, density unit
#            1e-21 g cm^-3, 16.9 Mpc).
#
# Writes <BLP_OUT>/SCALES.json: per run the mass and distance, the density scale and density
# unit, the flux reached and the tolerance, the calibration image, the ladders, the checks,
# and the scale used before. Every product runs with backend = :cpu.
# Env: BLP_RUNS (00999,00455,00122), BLP_RES_00999 (768), BLP_RES_00455 (960),
# BLP_RES_00122 (960), BLP_PIXELS_<run>, BLP_CHECKS (pixels,C,wide,pol,old),
# BLP_OUT (figures/paper/out/m87_scales).
# Run from the repository root:
#   julia -t 40 --project=. figures/paper/m87_calibrate.jl

using Printf, Dates
using BlackLightPlus
using BlackLightPlus.Numerics: Units
using BlackLightPlus.Data: athenak_bin_snapshot
include(joinpath(@__DIR__, "m87_scaling.jl"))
using .M87Scaling

module BBH
include(joinpath(@__DIR__, "fig_eht_bbh.jl"))
end
module Pol
include(joinpath(@__DIR__, "fig_pol_hires.jl"))
end

_list(s) = [strip(x) for x in split(s, ",") if !isempty(strip(x))]
const RUNS = _list(get(ENV, "BLP_RUNS", "00999,00455,00122"))
const CHECKS = _list(get(ENV, "BLP_CHECKS", "pixels,C,wide,pol,old"))
const OUT = get(ENV, "BLP_OUT", joinpath(@__DIR__, "out", "m87_scales"))
const RES_DEFAULT = Dict("00999" => 768, "00455" => 960, "00122" => 960)
const GIT_SHA = BBH.GIT_SHA
lg(a...) = (println(a...); flush(stdout))

"The configuration `cfg` with step divisor `C` and the step budget scaled with it."
with_C(cfg, C) = RunConfig(cfg.m_bh_cgs, cfg.mass_scale_cgs, cfg.frequency_hz, cfg.camera, cfg.plasma,
                           round(Int, cfg.max_steps * C / cfg.C), Float64(C), cfg.escape_factor,
                           cfg.integrator, cfg.capture_redshift, cfg.horizon_eps, cfg.winding)

"""
The deck of run `run`: the loaded snapshot, a plan builder `mk(; fov, res, C, ms, mode)` at
i = 17°, the code mass unit, the field of view of the figures, the guess of the scale, the
scale used before, and a description.
"""
deck(run) = run == "00999" ? deck_kerr() : deck_binary(run)

function deck_kerr()
    snap = AthenaKSnapshot(Pol.DATA)
    st = KerrSchildCartesian(Pol.SPIN)
    mk = (; fov, res, C = M87_C, ms, mode = StokesI()) ->
        prepare(Pol.pol_config(; inc = M87_INCLINATION_DEG, fov, res, C, ms); spacetime = st,
                matter = snap, mode)
    m_old = 6.2e9 * 1.989e33
    old = Dict{String,Any}("mass_scale_g" => 1.0e-21 * Units(m_old, 1.0).L_unit^3,
                           "m_bh_msun" => 6.2e9, "msun_g" => 1.989e33, "D_Mpc" => 16.9,
                           "normalization" => "density unit 1e-21 g cm^-3",
                           "generators" => "fig_pol_hires.jl, fig_mc_dyn_tests.jl (torus), fig_dyntests_limits.jl")
    return (snap = snap, mk = mk, m_bh = M87_MASS_G, fov = Pol.FOV, guess = 2.2e23, old = old,
            file = Pol.DATA, what = "Kerr accretion flow, AthenaK 00999, Kerr a = 0.9, R_high = 10",
            extra = Dict{String,Any}("spin" => Pol.SPIN))
end

function deck_binary(run)
    tag = Dict("00455" => "s455", "00122" => "s122")[run]
    path = BBH.SNAP_PATHS[tag]
    bs = BBH.binary_setup(path)
    snap = endswith(path, ".bin") ? athenak_bin_snapshot(path; gamma = bs.gamma) :
           AthenaKSnapshot(path; gamma = bs.gamma)
    mk = (; fov, res, C = M87_C, ms, mode = StokesI()) ->
        prepare(with_C(BBH.run_config(M87_INCLINATION_DEG, fov, res, ms, BBH.RH_FID, bs.gamma), C);
                spacetime = bs.st, matter = snap, mode)
    oldms = Dict("00455" => 2.272822681e28, "00122" => 1.002619513e21)[run]
    old = Dict{String,Any}("mass_scale_g" => oldms, "m_bh_msun" => 1.3e10, "msun_g" => BBH.MSUN,
                           "D_Mpc" => 16.8, "normalization" => "1 Jy at i = 17 deg (fig_eht_bbh.jl, 960^2, C = 320)",
                           "source" => "figures/paper/out/eht_bbh/$(tag)_i17_Rh40/records.json",
                           "other_defaults" => run == "00455" ?
                               Dict("fig_raptor_r07.jl" => 2.274019497e28, "fig_mc_binary.jl" => 2.272378166e28) :
                               Dict{String,Any}())
    return (snap = snap, mk = mk, m_bh = BBH.M_BH, fov = BBH.FOV, guess = oldms * 0.5^(1 / 1.92),
            old = old, file = path,
            what = "binary GRMHD, AthenaK $run, static SuperposedKerrSchild at phi0 = $(round(bs.phi0; digits = 6)), R_high = 40",
            extra = Dict{String,Any}("gamma" => bs.gamma, "sep_M" => bs.sep, "masses" => collect(bs.masses),
                                     "a_used" => collect(bs.a_used), "m_hole_msun" => M87_MASS_MSUN))
end

flux1(plan, ms) = m87_flux_ladder(plan, [ms])[1]

function calibrate_run(run)
    t0 = time()
    res = parse(Int, get(ENV, "BLP_RES_$run", string(RES_DEFAULT[run])))
    lg("== ", run, " res ", res, " threads ", Threads.nthreads())
    d = deck(run)
    lg(@sprintf("  loaded %s in %.1f s (%s)", d.file, time() - t0, d.what))
    plan = d.mk(; fov = d.fov, res, ms = d.guess)
    cal = m87_calibrate(plan, d.guess)
    ms = cal.ms
    lg(@sprintf("  mass_scale %.10e g: F %.6f Jy (|F/%.2f - 1| = %.1e, converged %s, slope %.4f, monotonic %s) in %.1f s",
                ms, cal.flux_jy, M87_FLUX_JY, abs(cal.flux_jy / M87_FLUX_JY - 1), cal.converged,
                cal.slope, cal.monotonic, cal.wall))
    checks = Dict{String,Any}()
    if "pixels" in CHECKS
        t = time()
        rows = Any[]
        pix = haskey(ENV, "BLP_PIXELS_$run") ? parse.(Int, _list(ENV["BLP_PIXELS_$run"])) : [res ÷ 2, 2res]
        for r in pix
            F = flux1(d.mk(; fov = d.fov, res = r, ms), ms)
            push!(rows, Dict("res" => r, "flux_Jy" => F, "rel_to_calibration" => F / cal.flux_jy - 1))
            lg(@sprintf("  pixels %d: F %.6f Jy (%+.2e against %d)", r, F, F / cal.flux_jy - 1, res))
        end
        checks["pixels"] = Dict("rows" => rows, "seconds" => time() - t)
    end
    if "C" in CHECKS
        t = time()
        r = res ÷ 2
        F0 = flux1(d.mk(; fov = d.fov, res = r, ms), ms)
        rows = Any[]
        for C in (80.0, 640.0)
            F = flux1(d.mk(; fov = d.fov, res = r, C, ms), ms)
            push!(rows, Dict("C" => C, "flux_Jy" => F, "rel_to_C320" => F / F0 - 1))
            lg(@sprintf("  C %g at %d: F %.6f Jy (%+.2e against C = %g)", C, r, F, F / F0 - 1, M87_C))
        end
        checks["C"] = Dict("res" => r, "flux_C320_Jy" => F0, "rows" => rows, "seconds" => time() - t)
    end
    if "wide" in CHECKS
        t = time()
        pw = d.mk(; fov = 4 * d.fov, res, ms)
        I = variants(pw, [ms]; backend = :cpu).images[1]
        res % 8 == 0 || error("wide check: res $res is not a multiple of 8")
        c = (3res ÷ 8 + 1):(5res ÷ 8)
        sr = m87_pixel_sr(4 * d.fov, res, d.m_bh) / 1e-23
        Fw = sum(I) * sr; Fi = sum(@view I[c, c]) * sr
        checks["wide"] = Dict("fov" => 4 * d.fov, "res" => res, "flux_Jy" => Fw, "inner_flux_Jy" => Fi,
                              "frac_outside_fov" => 1 - Fi / Fw, "seconds" => time() - t)
        lg(@sprintf("  wide %.0f M at %d: F %.5f Jy, central %.0f M %.5f Jy (outside %.2f%%)",
                    4 * d.fov, res, Fw, d.fov, Fi, 100 * (1 - Fi / Fw)))
    end
    if "pol" in CHECKS
        t = time()
        im = render(d.mk(; fov = d.fov, res, ms, mode = IQUV()); backend = :cpu)
        F = m87_flux_jy(im.I, d.fov, d.m_bh)
        checks["pol"] = Dict("flux_Jy" => F, "rel_to_target" => F / M87_FLUX_JY - 1,
                             "seconds" => time() - t)
        lg(@sprintf("  polarized: F %.6f Jy (%+.2e against %.2f Jy)", F, F / M87_FLUX_JY - 1, M87_FLUX_JY))
    end
    if "old" in CHECKS && run == "00999"
        t = time()
        m_old = d.old["m_bh_msun"] * d.old["msun_g"]
        cfg = Pol.pol_config(; inc = M87_INCLINATION_DEG, fov = d.fov, res, C = M87_C,
                             ms = d.old["mass_scale_g"])
        cfg = RunConfig(m_old, cfg.mass_scale_cgs, cfg.frequency_hz, cfg.camera, cfg.plasma, cfg.max_steps,
                        cfg.C, cfg.escape_factor, cfg.integrator, cfg.capture_redshift, cfg.horizon_eps,
                        cfg.winding)
        p = prepare(cfg; spacetime = KerrSchildCartesian(Pol.SPIN), matter = d.snap, mode = StokesI())
        I = variants(p, [d.old["mass_scale_g"]]; backend = :cpu).images[1]
        Dold = d.old["D_Mpc"] * 1e6 * 3.0856775814913673e18
        F = sum(I) * (d.fov / res * Units(m_old, 1.0).L_unit / Dold)^2 / 1e-23
        d.old["flux_i17_Jy"] = F
        lg(@sprintf("  scaling before: F %.5f Jy at 17 deg", F))
    end
    u = Units(d.m_bh, ms)
    return Dict{String,Any}(
        "run" => run, "snapshot" => d.file, "what" => d.what, "deck" => d.extra,
        "M_msun" => d.m_bh / M87_MSUN_G, "M_g" => d.m_bh, "msun_g" => M87_MSUN_G,
        "D_Mpc" => M87_DISTANCE_MPC, "D_cm" => M87_DISTANCE_CM, "frequency_hz" => M87_FREQUENCY_HZ,
        "mass_scale_g" => ms, "density_unit_g_cm3" => u.dens_unit, "Ne_unit_cm3" => u.Ne_unit,
        "B_unit_G" => u.B_unit,
        "target_Jy" => M87_FLUX_JY, "flux_Jy" => cal.flux_jy, "rel_error" => cal.flux_jy / M87_FLUX_JY - 1,
        "tolerance_rel" => M87_FLUX_RTOL, "converged" => cal.converged,
        "slope_dlnF_dlnms" => cal.slope, "monotonic" => cal.monotonic,
        "image" => Dict("mode" => "StokesI (unpolarized)", "inclination_deg" => M87_INCLINATION_DEG,
                        "fov_M" => d.fov, "res" => res, "C" => M87_C,
                        "max_steps" => plan.config.max_steps, "camera_distance_M" => plan.config.camera.distance,
                        "observer_azimuth" => "simulation +x axis (phi_o = 0)",
                        "flux" => "sum of the image over the field of view"),
        "ladders" => cal.ladders, "calibration_wall_s" => cal.wall, "checks" => checks,
        "old" => d.old, "wall_s" => time() - t0)
end

function main()
    mkpath(OUT)
    path = joinpath(OUT, "SCALES.json")
    runs = Dict{String,Any}()
    for run in RUNS
        runs[run] = calibrate_run(run)
        rec = Dict{String,Any}("runs" => runs, "git_sha" => GIT_SHA, "when" => string(Dates.now()),
                               "threads" => Threads.nthreads(), "generator" => "figures/paper/m87_calibrate.jl",
                               "scaling" => Dict("M_hole_msun" => M87_MASS_MSUN, "msun_g" => M87_MSUN_G,
                                                 "D_Mpc" => M87_DISTANCE_MPC, "frequency_hz" => M87_FREQUENCY_HZ,
                                                 "target_Jy" => M87_FLUX_JY, "inclination_deg" => M87_INCLINATION_DEG,
                                                 "C" => M87_C, "tolerance_rel" => M87_FLUX_RTOL))
        BBH.writejson(path, rec)
        GC.gc()
    end
    lg("wrote ", path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
