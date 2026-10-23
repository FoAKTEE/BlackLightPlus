# fig_grmhd.jl — Mahakala (arXiv:2304.03804) figures 2 and 8–11 on the AthenaK single-hole
# torus series (spin 0.9) with BlackLightPlus. The paper's imaging parameters are kept: hole mass
# 6.5e9 Msun, 230 GHz, plane-parallel camera at 1000 M and inclination 17°, field of view
# 64 M, the electron temperature from the internal energy (γ_i = 5/3, γ_e = 4/3,
# R_low = 1), σ ≤ 1. The torus's code density is not the paper's, so the mass scale
# (ρ_unit = 𝓜/L³) is set from the flux: 𝓜 = MREF gives 0.55 Jy at 230 GHz for M87's
# distance (16.8 Mpc) at R_high = 10, which stands in for the paper's 𝓜 = 5e25 (factor 250);
# the paper's ladder 5e23, 5e24, 1e25 maps to MREF/100, MREF/10, MREF/5.
#   fig9    the 230 GHz image of snapshot 00999 at R_high = 10, 40, 160 and 𝓜 = MREF (512²,
#           pixel 1/8 M), and the same image with the emission restricted to one region at a
#           time (vacuum elsewhere, so no absorption outside the region): the forward jet
#           (θ < π/4), the disk (π/4 ≤ θ ≤ 3π/4), the counter-jet (θ > 3π/4), and the shells
#           bounded by the prograde and retrograde ISCO and photon-orbit radii of a = 0.9;
#           with the capture map of the camera (fig 10's shadow);
#   fig8    the same deck at 256² (R_high = 10, 𝓜 = MREF) traced by BlackLightPlus and by Blacklight
#           (ref-code/c-white-blacklight, the binary BLACKLIGHT; the dump rewritten in the
#           Athena++ layout its reader parses by athdf_for_blacklight.py, samples outside
#           the mesh vacuum) on the same snapshot, for the NMSE of the paper's equation 25;
#   fig2    81 trajectories of the central ±16 M of a 48² camera at the fig 9 deck with the
#           invariant emissivity j_ν/ν² sampled along them;
#   fig11   per-shell profiles (shells of 0.5 M from r_h to 40 M) averaged over the 10 dumps
#           00909 … 00999 (300 M apart): the mean invariant intensity change
#           ΔI_ν/ν³ = Σ_rays Δ𝓘 / N_pix accumulated inside the shell along the marched rays
#           (absorption of light from deeper shells included) and the affine-path-weighted
#           mean invariant emissivity ⟨j_ν/ν²⟩ = Σ j ds / Σ ds over the ray samples in the
#           shell, for R_high = 10, 20, 40, 80 at 𝓜 = MREF/10 and i = 17°; 𝓜 = MREF/100,
#           MREF/10, MREF/5 at R_high = 10 and i = 17°; i = 5°, 17°, 60°, 85°, 120°, 175° at
#           R_high = 80 and 𝓜 = MREF/10. Rays: the fixed-control march of `trace` (C = 80)
#           at 192².
# Writes figures/paper/out/mahakala/grmhd/: fig9_R<R>_<region>.npy, fig9_shadow.npy,
# fig8_blp.npy and fig8_blacklight.npz (+ the deck), fig2_rays.npy (rows ray, x, y, z,
# j/ν²), fig11_<tag>.npy (rows r_mid, ΔI, ⟨j⟩), setup_<parts>.json (one per run).
# Run from the repository root:  julia -t 32 --project=. figures/mahakala/fig_grmhd.jl
# Env: BLP_MAHAKALA_DATA (the dump directory), BLACKLIGHT (the binary), BLP_MAHAKALA_PARTS
# (the figures to make, default "9,8,2,11").
using Dates, LinearAlgebra, Printf, StaticArrays
using BlackLightPlus
using BlackLightPlus.Spacetime: KerrSchildCartesian, ks_radius, metric_and_inverse, isco_radius,
                     photon_orbit_radius, horizon_radius
using BlackLightPlus.Matter: GridProvider, PlasmaPrimitives, decode
import BlackLightPlus.Matter: primitives, domain_radius
using BlackLightPlus.Transfer: StokesI, coefficients, segment
using BlackLightPlus.Data: AthenaKSnapshot, write_npy
using BlackLightPlus.Plans: transfer_params
using BlackLightPlus.Numerics: Units

const DATA = get(ENV, "BLP_MAHAKALA_DATA", "/data/haiyangw/ebh/build-no/src/bondi-100-output-r3/bin/no")
const OUT = joinpath(dirname(@__DIR__), "out", "mahakala", "grmhd")
const ROOT = dirname(dirname(@__DIR__))
const BLACKLIGHT = get(ENV, "BLACKLIGHT", joinpath(ROOT, "ref-code", "c-white-blacklight", "bin", "blacklight"))
const GIT_SHA = try
    sha = readchomp(`git -C $ROOT rev-parse HEAD`)
    isempty(readchomp(`git -C $ROOT status --porcelain -- src`)) ? sha : sha * "-dirty"
catch
    "unknown"
end
mkpath(OUT)
logf(fmt::String, args...) = (print(Printf.format(Printf.Format(fmt), args...)); flush(stdout))

const SPIN = 0.9
const ST = KerrSchildCartesian(SPIN)
const MSUN = 1.989e33
const MBH = 6.5e9 * MSUN
const NU = 2.3e11
const INC = 17.0
const FOV = 64.0
const DCAM = 1000.0
const SNAP9 = joinpath(DATA, "torus.mhd_w_bcc.00999.athdf")
const SNAPS11 = [joinpath(DATA, @sprintf("torus.mhd_w_bcc.%05d.athdf", n)) for n in 909:10:999]
const RHIGH9 = (10.0, 40.0, 160.0)
const MREF = 2.0e23
const M9 = MREF
const RES9 = 512
const RES8 = 256
const RES11 = 192
const RMS_P = isco_radius(SPIN; prograde = true); const RMS_M = isco_radius(SPIN; prograde = false)
const RPH_P = photon_orbit_radius(SPIN; prograde = true); const RPH_M = photon_orbit_radius(SPIN; prograde = false)
const RH = horizon_radius(ST)

config(res, inc, rhigh, mscale) = RunConfig(
    m_bh_cgs = MBH, mass_scale_cgs = mscale, frequency_hz = NU,
    camera = CameraConfig(inclination_deg = inc, distance = DCAM, fov = FOV, resolution = res),
    plasma = PlasmaConfig(rat_low = 1.0, rat_high = rhigh, gamma = 13.0 / 9.0, gamma_i = 5.0 / 3.0,
                          gamma_e = 4.0 / 3.0, use_p = false, sigma_cut = 1.0, theta_e_min = -1.0),
    max_steps = 4000, C = 80.0)

# a provider that is the snapshot where keep(x) and vacuum elsewhere
struct Masked{P,F} <: GridProvider
    snap::P
    keep::F
end
@inline function primitives(m::Masked, x::SVector{4})
    prim = primitives(m.snap, x)
    return m.keep(x) ? prim : PlasmaPrimitives{typeof(prim.rho)}()
end
domain_radius(m::Masked) = domain_radius(m.snap)

pos(y) = SVector(y[1], y[2], y[3], y[4])
costheta(x) = (r = ks_radius(ST, x); r > 0 ? x[4] / r : 0.0)
const C45 = cos(π / 4)
const REGIONS = (
    "jet" => x -> costheta(x) > C45,
    "disk" => x -> abs(costheta(x)) <= C45,
    "counterjet" => x -> costheta(x) < -C45,
    "ms_in" => x -> ks_radius(ST, x) < RMS_P,
    "ms_mid" => x -> RMS_P <= ks_radius(ST, x) <= RMS_M,
    "ms_out" => x -> ks_radius(ST, x) > RMS_M,
    "ph_in" => x -> ks_radius(ST, x) < RPH_P,
    "ph_mid" => x -> RPH_P <= ks_radius(ST, x) <= RPH_M,
    "ph_out" => x -> ks_radius(ST, x) > RPH_M)

json_str(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
json_str(x::Bool) = x ? "true" : "false"
json_str(x::Integer) = string(x)
json_str(x::Real) = isfinite(x) ? string(Float64(x)) : "null"
json_str(x::AbstractVector) = "[" * join(json_str.(x), ",") * "]"
json_str(d::AbstractDict) = "{" * join(("\"$k\": " * json_str(v) for (k, v) in sort(collect(d); by = first)), ",\n ") * "}"
setup = Dict{String,Any}("code" => GIT_SHA, "threads" => Threads.nthreads(), "date" => string(now()),
    "spin" => SPIN, "m_bh_msun" => 6.5e9, "frequency_hz" => NU, "inclination_deg" => INC, "fov" => FOV,
    "camera_distance" => DCAM, "r_ms" => [RMS_P, RMS_M], "r_ph" => [RPH_P, RPH_M], "r_h" => RH,
    "snapshot_fig9" => SNAP9, "snapshots_fig11" => SNAPS11, "rhigh_fig9" => collect(RHIGH9),
    "mass_scale_fig9" => M9, "res_fig9" => RES9, "res_fig8" => RES8, "res_fig11" => RES11,
    "distance_mpc" => 16.8)
const PARTS = split(get(ENV, "BLP_MAHAKALA_PARTS", "9,8,2,11"), ",")
save_setup() = open(io -> println(io, json_str(setup)), joinpath(OUT, "setup_" * join(PARTS, "_") * ".json"), "w")

t_all = time()
logf("loading %s\n", SNAP9)
snap = AthenaKSnapshot(SNAP9)
logf("  time %.1f, domain radius %.1f\n", snap.time, domain_radius(snap))

# ------------------------------------------------------------------- figure 9 (and 10)
"9" in PARTS && for rh in RHIGH9
    cfg = config(RES9, INC, rh, M9)
    t0 = time()
    img = render(prepare(cfg; spacetime = ST, matter = snap, mode = StokesI()); backend = :cpu)
    write_npy(joinpath(OUT, @sprintf("fig9_R%d_full.npy", rh)), img.I)
    logf("fig 9 R_high = %3d full: %.1f s, max I %.3e, sum %.3e\n", rh, time() - t0, maximum(img.I), sum(img.I))
    setup[@sprintf("fig9_R%d_full_sum", rh)] = sum(img.I)
    for (name, keep) in REGIONS
        t0 = time()
        im = render(prepare(cfg; spacetime = ST, matter = Masked(snap, keep), mode = StokesI()); backend = :cpu)
        write_npy(joinpath(OUT, @sprintf("fig9_R%d_%s.npy", rh, name)), im.I)
        setup[@sprintf("fig9_R%d_%s_sum", rh, name)] = sum(im.I)
        logf("           %-10s: %.1f s, sum %.3e (%.1f%% of full)\n", name, time() - t0, sum(im.I),
             100 * sum(im.I) / sum(img.I))
    end
    save_setup()
end
"9" in PARTS && let
    sh = shadow(prepare(config(RES9, INC, 10.0, M9); spacetime = ST, matter = nothing))
    write_npy(joinpath(OUT, "fig9_shadow.npy"), Float64.(sh.fate))
    logf("fig 9 shadow: %d captured of %d\n", count(==(1), sh.fate), length(sh.fate))
end

# ------------------------------------------------------------------- figure 8
"8" in PARTS && let
    cfg = config(RES8, INC, 10.0, M9)
    t0 = time()
    img = render(prepare(cfg; spacetime = ST, matter = snap, mode = StokesI()); backend = :cpu)
    write_npy(joinpath(OUT, "fig8_blp.npy"), img.I)
    logf("fig 8 BlackLightPlus: %.1f s, sum %.4e\n", time() - t0, sum(img.I))
    units = Units(MBH, M9)
    # Blacklight's athena reader wants the Athena++ layout: prim/B datasets, press = (γ-1) eint,
    # float32 arrays, fixed-type attributes
    conv = joinpath(OUT, "fig8_snapshot_athena_layout.athdf")
    run(`python3 $(joinpath(@__DIR__, "athdf_for_blacklight.py")) $SNAP9 $conv $(13.0 / 9.0)`)
    deck = """
model_type  = simulation
num_threads = 32
output_format = npz
output_file   = $(joinpath(OUT, "fig8_blacklight.npz"))
output_camera = false
checkpoint_geodesic_save = false
checkpoint_geodesic_load = false
checkpoint_sample_save   = false
checkpoint_sample_load   = false
simulation_format       = athena
simulation_file         = $conv
simulation_multiple     = false
simulation_coord        = cks
simulation_a            = $SPIN
simulation_m_msun       = 6.5e9
simulation_rho_cgs      = $(units.dens_unit)
simulation_interp       = true
simulation_block_interp = false
camera_type       = plane
camera_r          = $DCAM
camera_th         = $INC
camera_ph         = 0.0
camera_urn        = 0.0
camera_uthn       = 0.0
camera_uphn       = 0.0
camera_k_r        = 1.0
camera_k_th       = 0.0
camera_k_ph       = 0.0
camera_rotation   = 0.0
camera_width      = $FOV
camera_resolution = $RES8
ray_flat        = false
ray_terminate   = multiplicative
ray_factor      = 1.005
ray_integrator  = dp
ray_step        = 0.01
ray_max_steps   = 4000
ray_max_retries = 20
ray_tol_abs     = 1.0e-8
ray_tol_rel     = 1.0e-8
image_light           = true
image_num_frequencies = 1
image_frequency       = $NU
image_normalization   = infinity
image_polarization    = false
image_rotation_split  = false
image_time            = false
image_length          = false
image_lambda          = false
image_emission        = false
image_tau             = false
image_lambda_ave      = false
image_emission_ave    = false
image_tau_int         = false
image_crossings       = false
render_num_images = 0
slow_light_on = false
adaptive_max_level = 0
plasma_mu         = 0.5
plasma_ne_ni      = 1.0
plasma_model      = ti_te_beta
plasma_use_p      = false
plasma_gamma      = 1.4444444444444444
plasma_gamma_i    = 1.6666666666666667
plasma_gamma_e    = 1.3333333333333333
plasma_rat_low    = 1.0
plasma_rat_high   = 10.0
plasma_power_frac = 0.0
plasma_kappa_frac = 0.0
cut_rho_min          = -1.0
cut_rho_max          = -1.0
cut_n_e_min          = -1.0
cut_n_e_max          = -1.0
cut_p_gas_min        = -1.0
cut_p_gas_max        = -1.0
cut_theta_e_min      = -1.0
cut_theta_e_max      = -1.0
cut_b_min            = -1.0
cut_b_max            = -1.0
cut_sigma_min        = -1.0
cut_sigma_max        = 1.0
cut_beta_inverse_min = -1.0
cut_beta_inverse_max = -1.0
cut_omit_near        = false
cut_omit_far         = false
cut_omit_in          = -1.0
cut_omit_out         = -1.0
cut_midplane_theta   = 0.0
cut_midplane_z       = 0.0
cut_plane            = false
fallback_nan   = false
fallback_rho   = 0.0
fallback_pgas  = 0.0
fallback_kappa = 0.0
"""
    open(io -> write(io, deck), joinpath(OUT, "fig8_blacklight.input"), "w")
    if isfile(BLACKLIGHT)
        t0 = time()
        ok = success(pipeline(`$BLACKLIGHT $(joinpath(OUT, "fig8_blacklight.input"))`;
                              stdout = joinpath(OUT, "fig8_blacklight.log"), stderr = joinpath(OUT, "fig8_blacklight.log")))
        logf("fig 8 Blacklight: %s in %.1f s\n", ok ? "done" : "FAILED", time() - t0)
        setup["fig8_blacklight_ok"] = ok
    else
        logf("fig 8: no Blacklight binary at %s\n", BLACKLIGHT)
        setup["fig8_blacklight_ok"] = false
    end
    setup["fig8_dens_unit"] = units.dens_unit
    save_setup()
end

# ------------------------------------------------------------------- figure 2
"2" in PARTS && let
    cfg = config(48, INC, 10.0, M9)
    plan = prepare(cfg; spacetime = ST, matter = snap, mode = StokesI())
    P = transfer_params(cfg)
    t0 = time()
    tr = trace(plan; backend = :cpu)
    # the 9 × 9 rays of the central ±16 M of the image (pixels 12, 15, …, 36 on each axis)
    sel = [k for k in 1:length(tr.states) if ((k - 1) % 48) in 12:3:36 && ((k - 1) ÷ 48) in 12:3:36]
    rows = Vector{NTuple{5,Float64}}()
    for (ir, k) in enumerate(sel)
        S = tr.states[k]
        for j in 1:length(S)-1
            x = pos(S[j]); p = SVector(S[j][5], S[j][6], S[j][7], S[j][8])
            prim = primitives(snap, x)
            jv = 0.0
            if prim.rho > 0
                g = metric_and_inverse(ST, x)
                st = decode(prim, g, P.units, P.electrons, P.cuts)
                jv, _ = coefficients(StokesI(), P, st, -p, P.nu_obs)
            end
            push!(rows, (Float64(ir), x[2], x[3], x[4], Float64(jv)))
        end
    end
    write_npy(joinpath(OUT, "fig2_rays.npy"), reduce(vcat, [collect(t)' for t in rows]))
    logf("fig 2: %d rays, %d samples in %.1f s\n", length(sel), length(rows), time() - t0)
    setup["fig2_nrays"] = length(sel)
end

# ------------------------------------------------------------------- figure 11
const SHELL_W = 0.5
const SHELL_EDGES = collect(RH:SHELL_W:40.0)
const NSHELL = length(SHELL_EDGES) - 1
const M11 = MREF / 10
const COMBOS17 = [(10.0, M11), (20.0, M11), (40.0, M11), (80.0, M11), (10.0, MREF / 100), (10.0, MREF / 5)]
const INCS = (5.0, 17.0, 60.0, 85.0, 120.0, 175.0)
const COMBO_I = (80.0, M11)
tag(inc, rh, ms) = @sprintf("i%03d_R%03d_M%.2e", round(Int, inc), round(Int, rh), ms)
setup["mref"] = MREF
setup["fig11_ladders"] = [
    Dict("title" => "R_high ladder", "items" => [Dict("tag" => tag(INC, rh, M11), "label" => @sprintf("R_high = %d", rh)) for rh in (10, 20, 40, 80)]),
    Dict("title" => "mass-scale ladder", "items" => [Dict("tag" => tag(INC, 10.0, ms), "label" => lab) for (ms, lab) in ((MREF / 100, "M_ref/100"), (MREF / 10, "M_ref/10"), (MREF / 5, "M_ref/5"))]),
    Dict("title" => "inclination ladder", "items" => [Dict("tag" => tag(inc, COMBO_I[1], COMBO_I[2]), "label" => @sprintf("i = %d deg", inc)) for inc in INCS])]

"per-shell ΔI and Σ j ds, Σ ds of every combo for the trajectories `tr` through `snap`"
function profiles(tr, snapi, params)
    nc = length(params)
    nt = Threads.maxthreadid()
    dI = zeros(nt, nc, NSHELL); js = zeros(nt, nc, NSHELL); ds = zeros(nt, NSHELL)
    Threads.@threads for k in 1:length(tr.states)
        tid = Threads.threadid()
        S = tr.states[k]; D = tr.dls[k]
        I = zeros(nc)
        for j in 1:length(S)-1
            x = pos(S[j]); p = SVector(S[j][5], S[j][6], S[j][7], S[j][8]); dl = D[j]
            prim = primitives(snapi, x)
            prim.rho > 0 || continue
            r = ks_radius(ST, x)
            ib = floor(Int, (r - RH) / SHELL_W) + 1
            g = metric_and_inverse(ST, x)
            inshell = 1 <= ib <= NSHELL
            inshell && (ds[tid, ib] += dl)
            for c in 1:nc
                P = params[c]
                st = decode(prim, g, P.units, P.electrons, P.cuts)
                jc, ac = coefficients(StokesI(), P, st, -p, P.nu_obs)
                T, E = segment(StokesI(), jc, ac, dl)
                Inew = T * I[c] + E
                if inshell
                    dI[tid, c, ib] += Inew - I[c]
                    js[tid, c, ib] += jc * dl
                end
                I[c] = Inew
            end
        end
    end
    return dropdims(sum(dI; dims = 1); dims = 1), dropdims(sum(js; dims = 1); dims = 1), vec(sum(ds; dims = 1))
end

"11" in PARTS && let
    acc = Dict{String,Any}()        # tag => (dI sum over snapshots, js, ds)
    for inc in INCS
        combos = inc == INC ? COMBOS17 : [COMBO_I]
        cfg = config(RES11, inc, combos[1][1], combos[1][2])
        plan = prepare(cfg; spacetime = ST, matter = snap, mode = StokesI())
        t0 = time()
        tr = trace(plan; backend = :cpu)
        logf("fig 11 i = %3.0f: traced %d rays (%d states) in %.1f s\n", inc, length(tr.states),
             sum(length, tr.states), time() - t0)
        params = [transfer_params(config(RES11, inc, rh, ms)) for (rh, ms) in combos]
        for f in SNAPS11
            t0 = time()
            snapi = f == SNAP9 ? snap : AthenaKSnapshot(f)
            dI, js, ds = profiles(tr, snapi, params)
            for (c, (rh, ms)) in enumerate(combos)
                key = tag(inc, rh, ms)
                a = get!(acc, key, (zeros(NSHELL), zeros(NSHELL), zeros(NSHELL), Ref(0)))
                a[1] .+= dI[c, :]; a[2] .+= js[c, :]; a[3] .+= ds; a[4][] += 1
            end
            logf("           %s: %.1f s\n", basename(f), time() - t0)
        end
        npix = length(tr.states)
        for (c, (rh, ms)) in enumerate(combos)
            key = tag(inc, rh, ms)
            a = acc[key]
            n = a[4][]
            M = hcat(SHELL_EDGES[1:end-1] .+ SHELL_W / 2, a[1] ./ (n * npix), a[2] ./ max.(a[3], 1e-300))
            write_npy(joinpath(OUT, "fig11_$(key).npy"), M)
        end
        setup["fig11_npix"] = npix
        save_setup()
    end
end
setup["wall_s"] = time() - t_all
save_setup()
logf("wrote %s (%.0f s)\n", OUT, time() - t_all)
