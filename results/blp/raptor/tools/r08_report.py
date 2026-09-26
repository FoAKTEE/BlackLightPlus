"""r08_report.py — render results/blp/raptor/report.md from the R07 and R08 evidence
(evidence/R07/metrics.json, evidence/R07/polstep.json, evidence/R07/native_cold.txt,
evidence/R08/timing.json, patches/SERIES). Every number in the report is read from
those files. Usage: python3 r08_report.py [out.md]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAP = os.path.dirname(HERE)
EV = os.path.join(RAP, "evidence")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAP, "report.md")

met = json.load(open(os.path.join(EV, "R07", "metrics.json")))
pst = json.load(open(os.path.join(EV, "R07", "polstep.json")))
tim = json.load(open(os.path.join(EV, "R08", "timing.json")))
cold = [x for x in open(os.path.join(EV, "R07", "native_cold.txt")).read().strip().splitlines() if not x.startswith("#")]
series = open(os.path.join(RAP, "patches", "SERIES")).read().splitlines()
fork_commit = dict(x.split() for x in open(os.path.join(EV, "R07", "fork_commit")).read().strip().splitlines())
TH = met["thresholds"]
INCS = sorted(met["blp"], key=int)


def e(x, n=2):
    return f"{x:.{n}e}"


def f(x, n=4):
    return f"{x:+.{n}f}"


def flag(ok):
    return "pass" if ok else "**miss**"


L = []
w = L.append
w("# RAPTOR and BlackLightPlus on the binary-black-hole snapshots")
w("")
w("Nodes R07 (s455 image comparison), R08 (CPU timings), R09 (spinning snapshots) and R10 (i = 90° AMR-figure deck) of `dag.md`; design and attribution "
  "rule in `design.md` (§3 conventions, §4 modifications, §5 attribution). Every number below "
  "is read from `evidence/R07/metrics.json`, `evidence/R07/polstep.json`, "
  "`evidence/R07/native_cold.txt` and `evidence/R08/timing.json` by "
  "`tools/r08_report.py`; the R09 and R10 sections (spinning snapshots and the i = 90° AMR-figure deck) "
  "are read from `evidence/R09` and `evidence/R10` by `tools/r09_report.py`.")
w("")
w("## Setup")
w("")
b17 = met["blp"]["17"]["observables"]
w("| item | value (both codes) |")
w("|---|---|")
w("| snapshot | s455, `SANE_stage2_horizon_zero_spin/torus.mhd_w_bcc.00455.bin` (AthenaK `.bin`, 512 blocks of 64³) |")
w("| binary | static superposed Kerr–Schild, m_A = 0.5, a_A = 0, sep 25, centres ±(12.19120680, −2.76124551, 0) from the dump time |")
w("| scaling | M = 1.3e10 M_sun (6.5e9 per hole), D = 16.8 Mpc, 230 GHz; θ_g = " + f"{met['uas_per_M']:.4f}" + " µas/M |")
w("| plasma | thermal, R_low 1, R_high 40, σ > 1 cut, γ = 13/9 from the dump, n_e = ρ/m_p, mass scale 2.274019497e28 g (the calibrated value of `figures/out/eht_bbh/s455_i17_Rh40`, used by both codes) |")
w("| camera | plane-parallel at d = 1000 M, φ_o = 0, FOV 56 M, " + f"{met['npix']}²" + "; i = 17°, 60°, 90° |")
w("| march | RK4, h = min_A(r_A − r_h,A)/C, nmax = 50 C, capture d < 1e−3 or E_n > 100, escape R > 1100 |")
w("| transfer | polarized along the whole ray; BlackLightPlus coefficients (M21, `expm1` Kirchhoff, j_P ≤ j_I and α_P ≤ α_I rescales) |")
w(f"| codes | BlackLightPlus `{met.get('blp_git_sha', '')[:7]}` CPU path (`tools/r07_blp.jl`); RAPTOR fork `blp-bbh` (`tools/r07_build.sh`, `tools/r07_raptor.sh`), legs a/b built at `{fork_commit['a,b'][:7]}`, legs c/stock at `{fork_commit['c,stock'][:7]}` (the two differ only in `rho_Q_thermal`, which legs a/b do not call) |")
w("")
w("RAPTOR legs:")
w("")
w("- **(a) matched**: M1–M9 and M12 with RAPTOR's own polarized step (`check_stiffness` → RK4 or implicit trapezoid, f^μ transport between samples).")
w("- **(b) matched solver**: (a) plus M11, the exact constant-coefficient step of BlackLightPlus `stokes_segment` (`POL_SOLVER_LD85`).")
w("- **(c) native fits** (report only): (a) with the upstream emission fits (`EMIS_BLP 0`).")
w("- **stock** (report only): the upstream choices that can run on this data: upstream fits, upstream constants (G 6.674e−8, M_sun 1.989e33, kpc 3.086e21, m_e, m_p, e, h), the upstream plasma decode of `model/bhac` (n_e = ρ/(m_p + m_e) + 1e−6, b² floor, Θ_e = (u/ρ)(1/3)(m_p/m_e)/(1 + R), Θ_e ∈ [1e−3, 100], r < 1 and r > 50 cuts) and the polarized radius 50 M, on the matched geometry, camera, march and IEEE/IAU readout. Upstream RAPTOR has no binary metric, no AthenaK reader and a single-hole step control whose inner cutoff is the horizon of a hole at the origin, so the geometry legs cannot run unmodified; the upstream Bardeen camera at r = 1e4 differs from the plane-parallel camera by a sub-pixel projection and is not used.")
w("")
w("## Modifications to RAPTOR")
w("")
w("Branch `blp-bbh` from upstream `b91e937`; the patch series is `patches/` (`fork.md`).")
w("")
w("| commit | title |")
w("|---|---|")
for ln in series:
    p = ln.split(" ", 1)
    if len(p) == 2 and len(p[0]) == 7 and "(" in p[1]:
        w(f"| `{p[0]}` | {p[1]} |")
w("")
w("The R07/R08 modifications:")
w("")
w(f"- **M11** `pol_rte_ld85_step`: port of `stokes_segment` + `stokes_admissible`; "
  f"{pst['ncase']} random segments (δτ 1e−4–30, Faraday depth 1e−3–300 per segment) agree with "
  f"BlackLightPlus `stokes_couple` to {e(pst['ld85_max_rel_diff'], 1)} (relative).")
w(f"- **M12** one `parallel for schedule(dynamic, 64)` over every pixel, `RAPTOR_TIMERS` phase timers, "
  "per-pixel end class and stored-state count in the HDF5 image; the R01 synthetic image (switches off) "
  "stays bitwise identical to upstream `b91e937`.")
w("- **native-coefficient fix** `rho_Q_thermal`: K₁/K₂ → 1 when K₂(1/Θ_e) underflows (the guard "
  "`rho_V_thermal` already applies to K₀/K₂). " + cold[0].replace("native_cold: ", "Before: ") + "; after: "
  + cold[-1].replace("native_cold: ", "") + ".")
w("- **upstream plasma switch** `ATHK_UPSTREAM_PLASMA` for the stock leg.")
w("")
w("## Native polarized step")
w("")
tf = pst["native_trapezoid_fraction"]
order = pst["native_convergence"]["observed_order"]
w(f"`check_stiffness` evaluates the RK4 amplification 1 + τ + τ²/2 + τ³/6 + τ⁴/24 with τ = +λ dl "
  f"on `dl_current` without the `C_CONST` factor of the step, and selects the implicit trapezoid when "
  f"any |amplification| > 0.99. For any absorbing segment the amplification exceeds 1, so the "
  f"trapezoid is taken on {100 * tf:.1f}% of the test segments; the step converges at order "
  f"{order[-1]:.2f} (`polstep.json`). Its phase error per segment grows with the Faraday depth per "
  f"segment:")
w("")
w("| Faraday depth per segment | median relative error | max |")
w("|---|---|---|")
for k, v in sorted(met_items := pst["native_rel_err_by_faraday_depth_per_step"].items(), key=lambda kv: float(kv[0].split("-")[0])):
    w(f"| {k} | {e(v['median'])} | {e(v['max'])} |")
w("")
w("## Results")
w("")
w("Thresholds (`dag.md` R07): |ΔF/F| ≤ " + f"{TH['flux_rel']:g}" + " (≤ " + f"{TH['flux_rel_richardson']:g}" +
  " after Richardson); L1(I) ≤ " + f"{TH['l1_I']:g}" + "; NMSE(I) ≤ " + f"{TH['nmse_I']:g}" + "; |Δm_net|, |Δ⟨|m|⟩| ≤ " +
  f"{TH['m_net']:g}" + "; |ΔEVPA_net| ≤ " + f"{TH['evpa_net_deg']:g}" + "°; |Δv_net| ≤ " + f"{TH['v_net']:g}" +
  "; ring diameters within " + f"{TH['ring_diameter_uas']:g}" + " µas (native and 20 µas blur); |Δf_C| ≤ " +
  f"{TH['f_C']:g}" + "; i = 90°: the pair unresolved in both codes. m_net, ⟨|m|⟩, v_net and EVPA_net are "
  "the 20 µas-blurred EHT definitions of `figures/plot_eht_bbh.py`; L1 and NMSE are on the native 480² "
  "images. Differences are RAPTOR − BlackLightPlus.")
w("")
for leg, name in (("ra", "Leg (a), native polarized step"), ("rb", "Leg (b), LD85 step (M11)")):
    w(f"### {name}")
    w("")
    w("| i | C | ΔF/F | L1(I) | NMSE(I) | L1(\\|P\\|) | L1(V) | Δm_net | Δ⟨\\|m\\|⟩ | ΔEVPA_net [°] | Δv_net | Δ ring d [µas] max | Δf_C max | class agree | gate |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for inc in INCS:
        E = met["legs"][leg][inc]
        for C in E["C_levels"]:
            m = E["levels"][str(C)]
            dd = max(abs(m[f"d_hole{k}_{lab}_diameter_uas"]) for k in (1, 2) for lab in ("native", "blur20"))
            dfc = max(abs(m[f"d_hole{k}_{lab}_f_C"]) for k in (1, 2) for lab in ("native", "blur20"))
            fails = [k for k, v in m["gate"].items() if not v]
            att = m["attribution"]
            g = "pass" if not fails else "miss " + ", ".join(
                f"{k} ({att[k]['class']})" if k in att else k for k in fails)
            w(f"| {inc} | {C} | {e(m['flux_rel'])} | {e(m['l1_I'])} | {e(m['nmse_I'])} | {e(m['l1_P'])} | "
              f"{e(m['l1_V'])} | {f(m['d_m_net'])} | {f(m['d_mabs_avg'])} | {m['d_evpa_net_deg']:+.2f} | "
              f"{f(m['d_v_net'], 5)} | {dd:.2f} | {dfc:.3f} | {m['class_agreement']:.6f} | {g} |")
    w("")
    w("Flux after Richardson extrapolation of each code's C ladder (three finest levels):")
    w("")
    w("| i | RAPTOR limit [Jy] | BlackLightPlus limit [Jy] | relative difference | gate |")
    w("|---|---|---|---|---|")
    for inc in INCS:
        E = met["legs"][leg][inc]
        rr, rb_ = E["flux_richardson_raptor"], E["flux_richardson_blp"]
        w(f"| {inc} | {rr['limit']:.6f}{' (order ' + format(rr['order'], '.2f') + ')' if rr.get('order') else ' (non-monotone: finest level)'} | "
          f"{rb_['limit']:.6f}{' (order ' + format(rb_['order'], '.2f') + ')' if rb_.get('order') else ' (non-monotone: finest level)'} | "
          f"{e(E['flux_rel_richardson'])} | {flag(E['flux_richardson_pass'])} |")
    w("")
    w("Evaluated levels (C = 160 and the finest level):")
    w("")
    for inc in INCS:
        for C, s in met["legs"][leg][inc]["evaluated"].items():
            w(f"- i = {inc}°, C = {C}: " + ("all thresholds met" if s["pass"] else
              "misses " + ", ".join(s["failed"]) + ("; model-attributed: " + ", ".join(s["model_attributed"])
                                                    if s["model_attributed"] else "; every miss attributed to discretisation")))
    w("")

w("### Attribution (design.md §5)")
w("")
ra17 = met["legs"]["ra"]["17"]["levels"]
Cs17 = met["legs"]["ra"]["17"]["C_levels"]
w("**Leg (a) − BlackLightPlus is RAPTOR's polarized step.** The two codes march the same states "
  "(stored samples per image identical up to the terminal state of each ray: e.g. i = 17°, C = 80: "
  f"{ra17['80']['samples_raptor']} RAPTOR vs {ra17['80']['samples_blp']} BlackLightPlus, a difference of "
  "one per pixel), share every end class, and with the LD85 step (leg b) the images agree to "
  f"L1(I) ≤ {e(max(met['legs']['rb'][i]['levels'][str(C)]['l1_I'] for i in INCS for C in met['legs']['rb'][i]['C_levels']), 1)}, "
  f"L1(|P|) ≤ {e(max(met['legs']['rb'][i]['levels'][str(C)]['l1_P'] for i in INCS for C in met['legs']['rb'][i]['C_levels']), 1)} and "
  f"L1(V) ≤ {e(max(met['legs']['rb'][i]['levels'][str(C)]['l1_V'] for i in INCS for C in met['legs']['rb'][i]['C_levels']), 1)} "
  "at every level, falling with C: the O(h⁴) truncation difference between the Christoffel form (x^μ, k^μ) and the Hamiltonian form (x^μ, p_μ) of the same RK4 march. "
  "The leg (a) differences in |P| and V fall with C at the rate of RAPTOR's own self-convergence "
  f"(i = 17°: L1(|P|) " + ", ".join(f"{e(ra17[str(C)]['l1_P'], 1)}" for C in Cs17) +
  f" and L1(V) " + ", ".join(f"{e(ra17[str(C)]['l1_V'], 1)}" for C in Cs17) + f" for C = {', '.join(map(str, Cs17))}), "
  "so they are the discretisation error of the trapezoid step, not a model difference.")
w("")
w("**Net EVPA at i = 60° and 90°.** The flux-weighted per-pixel EVPA change of each code between C = 160 and "
  "its finest level, and the Faraday depth through the bright pixels (RAPTOR `tauF`, "
  "10/50/90th percentiles), are:")
w("")
w("| i | Faraday depth p10/p50/p90 [rad] | m_net (BLP, C = 160) | BLP per-pixel EVPA change C=160→finest [°] | RAPTOR (a) same [°] | (a) − BLP per-pixel [°] at C = 160 |")
w("|---|---|---|---|---|---|")
for inc in INCS:
    m = met["legs"]["ra"][inc]["levels"]["160"]
    fd = m["faraday_depth_raptor_bright_pct_10_50_90"]
    w(f"| {inc} | {fd[0]:.0f} / {fd[1]:.0f} / {fd[2]:.0f} | {met['blp'][inc]['observables']['160']['m_net']:.4f} | "
      f"{m['eps_blp']['evpa_pixel_fw_deg']:.1f} | {m['eps_raptor']['evpa_pixel_fw_deg']:.1f} | "
      f"{m['evpa_pixel']['flux_weighted_mean_abs_deg']:.1f} |")
w("")
w("At i = 60° and 90° the rays cross the cold, dense orbital plane; Faraday depths of hundreds to "
  "ten thousand radians scramble the EVPA below the pixel scale, the net polarization m_net is a "
  "residual of order 0.01 of cancelling contributions, and the net EVPA of either code moves by "
  "tens of degrees between C levels. The EVPA_net misses of leg (a) are within 2 max(ε_R, ε_B) and "
  "are classed as discretisation; the 2° threshold is not resolvable for these two inclinations by "
  "either code at C ≤ 320. At i = 17° (median Faraday depth "
  f"{met['legs']['ra']['17']['levels']['160']['faraday_depth_raptor_bright_pct_10_50_90'][1]:.0f} rad) both codes converge and leg (a) meets "
  "the EVPA threshold from C = " +
  str(min(int(C) for C, m in met["legs"]["ra"]["17"]["levels"].items()
          if all(m2["gate"]["evpa_net_deg"] for C2, m2 in met["legs"]["ra"]["17"]["levels"].items() if int(C2) >= int(C)))) + ".")
w("")
w("**Resolution.** BlackLightPlus alone, i = 17°, C = 80 (`metrics.json` `resolution`):")
w("")
w("| npix | F [Jy] | m_net | leg (a): ΔF/F | L1(I) | L1(\\|P\\|) | Δm_net | leg (b): L1(I) | L1(\\|P\\|) |")
w("|---|---|---|---|---|---|---|---|---|")
for n in sorted(met["resolution"].get("ra", {}), key=int):
    a = met["resolution"]["ra"][n]
    bb = met["resolution"].get("rb", {}).get(n)
    w(f"| {n} | {a['observables_blp']['flux_I_Jy']:.5f} | {a['observables_blp']['m_net']:.4f} | {e(a['flux_rel'])} | "
      f"{e(a['l1_I'])} | {e(a['l1_P'])} | {f(a['d_m_net'])} | {e(bb['l1_I']) if bb else '—'} | {e(bb['l1_P']) if bb else '—'} |")
w("")
w("The net linear polarization of s455 at i = 17° changes by more than 0.01 between 240², 480² and "
  "960² in both codes: the Faraday-rotated EVPA pattern is not resolved at the 480² pixel. The "
  "code-to-code comparison holds at every resolution.")
w("")
w("### Report-only legs (C = 80)")
w("")
w("| leg | i | ΔF/F | L1(I) | Δm_net | Δ⟨\\|m\\|⟩ | ΔEVPA_net [°] | Δv_net |")
w("|---|---|---|---|---|---|---|---|")
lab = {"rc": "(c) upstream fits", "rstock": "stock", "rc0": "(c) upstream fits, before the ρ_Q fix",
       "rstock0": "stock, before the ρ_Q fix"}
for leg in ("rc", "rc0", "rstock", "rstock0"):
    for inc in INCS:
        x = met["report_legs"].get(leg, {}).get(inc)
        if not x:
            continue
        m = x["vs_blp"]
        fm = lambda v, n=4: "—" if v != v else f"{v:+.{n}f}"
        w(f"| {lab[leg]} | {inc} | {m['flux_rel']:+.4f} | {m['l1_I']:.4f} | {fm(m['d_m_net'])} | {fm(m['d_mabs_avg'])} | "
          f"{m['d_evpa_net_deg']:+.1f} | {fm(m['d_v_net'], 5)} |")
w("")
rcl = [met["report_legs"]["rc"][i]["vs_blp"] for i in INCS]
w(f"- The upstream fits change the flux by {min(m['flux_rel'] for m in rcl):+.1%} to {max(m['flux_rel'] for m in rcl):+.1%} "
  f"and m_net by up to {max(abs(m['d_m_net']) for m in rcl):.3f} (compare the R03 ratio "
  "table, `evidence/R03/summary.json`).")
w("- Before the ρ_Q fix, the upstream fits blank every pixel whose ray meets a sample with "
  "Θ_e below the K₂ underflow (1/Θ_e ≳ 705; the largest non-finite value on the test grid is at "
  "Θ_e = " + cold[0].split()[-1] + "): a NaN ρ_Q makes I_pol NaN, `POLARIZATION_ACTIVE` is cleared and the pixel reads "
  "out zero. Leg (c) has no Θ_e bound and loses " +
  ", ".join(f"{-met['report_legs']['rc0'][i]['vs_blp']['flux_rel']:.0%} (i = {i}°)" for i in INCS) + " of "
  "the flux. The stock images are the same before and after the fix: with its Θ_e ≥ 1e−3 cut no s455 "
  "sample reaches the NaN range.")
stl = [met["report_legs"]["rstock"][i]["vs_blp"]["flux_rel"] for i in INCS]
w(f"- The stock leg is {-max(stl):.0%}–{-min(stl):.0%} fainter. The upstream decode takes Θ_e = (u/ρ)(1/3)(m_p/m_e)/(1 + R), "
  "and u/(3ρ) = p/(3(γ − 1)ρ) = 0.75 p/ρ for γ = 13/9, i.e. 0.75 × the pressure-based Θ_e; it adds the Θ_e ∈ [1e−3, 100] "
  "and r ≤ 50 cuts, n_e = ρ/(m_p + m_e) + 1e−6 cm⁻³, the 1e−6 b² floor and the upstream constants, at the "
  "same mass scale; the flux change is not decomposed into these terms.")
w("")
w("## CPU timings (R08)")
w("")
w(f"Image: {tim['image']}. Host: {tim['host']}. Protocol: {tim['protocol']}.")
w("")
w("| run | load | setup | trace | transfer | output | end-to-end [s] | rays/s (trace + transfer) | samples/s | peak RSS [GB] |")
w("|---|---|---|---|---|---|---|---|---|---|")
for k, v in tim["runs"].items():
    p = v["phases_s"]
    w(f"| {v['label']} | {p['load']:.2f} | {p['setup']:.3f} | {p['trace']:.2f} | {p['transfer']:.2f} | {p['output']:.2f} | "
      f"{v['end_to_end_s']:.2f} | {v['rays_per_s']:.0f} | {v['samples_per_s']:.3g} | {v['maxrss_GB']:.1f} |")
w("")
blps = [v for v in tim["runs"].values() if v["code"] == "BlackLightPlus"]
if blps:
    ths = sorted({v["threads"] for v in tim["runs"].values()})
if len(ths) >= 2:
    lo, hi = ths[0], ths[-1]
    parts = []
    for code, key in (("BlackLightPlus", "blp_t{}"), ("RAPTOR (a)", "raptor_a_t{}"), ("RAPTOR (b)", "raptor_b_t{}")):
        a, b_ = tim["runs"].get(key.format(lo)), tim["runs"].get(key.format(hi))
        if a and b_:
            sp = b_["rays_per_s"] / a["rays_per_s"]
            parts.append(f"{code} {sp:.2f}× ({100 * sp / (hi / lo):.0f}%)")
    w(f"Strong scaling of the ray phases (trace + transfer) from {lo} to {hi} threads: " + ", ".join(parts) + ". "
      f"At {hi} threads BlackLightPlus is {tim['runs'][f'blp_t{hi}']['rays_per_s'] / tim['runs'][f'raptor_a_t{hi}']['rays_per_s']:.2f}× "
      f"RAPTOR (a) in rays/s and {tim['runs'][f'raptor_a_t{hi}']['end_to_end_s'] / tim['runs'][f'blp_t{hi}']['end_to_end_s']:.2f}× faster end to end.")
    w("")
w("BlackLightPlus compilation (24² warm-up render, excluded above): " +
      ", ".join(f"{v['threads']} thr {v['jit_warmup_s']:.1f} s" for v in blps) + ".")
w("")
w("Phase definitions. RAPTOR (`RAPTOR_TIMERS`, M12): load = `read_model` + `init_model` (snapshot read); "
  "setup = constants and camera blocks; trace and transfer = the wall of the pixel loop split in "
  "proportion to the thread-summed geodesic (camera ray, march) and transfer times; output = spectrum, "
  "HDF5 and ASCII image writes. BlackLightPlus (`r07_blp.jl`): load = header, binary setup and "
  "`athenak_bin_snapshot`; setup = `camera_rays`; trace = `trace_rays_sampled`; transfer = "
  "`polarized_march` over the rays; output = five `.npy` writes. RAPTOR's peak RSS is the process "
  "maximum; BlackLightPlus reports `Sys.maxrss()` of the Julia process, which holds the image states "
  "of every ray (`trace_rays_sampled` stores the path before the transfer) and the compiled code.")
w("")
w("Load average at the start and end of each leg (1/5/15 min; the 1-min value during a 64-thread leg includes "
  "the leg's own threads): " + "; ".join(tim["loadavg_log"]) + ".")
w("")
w("**GPU legs (deferred, `dag.md` R11).** No usable device on this host (NVML error). The legs on a host with A100s:")
w("")
w("- BlackLightPlus: the s455 image is `march_rays_gpu_spacetime(cfg, st, snap, camera_rays(st, cam); "
  "polarized = true, C = 80, nmax = 4000, devices = [0])` and `devices = [0, 1]` (`src/gpu_spacetime.jl`: the "
  "CPU march in-kernel on the superposed Kerr–Schild metric, Float64, one snapshot upload per device), with "
  "`cfg`, `st`, `snap`, `cam` of `tools/r07_blp.jl`; the timed phases are the snapshot upload, the fused "
  "march and the device-to-host copy, best of 3 warm runs, and the image is checked against the CPU "
  "image of `r07_blp.jl` (L1 ≤ 1e−10). The Kerr-deck scaling legs of `bench/bench_multigpu.jl` "
  "(`julia --project=. -t 8 bench/bench_multigpu.jl`) give the pool's 1→2 device speed-up.")
w("- RAPTOR: M13 (direct calls instead of the `rk4_step` function pointer, iterative `determ`, "
  "preallocated per-ray buffers, device Bessels) and `#pragma acc parallel loop gang vector` over the "
  "M12 pixel loop; built with `nvc -acc=gpu -gpu=cc80` in an `r07_build.sh` run directory and run with "
  "`r07_raptor.sh` on the same inputs (one device; RAPTOR has no multi-device path). The port is not "
  "built; `nvc -mp` builds of the CPU sources exist (`fork.md`).")
w("")
sys.path.insert(0, HERE)
import r09_report  # noqa: E402
has_r09 = r09_report.render_r09(w, EV)
has_r10 = r09_report.render_r10(w, EV)
w("## Figures")
w("")
w("- `figures/images_i17.png`, `images_i60.png`, `images_i90.png`: I, |P|, EVPA and V from BlackLightPlus and RAPTOR (a) at C = 320, with the per-pixel differences of legs (a) and (b).")
w("- `figures/convergence.png`: normalized L1 of I, |P| and V against C: each code's self-convergence and the code-to-code differences of legs (a) and (b).")
w("- `figures/timing.png`: phase walls of the R08 runs.")
if has_r09:
    w("- `figures/r09_s122_images_i17.png` … `r09_s093_images_i90.png`: the R09 panels (as `images_i*.png`) for s122 and s093 at C = 320; `figures/r09_s122_convergence.png`, `r09_s093_convergence.png`: their convergence curves.")
if has_r10:
    w("- `figures/r10_i90.png`: the R10 deck: BlackLightPlus uniform, RAPTOR (b) uniform and the block-averaged AMR mosaic (I, |P|, EVPA, V), with the differences RAPTOR − BlackLightPlus and RAPTOR composite − AMR; `figures/r10_zone.png`: the central 20 M zone at the finest AMR pixel.")
w("")
w("## Caveats")
w("")
w("- The thresholds are evaluated on the fiducial mass scale of the BlackLightPlus figure record; neither code is re-calibrated.")
w("- m_net, ⟨|m|⟩ and EVPA_net of s455 at i = 60° and 90° are not converged in C or in resolution in either code (Faraday depth 10²–10⁴ rad); they are compared code to code at equal discretisation only.")
w("- RAPTOR's upstream Bardeen camera, native step size and single-hole termination are not exercised; the stock leg runs on the matched geometry.")
w("- Leg (d) of `dag.md` (RAPTOR's native `stepsize()`) is not run: the native step and termination are single-hole (r from the origin, horizon of a hole at the origin) and do not terminate at the two SKS horizons.")
w("- Timings cover 16 and 64 threads (the host is shared; `dag.md` lists 1, 16, 64, 128).")
w("- Timings are on a shared host; the load average is recorded next to each leg. RAPTOR's rays/s include the pixel-loop scheduling of 230,400 rays in chunks of 64; BlackLightPlus threads over rays in the same way (`Threads.@threads :dynamic`).")
w("")
open(OUT, "w").write("\n".join(L) + "\n")
print("wrote", OUT)
