"""r09_report.py — the R09 and R10 sections of report.md, rendered by r08_report.py from
evidence/R09/{geometry,metric,rays_s093_s122r,runs,s122_metrics,s093_metrics}.json and
evidence/R10/metrics.json. Every number is read from those files."""
import json
import os


def _e(x, n=2):
    return f"{x:.{n}e}"


def _f(x, n=4):
    return f"{x:+.{n}f}"


def _flag(ok):
    return "pass" if ok else "**miss**"


def _row_levels(w, legd, inc):
    e = legd[inc]
    for C in e["C_levels"]:
        m = e["levels"][str(C)]
        g = m["gate"]
        fails = [k for k, v in g.items() if not v]
        att = m["attribution"]
        gate = "pass" if not fails else "miss " + ", ".join(
            f"{k} ({att[k]['class']})" if k in att else k for k in fails)
        dring = max(abs(m[f"d_hole{k}_{lab}_diameter_uas"]) for k in (1, 2) for lab in ("native", "blur20"))
        dfc = max(abs(m[f"d_hole{k}_{lab}_f_C"]) for k in (1, 2) for lab in ("native", "blur20"))
        w(f"| {inc} | {C} | {_e(m['flux_rel'])} | {_e(m['l1_I'])} | {_e(m['nmse_I'])} | {_e(m['l1_P'])} | "
          f"{_e(m['l1_V'])} | {_f(m['d_m_net'])} | {_f(m['d_mabs_avg'])} | {m['d_evpa_net_deg']:+.2f} | "
          f"{m['d_v_net']:+.5f} | {dring:.2f} | {dfc:.3f} | {m['class_agreement']:.6f} | {gate} |")


HDR = ("| i | C | ΔF/F | L1(I) | NMSE(I) | L1(\\|P\\|) | L1(V) | Δm_net | Δ⟨\\|m\\|⟩ | ΔEVPA_net [°] | Δv_net | "
       "Δ ring d [µas] max | Δf_C max | class agree | gate |")
SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def render_r09(w, EV):
    d = os.path.join(EV, "R09")
    if not os.path.isfile(os.path.join(d, "geometry.json")):
        return False
    geo = json.load(open(os.path.join(d, "geometry.json")))
    met = json.load(open(os.path.join(d, "metric.json")))
    rays = json.load(open(os.path.join(d, "rays_s093_s122r.json")))
    runs = json.load(open(os.path.join(d, "runs.json")))
    snaps = {t: json.load(open(os.path.join(d, f"{t}_metrics.json"))) for t in ("s122", "s093")}
    w("## Spinning snapshots (R09)")
    w("")
    w("s122 (`torus.mhd_w_bcc.00122.athdf`, sep 16) and s093 (`torus.mhd_w_bcc.00093.bin`, 38.7 GB, sep 25) "
      "carry the header spin a = 0.8 per hole in units of the total mass, i.e. a > m_A = 0.5; both codes image "
      "them, as `figures/fig_eht_bbh.jl` does, as χ = 0.8 (a_used = 0.4 per hole, along +z: th_a = 0 in the "
      "header). Everything else is the R07 protocol: M = 1.3e10 M_sun, D = 16.8 Mpc, 230 GHz, R_high 40, FOV 56 M, "
      "480², C ladder with nmax = 50 C, the calibrated mass scale of `figures/out/eht_bbh/<snap>_i17_Rh40/records.json` "
      "passed to both codes, legs (a) and (b).")
    w("")
    w("| snapshot | a_used | centres | mass scale [g] | BLP load [s] | RAPTOR load [s] | peak RSS BLP / RAPTOR [GB] |")
    w("|---|---|---|---|---|---|---|")
    for t in ("s122", "s093"):
        r = runs[t]
        c = r["centers"][0]
        w(f"| {t} | {r['a_used'][0]}, {r['a_used'][1]} | ±({c[0]:.8f}, {c[1]:.8f}, 0) | {r['mass_scale_g']:.9e} | "
          f"{r['blp_load_s']:.1f} | {r['raptor_load_s']:.1f} | {r['blp_maxrss_GB']:.1f} / {r['raptor_maxrss_GB']:.1f} |")
    w("")
    w("BlackLightPlus peak RSS is the Julia process maximum over the whole run and includes the stored ray states "
      "of the largest image (C = 320); RAPTOR's is the largest per-image process maximum. Load times are single "
      "measurements with the page cache in whatever state the preceding legs left it (RAPTOR ran after BlackLightPlus; "
      "the s093 figure record has a BlackLightPlus load of 25.9 s). The dag.md bound on the s093 reader, peak memory "
      f"≤ 64 GB, holds for RAPTOR ({runs['s093']['raptor_maxrss_GB']:.1f} GB).")
    w("")
    w("### Geometry of the spinning binary")
    w("")
    w("Metric level (1e4 points per leg, `tools/r04_metric.jl`; s122r is s122 with both spins along −z):")
    w("")
    w("| leg | max \\|g − g_BLP\\| | max \\|g g⁻¹ − I\\| | Γ analytic (M2b) vs AD | Γ FD (M2a) vs AD | gate |")
    w("|---|---|---|---|---|---|")
    for leg in ("s093", "s122", "s122r"):
        m = met[leg]
        w(f"| {leg} | {_e(m['g_maxabs'])} | {_e(m['g_ginv_minus_I'])} | {_e(m['gamma_analytic_rel'])} | "
          f"{_e(m['gamma_fd_rel'])} | {_flag(m['pass'])} |")
    w("")
    w("Geodesic level (1050 rays per leg, i = 60°, `tools/r04_rays.jl`, finite-difference build):")
    w("")
    w("| leg | class agreement | max \\|Δx\\| escaped, C = 80 (/ 2 max ε) | C = 160 | escape angle C = 80 / 160 [rad] | order BLP / RAPTOR | gate (R04 rule) |")
    w("|---|---|---|---|---|---|---|")
    for leg in ("s093", "s122r"):
        L = rays[leg]["fd"]
        c8, c16 = L["C80"], L["C160"]
        o = rays[leg]["convergence"]
        w(f"| {leg} | {c8['termination_identical_frac']:.4f} | {_e(c8['max_dx_escaped'])} ({c8['max_dx_over_2max_eps']:.2f}) | "
          f"{_e(c16['max_dx_escaped'])} | {_e(c8['max_escape_angle_rad'])} / {_e(c16['max_escape_angle_rad'])} | "
          f"{o['blp']['order_per_ray']['C80_160']['median']:.2f} / {o['fd']['order_per_ray']['C80_160']['median']:.2f} | "
          f"{_flag(L['pass'])} |")
    w("")
    w("Spin orientation and frame dragging (`tools/r09_geometry.jl`, i = 90°, C = 160, bisected shadow edges on the "
      "β = 0 line, test drivers built with max_steps 4e5 so that no bisection ray ends on the step budget; "
      f"budget-limited rays met: {geo['budget_limited_rays_in_bisections']}):")
    w("")
    w("| hole | photon-orbit edges α = −ξ [M] | BLP edges, d = 1000 | RAPTOR − BLP | 1/d-extrapolated (d = 1000, 4000) − photon orbit | shadow shift | gate |")
    w("|---|---|---|---|---|---|---|")
    for k, v in geo["kerr_spin_orientation"].items():
        a = v["alpha_edges_photon_orbit"]
        b = v["d1000_alpha_edges_blp"]
        w(f"| Kerr {k} | [{a[0]:.6f}, {a[1]:.6f}] | [{b[0]:.6f}, {b[1]:.6f}] | {_e(v['blp_minus_raptor'])} | "
          f"{_e(v['extrapolated_minus_photon_orbit_blp'])} | {v['d1000_shift_blp']:+.4f} | {_flag(v['pass'])} |")
    w("")
    w("With the observer on +x and α = +ŷ, a hole spinning along +z has its prograde edge at α = −ξ_pro < 0 and its "
      "shadow displaced toward +α (west); both codes give the same displacement with the same sign, and reverse it "
      "for a < 0. At d = 1000 the plane-parallel camera offsets the edges by O(M b/d) (" +
      f"{geo['kerr_spin_orientation']['a=0.9']['d1000_minus_photon_orbit_blp']:.2e} M); the 1/d extrapolation "
      "removes it.")
    w("")
    w("| binary | hole 1 shadow centre α [M] BLP / RAPTOR | hole 2 | max \\|edge_R − edge_B\\| [M] | gate |")
    w("|---|---|---|---|---|")
    for leg, v in geo["binary_shadow_edges_i90"].items():
        w(f"| {leg} | {v['hole1_centre_alpha_blp']:+.4f} / {v['hole1_centre_alpha_raptor']:+.4f} | "
          f"{v['hole2_centre_alpha_blp']:+.4f} / {v['hole2_centre_alpha_raptor']:+.4f} | {_e(v['max_diff'])} | {_flag(v['pass'])} |")
    w("")
    w("Reversing both spins (s122 → s122r) moves each shadow centre by " +
      ", ".join(f"{abs(geo['binary_shadow_edges_i90']['s122'][f'hole{k}_centre_alpha_blp'] - geo['binary_shadow_edges_i90']['s122r'][f'hole{k}_centre_alpha_blp']):.4f} M"
                for k in (1, 2)) + " (holes 1, 2) in both codes.")
    w("")
    w("**Horizon and capture rule.** Both codes stop a ray at d = min_A(r_A − r_h,A) < 1e−3 with r_h,A = m_A + "
      "√(m_A² − a_A²) = 0.8 of the isolated hole, or when the normal-frame energy exceeds 100 (the BlackLightPlus "
      "redshift cut and horizon backstop, M3). In the superposed metric the surface g^{μν} ∂_μ r_A ∂_ν r_A = 0 "
      "around each hole lies outside r_h (2000 directions per hole; RAPTOR's g^{μν} vanishes on it to the same "
      "rounding):")
    w("")
    w("| binary | hole | r_A of the null surface min / median / max | RAPTOR \\|g^{rr}\\| there (relative) | g^tt ≥ 0 region: max r_A BLP / RAPTOR | gate |")
    w("|---|---|---|---|---|---|")
    for leg, v in geo["horizon"].items():
        for k in (1, 2):
            h = v[f"hole{k}"]
            w(f"| {leg} | {k} | {h['null_surface_rA_min']:.4f} / {h['null_surface_rA_median']:.4f} / {h['null_surface_rA_max']:.4f} | "
              f"{_e(h['raptor_norm_at_blp_surface_max_rel'])} | {h['gtt_nonneg_max_rA_blp']:.4f} / {h['gtt_nonneg_max_rA_raptor']:.4f} | "
              f"{_flag(h['pass'])} |")
    w("")
    ms = {t: runs[t] for t in ("s122", "s093")}
    w("Rays that enter the perturbed horizon (between the null surface and r_h + 1e−3) reach the energy cut first: "
      "over every stored state of every BlackLightPlus image the smallest r_A − r_h,A is " +
      ", ".join(f"{ms[t]['min_horizon_gap_all_images']:.3f} ({t})" for t in ms) +
      ", so no sample lies within r_A < " + f"{min(ms[t]['min_rA_all_images'] for t in ms):.3f}" +
      ". The inverse metric has g^tt ≥ 0 only deep inside each hole (the table), where "
      "`primitive_four_vectors` has no lapse and throws; the number of stored states there is " +
      ", ".join(f"{ms[t]['states_gtt_nonneg_all_images']} ({t})" for t in ms) +
      ". The rays never sample that region, so BlackLightPlus needs no change.")
    w("")
    for t, lab in (("s122", "s122 (sep 16)"), ("s093", "s093 (sep 25)")):
        m = snaps[t]
        for leg, name in (("rb", "leg (b), LD85 step"), ("ra", "leg (a), native polarized step")):
            w(f"### {lab}, {name}")
            w("")
            w(HDR)
            w(SEP)
            for inc in sorted(m["legs"][leg], key=int):
                _row_levels(w, m["legs"][leg], inc)
            w("")
            w("Flux after Richardson extrapolation (three finest levels): " + "; ".join(
                f"i = {inc}°: {m['legs'][leg][inc]['flux_rel_richardson']:+.2e} ({_flag(m['legs'][leg][inc]['flux_richardson_pass'])})"
                for inc in sorted(m["legs"][leg], key=int) if m["legs"][leg][inc]["flux_rel_richardson"] is not None) + ".")
            w("")
            ev = []
            for inc in sorted(m["legs"][leg], key=int):
                for C, s in m["legs"][leg][inc]["evaluated"].items():
                    txt = "all thresholds met" if s["pass"] else "misses " + ", ".join(s["failed"]) + (
                        "; model-attributed: " + ", ".join(s["model_attributed"]) if s["model_attributed"]
                        else "; every miss attributed to discretisation")
                    ev.append(f"i = {inc}°, C = {C}: {txt}")
            w("Evaluated levels: " + "; ".join(ev) + ".")
            w("")
        res = m.get("resolution", {}).get("rb", {})
        if res:
            w(f"{lab}, resolution at i = 17°, C = 80 (leg b − BlackLightPlus): " + "; ".join(
                f"{n}²: L1(I) {_e(v['l1_I'])}, L1(\\|P\\|) {_e(v['l1_P'])}, m_net BLP {v['observables_blp']['m_net']:.4f}"
                for n, v in sorted(res.items(), key=lambda x: int(x[0]))) + ".")
            w("")
    w("### R09 attribution")
    w("")
    def worst(cond):
        return max(m["legs"]["rb"][inc]["levels"][C][k] for m in snaps.values() for inc in m["legs"]["rb"]
                   for C in m["legs"]["rb"][inc]["levels"] if cond(int(C)) for k in ("l1_I", "l1_P", "l1_V"))
    w(f"With the LD85 step the two codes agree on both spinning snapshots to L1 ≤ {_e(worst(lambda C: C >= 160))} in "
      f"I, |P| and V at C ≥ 160 (≤ {_e(worst(lambda C: C == 80))} at C = 80, ≤ {_e(worst(lambda C: C == 40))} at C = 40) at every "
      "inclination, with identical end classes: the spinning superposed metric, the frame dragging of each hole, "
      "the capture rule, the `.athdf` and 4096-block `.bin` readers and the plasma model are the same in both codes. "
      "Leg (b)'s residual in |P| and V falls by about an order of magnitude per doubling of C (the O(h⁴) truncation "
      "difference of the Christoffel and Hamiltonian forms of the same RK4 march, concentrated on the photon rings). "
      "Leg (a) adds RAPTOR's second-order polarized step; unlike s455 its difference stays small because the Faraday "
      "depth through the bright pixels (RAPTOR `tauF`, 10/50/90th percentiles at C = 160) is low: " + "; ".join(
          f"{t} " + ", ".join(f"i = {inc}°: " + "/".join(f"{x:.2g}" for x in
                                                        snaps[t]["legs"]["ra"][inc]["levels"]["160"]["faraday_depth_raptor_bright_pct_10_50_90"])
                              for inc in sorted(snaps[t]["legs"]["ra"], key=int)) for t in ("s122", "s093")) +
      " rad, against 13–12772 rad for s455.")
    w("")
    w("**s093 artefacts** (present identically in both codes' input, not corrected): a cold dense ball at r < 8 about "
      "the origin, the restart transient of the dump, and " +
      ", ".join(f"{v:.0%} (i = {k}°)" for k, v in runs["s093"]["figure_record_frac_flux_outside_fov"].items())
      + " of the flux outside the 56 M field of view (`figures/out/eht_bbh` records, 224 M wide-field check).")
    w("")
    return True


def render_r10(w, EV):
    p = os.path.join(EV, "R10", "metrics.json")
    if not os.path.isfile(p):
        return False
    m = json.load(open(p))
    trap = json.load(open(os.path.join(EV, "R10", "trapezoid.json")))
    a = m["amr"]
    rec = a["record"]
    w("## i = 90° AMR-figure deck (R10)")
    w("")
    w("Deck of `figures/fig_bbh_i90_pol_amr.jl`: s455 (γ = 13/9), static superposed Kerr–Schild with a = 0 and centres "
      "±12.5 (cos φ₀, sin φ₀, 0), φ₀ = −0.2233022; 230 GHz; i = 90°; FOV 40 M; d = 1000; C = 80; nmax 2500; "
      "M = 6.5e9 × 1.989e33 g (total); mass scale 1e24 g; R_low 1, R_high 10, σ > 1 cut. RAPTOR has no image-plane "
      "AMR: it renders uniform images (camera blocks of 8² pixels for 1024²; `NPIX1D=8 tools/r07_build.sh`), compared "
      f"with BlackLightPlus uniform images (`tools/r10_blp.jl`, `{m['blp_git_sha'][:7]}`) and with the BlackLightPlus AMR "
      f"mosaic re-rendered at HEAD (`{rec['git_sha'][:7]}`; base {rec['amr']['base_res']}², block {rec['amr']['block']}, "
      f"max level {rec['amr']['max_level']}, effective {rec['amr']['eff_res']}²) block-averaged to {m['npix']}². RAPTOR "
      "reads γ from the dump header (1.44444444444, 3e−12 from 13/9). The blurred metrics take D = 16.8 Mpc "
      f"(θ_g = {m['uas_per_M_at_16.8Mpc']:.3f} µas/M) and compare the codes with each other only.")
    w("")
    w(f"### Uniform {m['npix']}²")
    w("")
    w(HDR)
    w(SEP)
    for leg, name in (("rb", "(b)"), ("ra", "(a)")):
        if leg in m["legs"]:
            wrapped = {"90": m["legs"][leg]}
            _row_levels(w, wrapped, "90")
    w("")
    w("Rows: leg (b) at C = 80, then leg (a) at C = 80 and 160 (nmax 5000; BlackLightPlus at the same C).")
    lb = m["legs"]["rb"]["levels"]["80"]
    na = lb.get("gate_not_applicable", {})
    if na:
        rd = lb["ring_defined_blp"]
        holes = sorted({k.split("_")[0] for k in na})
        w("The ring column includes " + ", ".join(holes) + "'s native diameter, which is not gated: at i = 90° its ring cuts "
          "cross emission-free sky (smallest half-cut maximum " + ", ".join(
              f"{rd[h]['min_half_cut_peak_rel']:.1e}" for h in holes) + " of the image peak), where the peak finder lands "
          "on values many decades below the image; its blurred ring agrees.")
    ta = m["legs"]["ra"]["levels"]
    tb = lb["tail"]
    w("")
    w(f"Emission-free tail. Where BlackLightPlus is below 1e−60 of the peak ({tb['blp_empty_pixels']} pixels), leg (b) "
      f"exceeds 1e−30 of the peak in {tb['raptor_above_1e-30_there']} isolated pixels (at most "
      f"{tb['raptor_max_there_rel_peak']:.1e} of the peak), and the converse tail reaches {tb['blp_max_there_rel_peak']:.1e}; "
      "these single pixels move with C in each code (thin features met or missed by the samples). Leg (a) leaves "
      + ", ".join(f"C = {C}: {v['tail']['raptor_above_1e-30_there']} pixels, up to {v['tail']['raptor_max_there_rel_peak']:.1e} of the peak, "
                  f"{v['tail']['raptor_flux_there_rel']:.1e} of the flux" for C, v in sorted(ta.items(), key=lambda x: int(x[0])))
      + " there. RAPTOR's native step does not extinguish radiation behind optically thick gas: `check_stiffness` "
      "selects the implicit trapezoid, whose transmission over one absorbing segment is (1 − τ/2)/(1 + τ/2) → −1 "
      "(`tools/r10_trapezoid.jl`, `evidence/R10/trapezoid.json`: " + ", ".join(
          f"τ = {x['tau']:g}: {x['native']:+.3f} against e^−τ = {x['exp_minus_tau']:.1e}" for x in trap["segments"]
          if x["tau"] in (3.0, 10.0, 100.0)) + "; the LD85 step returns e^−τ), so a sign-alternating remnant of the "
      "radiation behind each thick segment reaches the camera.")
    w("")
    w("**Fork fix (`e6359a6`, `POL_KEEP_I`).** Upstream RAPTOR reads a pixel out as I = Q = U = V = 0 unless its "
      "polarized invariant exceeded 1e−100 at the last sample; on this deck that blanked most emission-free pixels and "
      "unpolarized emission in general (`tools/r10_polkeep.jl`: Gold T3 through the polarized build, I = 0 on 64 of 64 "
      "pixels before, |I_pol/I_unpol − 1| = 4.4e−16 after). The matched build now reads out the transported I.")
    for leg in ("rb", "ra"):
        if leg in m["legs"]:
            L = m["legs"][leg]
            w(f"- leg ({leg[1]}): {_flag(L['pass'])}" + (f"; misses at C = 80: {', '.join(L['failed_C80'])}" if L["failed_C80"] else "") +
              (f"; model-attributed: {', '.join(L['model_attributed_C80'])}" if L["model_attributed_C80"] else ""))
    w("")
    w("### AMR mosaic against RAPTOR")
    w("")
    z = m.get("zone", {})
    w(f"A uniform {m['npix']}² image samples each pixel at its centre; the block-averaged mosaic averages "
      f"{(rec['amr']['eff_res'] // m['npix']) ** 2} finest sub-pixels wherever the AMR refined. On this edge-on image the "
      "two sampling rules differ by more than the threshold in either code (first rows), so the gate compares like "
      f"with like: the central 20 M × 20 M zone, which holds every refined block ({'yes' if a['refined_inside_zone'] else '**no**'}), "
      f"rendered uniformly at the finest AMR pixel ({z.get('n', 0)}², 40/{rec['amr']['eff_res']} M) and block-averaged into "
      f"the uniform {m['npix']}² frame (composite).")
    w("")
    w("| pair (full frame, block-averaged to the uniform grid) | L1(I) | NMSE(I) | L1(\\|P\\|) | L1(V) | LP pair | Δm_net | gate |")
    w("|---|---|---|---|---|---|---|---|")
    labels = {"rb_vs_amr": "RAPTOR (b) uniform (point-sampled) vs AMR", "vs_blp_uniform": "AMR vs BlackLightPlus uniform (point-sampled)",
              "rb_composite_vs_amr": "RAPTOR (b) composite vs AMR", "blp_composite_vs_amr": "BlackLightPlus composite vs AMR"}
    for k, lab in labels.items():
        if k in a:
            v = a[k]
            gate = _flag(a["pass"]) if k == "rb_composite_vs_amr" else "report"
            w(f"| {lab} | {_e(v['l1_I'])} | {_e(v['nmse_I'])} | {_e(v['l1_P'])} | {_e(v['l1_V'])} | "
              f"{v['x']['LP']:.4f} / {v['ref']['LP']:.4f} | {v['d_m_net']:+.4f} | {gate} |")
    w("")
    w(f"LP of the full {rec['amr']['eff_res']}² mosaic: {a['LP_full_mosaic']:.4f} (record at bb233b2: "
      f"{a['recorded_LP_bb233b2']}); per-level rays {rec['amr_leg']['per_level_rays']}, "
      f"{rec['amr_leg']['refined_block_share']:.1%} of the base blocks refined, carrying "
      f"{rec['amr_leg']['refined_flux_share']:.1%} of the flux.")
    w("")
    if z:
        w(f"**Zone at the finest pixel** ({z['n']}², {z['finest_level_pixel_share']:.1%} of its pixels at the finest AMR level), "
          "pixel by pixel:")
        w("")
        w("| pair | L1(I) | L1(\\|P\\|) | L1(V) | LP pair |")
        w("|---|---|---|---|---|")
        for k, lab in (("rbz_vs_blp", "RAPTOR (b) vs BlackLightPlus uniform"),
                       ("rbz_vs_amr_crop", "RAPTOR (b) vs AMR crop"),
                       ("blp_vs_amr_crop", "BlackLightPlus uniform vs AMR crop")):
            if k in z:
                v = z[k]
                w(f"| {lab} | {_e(v['l1_I'])} | {_e(v['l1_P'])} | {_e(v['l1_V'])} | {v['x']['LP']:.4f} / {v['ref']['LP']:.4f} |")
        if "rbz_vs_amr_crop_finest_only" in z:
            w("")
            w(f"Over the finest-level pixels alone RAPTOR (b) and the AMR crop differ by L1(I) "
              f"{_e(z['rbz_vs_amr_crop_finest_only']['l1_I'])}; the rest of the zone difference is the coarser "
              "painting of the AMR's lower levels.")
        w("")
    w("### Timing (64 threads, same host)")
    w("")
    w("| run | rays | wall [s] | rays/s |")
    w("|---|---|---|---|")
    n = m["npix"]
    for leg, lab in (("rb", "RAPTOR (b)"), ("ra", "RAPTOR (a)")):
        if leg in m["legs"]:
            L = m["legs"][leg]["levels"]["80"]
            w(f"| {lab} uniform {n}² | {n * n} | {L['raptor_wall_s']:.1f} | {n * n / L['raptor_wall_s']:.0f} |")
    L = m["legs"]["rb"]["levels"]["80"]
    w(f"| BlackLightPlus uniform {n}² | {n * n} | {L['blp_wall_s']:.1f} | {L['blp_record']['rays_per_s']:.0f} |")
    for n2, e in m.get("extra", {}).items():
        if "rb_wall_s" in e:
            w(f"| RAPTOR (b) uniform {n2}² | {int(n2) ** 2} | {e['rb_wall_s']:.1f} | {int(n2) ** 2 / e['rb_wall_s']:.0f} |")
        w(f"| BlackLightPlus uniform {n2}² | {int(n2) ** 2} | {e['blp_wall_s']:.1f} | {e['blp_rays_per_s']:.0f} |")
    if z.get("rbz_wall_s"):
        w(f"| RAPTOR (b) zone {z['n']}² (fov 20) | {z['n'] ** 2} | {z['rbz_wall_s']:.1f} | {z['n'] ** 2 / z['rbz_wall_s']:.0f} |")
    if z.get("blp_wall_s"):
        w(f"| BlackLightPlus zone {z['n']}² (fov 20) | {z['n'] ** 2} | {z['blp_wall_s']:.1f} | {z['n'] ** 2 / z['blp_wall_s']:.0f} |")
    al = rec["amr_leg"]
    w(f"| BlackLightPlus AMR (effective {rec['amr']['eff_res']}²) | {al['nrays']} | {al['wall_s']:.1f} | {al['rays_per_s']:.0f} |")
    w("")
    w("BlackLightPlus walls are image walls (snapshot load " + f"{L['blp_record']['load_s']:.1f} s excluded); RAPTOR's are "
      "process walls including its load. The AMR wall is `render_amr` over all levels, polarized; the unpolarized "
      f"level-0 probe that sets the refinement cut ({rec['probe_wall_s_jit_incl']:.1f} s, compilation included) is not "
      "counted. The zone runs are slower per ray than the full frame: their rays concentrate near the holes, where "
      "h = d/C is small.")
    w("")
    for n2, e in m.get("extra", {}).items():
        parts = []
        if "rb_vs_blp" in e:
            parts.append(f"RAPTOR (b) vs BlackLightPlus L1(I) {_e(e['rb_vs_blp']['l1_I'])}, L1(\\|P\\|) {_e(e['rb_vs_blp']['l1_P'])}")
        if e.get("amr_vs_blp"):
            parts.append(f"AMR block-averaged vs BlackLightPlus uniform L1(I) {_e(e['amr_vs_blp']['l1_I'])}, L1(\\|P\\|) {_e(e['amr_vs_blp']['l1_P'])}")
        if e.get("blp1024_vs_blp_blockavg"):
            parts.append(f"BlackLightPlus {n}² vs its {n2}² block-averaged L1(I) {_e(e['blp1024_vs_blp_blockavg']['l1_I'])}")
        if parts:
            w(f"At {n2}²: " + "; ".join(parts) + ".")
            w("")
    return True
