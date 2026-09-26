"""r09_collect.py — per-image run records of the R09 image legs into one evidence file:
BlackLightPlus flux, classes, sampled-state safety (smallest r_A - r_h,A and r_A over all
stored states, stored states with g^tt >= 0), load time, walls and peak RSS; RAPTOR walls,
M12 phase timers (load = snapshot read) and peak RSS.
Usage: python3 r09_collect.py <out.json> <tag>=<rundir> [...]
Exit 0 iff no stored state of any BlackLightPlus image has g^tt >= 0."""
import json
import os
import sys

out = {}
ok = True
for spec in sys.argv[2:]:
    tag, run = spec.split("=", 1)
    d = {"blp": {}, "raptor": {}}
    for name in sorted(os.listdir(run)):
        p = os.path.join(run, name)
        if name.startswith("blp_") and os.path.isfile(os.path.join(p, "record.json")):
            r = json.load(open(os.path.join(p, "record.json")))
            d["blp"][name] = {k: r[k] for k in ("flux_IQUV_Jy", "classes_cap1_cap2_esc_nmax_nonfinite",
                                                "sample_safety", "samples", "maxrss_GB", "threads",
                                                "rays_per_s", "git_sha")}
            d["blp"][name]["timing_s"] = r["timing_best_s"]
            ok &= r["sample_safety"]["states_gtt_nonneg"] == 0
            d.setdefault("a_used", r["a_used"])
            d.setdefault("centers", r["centers"])
            d.setdefault("mass_scale_g", r["mass_scale_g"])
            d.setdefault("snapshot", r["snapshot"])
        elif name[:3] in ("ra_", "rb_") and os.path.isfile(os.path.join(p, "timing.json")):
            t = json.load(open(os.path.join(p, "timing.json")))
            d["raptor"][name] = {"wall_s": t["wall_best_s"], "maxrss_GB": t["maxrss_GB"],
                                 "threads": t["threads"], "phases": t.get("phases_best")}
    saf = [v["sample_safety"] for v in d["blp"].values()]
    d["min_horizon_gap_all_images"] = min(s["min_horizon_gap"] for s in saf)
    d["min_rA_all_images"] = min(s["min_rA"] for s in saf)
    d["states_gtt_nonneg_all_images"] = sum(s["states_gtt_nonneg"] for s in saf)
    d["blp_load_s"] = max(v["timing_s"]["load"] for v in d["blp"].values())
    d["blp_maxrss_GB"] = max(v["maxrss_GB"] for v in d["blp"].values())
    d["raptor_load_s"] = max((v["phases"] or {}).get("load", 0.0) for v in d["raptor"].values())
    d["raptor_maxrss_GB"] = max(v["maxrss_GB"] for v in d["raptor"].values())
    fo = {}
    for inc in (17, 60, 90):
        fr = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "figures", "out", "eht_bbh",
                          f"{tag}_i{inc}_Rh40", "records.json")
        if os.path.isfile(fr):
            fo[str(inc)] = json.load(open(fr))["diagnostics"]["frac_flux_outside_fov"]
    d["figure_record_frac_flux_outside_fov"] = fo
    out[tag] = d
    print(f"{tag}: min r_A - r_h over all stored states {d['min_horizon_gap_all_images']:.3e} (min r_A "
          f"{d['min_rA_all_images']:.4f}), states with g^tt >= 0: {d['states_gtt_nonneg_all_images']} | load BLP "
          f"{d['blp_load_s']:.1f} s RAPTOR {d['raptor_load_s']:.1f} s | peak RSS BLP {d['blp_maxrss_GB']:.1f} GB "
          f"RAPTOR {d['raptor_maxrss_GB']:.1f} GB")
out["pass"] = bool(ok)
json.dump(out, open(sys.argv[1], "w"), indent=1)
sys.exit(0 if ok else 1)
