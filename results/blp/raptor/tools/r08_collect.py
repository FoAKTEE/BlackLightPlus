"""r08_collect.py — collect the R08 CPU timings (r08_timing.sh output) into one evidence
file. Per code, leg and thread count: phase walls (best of the repeats; RAPTOR from its
M12 RAPTOR_TIMERS line, BlackLightPlus from r07_blp.jl), process wall (RAPTOR,
/usr/bin/time), rays/s over the ray phases (trace + transfer), samples/s, peak RSS,
and the load average around each leg. The same image (s455, i = 17, 480^2, polarized,
C = 80) in both codes.
Usage: python3 r08_collect.py <r08 outdir> <out.json> [npix]"""
import json
import os
import statistics
import sys

d, out = sys.argv[1], sys.argv[2]
npix = int(sys.argv[3]) if len(sys.argv) > 3 else 480
runs = {}
PH = ("load", "setup", "trace", "transfer", "output")
for name in sorted(os.listdir(d)):
    p = os.path.join(d, name)
    if name.startswith("t") and name.endswith(f"_i17_C80_n{npix}"):
        tag = name.split("_")[0]                    # t<leg><threads>
        leg, th = tag[1], int(tag[2:])
        t = json.load(open(os.path.join(p, "timing.json")))
        pb = t["phases_best"]
        ph = {k: pb[k] for k in PH}
        raywall = pb["trace"] + pb["transfer"]
        runs[f"raptor_{leg}_t{th}"] = {
            "code": "RAPTOR", "leg": leg, "threads": th,
            "label": f"RAPTOR ({leg}{', LD85 step' if leg == 'b' else ', native step'}), {th} thr",
            "phases_s": ph, "phases_runs": t["phases_runs"],
            "process_wall_s": [r["wall_s"] for r in t["runs"]],
            "process_wall_best_s": t["wall_best_s"],
            "process_wall_median_s": statistics.median(r["wall_s"] for r in t["runs"]),
            "loop_wall_best_s": pb["loop"],
            "thread_seconds_trace": pb["trace_cpu"], "thread_seconds_transfer": pb["transfer_cpu"],
            "rays": int(pb["rays"]), "samples": int(pb["samples"]),
            "rays_per_s": pb["rays"] / raywall, "samples_per_s": pb["samples"] / raywall,
            "end_to_end_s": sum(ph.values()), "maxrss_GB": t["maxrss_GB"]}
    if name.startswith("blp_t"):
        th = int(name[5:])
        sub = os.path.join(p, f"blp_i17_C80_n{npix}", "record.json")
        r = json.load(open(sub))
        tb = r["timing_best_s"]
        ph = {k: tb[k] for k in PH}
        raywall = tb["trace"] + tb["transfer"]
        runs[f"blp_t{th}"] = {
            "code": "BlackLightPlus", "threads": th, "label": f"BlackLightPlus (CPU), {th} thr",
            "phases_s": ph, "image_walls_s": r["walls_s"], "image_wall_best_s": tb["image_wall"],
            "image_wall_median_s": statistics.median(r["walls_s"]),
            "jit_warmup_s": tb["jit_warmup"], "rays": r["rays"], "samples": r["samples"],
            "rays_per_s": r["rays"] / raywall, "samples_per_s": r["samples"] / raywall,
            "end_to_end_s": sum(ph.values()), "maxrss_GB": r["maxrss_GB"], "git_sha": r["git_sha"]}
load = open(os.path.join(d, "loadavg.log")).read().splitlines() if os.path.isfile(os.path.join(d, "loadavg.log")) else []
order = sorted(runs, key=lambda k: (runs[k]["threads"], runs[k]["code"] != "BlackLightPlus", k), reverse=False)
res = {"image": f"s455, i = 17 deg, {npix}^2, polarized, C = 80, nmax 4000, 230 GHz",
       "host": "2x Xeon Platinum 8360Y (72 cores, 144 hardware threads), shared",
       "protocol": "best of 3 warm runs per leg (page cache warm; BlackLightPlus compiled in a 24^2 warm-up render, "
                   "excluded); RAPTOR OMP_PROC_BIND=close OMP_PLACES=cores; BlackLightPlus julia -t T; "
                   "phases: load (snapshot read), setup (camera), trace (geodesics), transfer (polarized RT), output",
       "loadavg_log": load, "runs": {k: runs[k] for k in order},
       "title": f"s455 i=17° {npix}² polarized C=80: CPU phase walls (best of 3)"}
json.dump(res, open(out, "w"), indent=1)
for k in order:
    v = runs[k]
    print(f"{v['label']:40s} " + " ".join(f"{p} {v['phases_s'][p]:7.2f}" for p in PH) +
          f" | end-to-end {v['end_to_end_s']:7.2f} s | {v['rays_per_s']:8.0f} rays/s | {v['maxrss_GB']:.1f} GB")
