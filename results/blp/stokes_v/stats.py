import numpy as np, os, sys
D = "/home/haiyangw/.claude/jobs/b085d9a7/tmp/kpolaris/diag_v"
B = "/data/haiyangw/claude/BlackLightPlus/figures/out/kpolaris/diag_v"
T = "/home/haiyangw/.claude/jobs/b085d9a7/tmp/kpolaris/bl_runs"
def load_bl(variant, nu):
    f = f"{T}/fast_{nu}.npz" if variant == "full" else f"{D}/bl_{variant}_{nu}.npz"
    if not os.path.exists(f): return None
    z = np.load(f); return {k: np.array(z[k + "_nu"], float) for k in "IQUV"}
def load_blp(variant, nu):
    f = f"{B}/blp_{variant}_{nu}_I.npy"
    if not os.path.exists(f): return None
    return {k: np.load(f"{B}/blp_{variant}_{nu}_{k}.npy") for k in "IQUV"}
def stats(a, b, m, thr=0.1):
    sig = m & (np.abs(b) > thr * np.nanmax(np.abs(b[m])))
    same = np.mean(np.sign(a[sig]) == np.sign(b[sig])) if sig.any() else np.nan
    r = np.sum(np.abs(a[m])) / np.sum(np.abs(b[m])); c = np.corrcoef(a[m], b[m])[0, 1]
    return r, c, same
for nu in ("1000", "230"):
    print(f"==== {nu} GHz")
    for v in ("full", "norho", "norhoq", "norhov"):
        bl, blp = load_bl(v, nu), load_blp(v, nu)
        if bl is None or blp is None: print(f"  {v}: missing", bl is None, blp is None); continue
        m = np.isfinite(bl["I"]) & np.isfinite(bl["V"])
        out = [f"  {v:7s} I ratio {blp['I'][m].sum()/bl['I'][m].sum():.4f}"]
        for k in "QUV":
            r, c, s = stats(blp[k], bl[k], m)
            out.append(f"{k}: ratio {r:.2f} corr {c:+.3f} sign {s:.2f} net {blp[k][m].sum()/blp['I'][m].sum():+.4f}/{bl[k][m].sum()/bl['I'][m].sum():+.4f}")
        print(" | ".join(out))
    # cross-variant within each code: how much do the Faraday terms change V
    for code, load in (("BLP", load_blp), ("BL", load_bl)):
        full, norho = load("full", nu), load("norho", nu)
        if full is None or norho is None: continue
        m = np.isfinite(full["V"]) & np.isfinite(norho["V"])
        print(f"  {code}: |V| full/norho = {np.sum(np.abs(full['V'][m]))/np.sum(np.abs(norho['V'][m])):.2f}, corr(full, norho V) = {np.corrcoef(full['V'][m], norho['V'][m])[0,1]:+.3f}")
