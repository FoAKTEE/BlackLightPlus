import numpy as np, os
D = "/home/haiyangw/.claude/jobs/b085d9a7/tmp/kpolaris/diag_v"
B = "/data/haiyangw/claude/BlackLightPlus/figures/out/kpolaris/diag_v"
T = "/home/haiyangw/.claude/jobs/b085d9a7/tmp/kpolaris/bl_runs"
def bl(name, nu):
    f = f"{T}/fast_{nu}.npz" if name == "stock" else f"{D}/bl_{name}_{nu}.npz"
    z = np.load(f); return {k: np.array(z[k + "_nu"], float) for k in "IQUV"}
def blp(nu): return {k: np.load(f"{B}/blp_full_{nu}_{k}.npy") for k in "IQUV"}
def nmse(a, b, m): return np.sum((a[m] - b[m])**2) / np.sum(b[m]**2)
for nu in ("1000", "230"):
    p = blp(nu)
    print(f"==== {nu} GHz: BLP (full physics) against Blacklight variants")
    for name in ("stock", "fixm23", "split"):
        try: b = bl(name, nu)
        except FileNotFoundError: print(f"  {name}: missing"); continue
        m = np.isfinite(b["I"]) & np.isfinite(b["V"]) & np.isfinite(b["Q"]) & np.isfinite(b["U"])
        row = [f"  {name:7s}"]
        for k in "IQUV":
            sig = m & (np.abs(b[k]) > 0.1 * np.nanmax(np.abs(b[k][m])))
            same = np.mean(np.sign(p[k][sig]) == np.sign(b[k][sig]))
            row.append(f"{k}: NMSE {nmse(p[k], b[k], m):.2e} corr {np.corrcoef(p[k][m], b[k][m])[0,1]:+.3f} |sum| ratio {np.sum(np.abs(p[k][m]))/np.sum(np.abs(b[k][m])):.2f} sign {same:.2f}")
        row.append(f"net V/I {p['V'][m].sum()/p['I'][m].sum():+.4f} vs {b['V'][m].sum()/b['I'][m].sum():+.4f}; net U/I {p['U'][m].sum()/p['I'][m].sum():+.4f} vs {b['U'][m].sum()/b['I'][m].sum():+.4f}")
        print(" | ".join(row))
