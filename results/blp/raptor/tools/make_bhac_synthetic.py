"""Write a small synthetic BHAC-format (.dat) snapshot readable by RAPTOR's
model/bhac reader (CKS metric, SFC=0), plus the matching grid_cks.in.

Layout (RAPTOR model/bhac/model.c:530-848): per leaf, nwini conserved arrays
of nx^3 doubles (i fastest), then (nx+1)^3*nws staggered doubles; then the
forest flags (one int per node, level-1 blocks with i fastest); then nx[ndim]
ints, neqpar doubles, and the 40-byte tail
(nleafs, levmax, ndim, ndir, nw, nws, neqpar, it : int32; t : float64).

Plasma: gamma=4/3 torus-like Gaussian ring (R0=12, width 4) at rest in the
normal-observer frame, uniform vertical field with plasma beta 10.
"""
import sys
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "synthetic.dat"
spin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9375
L, NL, NB = 40.0, 64, 16          # half-width, cells per dim, cells per block
gam = 4.0 / 3.0
ng = NL // NB
nw, nws, neqpar = 11, 3, 4        # D S1 S2 S3 TAU B1 B2 B3 DS LFAC XI
dx = 2 * L / NL

def ks_r(x, y, z, a):
    R2 = x * x + y * y + z * z
    return np.sqrt(0.5 * (R2 - a * a + np.sqrt((R2 - a * a) ** 2 + 4 * a * a * z * z)))

def gdd_cks(x, y, z, a):
    r = ks_r(x, y, z, a)
    f = 2 * r ** 3 / (r ** 4 + a * a * z * z)
    l = np.array([np.ones_like(r), (r * x + a * y) / (r * r + a * a),
                  (r * y - a * x) / (r * r + a * a), z / r])
    g = np.einsum("i...,j...->ij...", l, l) * f
    for i in range(4):
        g[i, i] += -1.0 if i == 0 else 1.0
    return g

blocks = []
for k in range(ng):
    for j in range(ng):
        for i in range(ng):
            idx = np.arange(NB)
            xc = -L + (i * NB + idx + 0.5) * dx
            yc = -L + (j * NB + idx + 0.5) * dx
            zc = -L + (k * NB + idx + 0.5) * dx
            Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")  # i fastest
            R = np.sqrt(X * X + Y * Y)
            rho = np.exp(-((R - 12.0) ** 2 + Z ** 2) / (2 * 4.0 ** 2)) + 1e-6
            u = 0.03 * rho
            p = (gam - 1) * u
            g = gdd_cks(X, Y, Z, spin)
            # Eulerian vertical field B^z with b^2/2 = p/10 at rest (b^2 = g_zz B^z B^z)
            Bz = np.sqrt(2 * p / 10.0 / g[3, 3])
            lfac = np.ones_like(rho)
            cons = np.zeros((nw,) + rho.shape)
            cons[0] = rho * lfac                    # D
            cons[7] = Bz                            # B3
            cons[9] = lfac                          # LFAC
            cons[10] = lfac ** 2 * (rho + gam * u)  # XI
            cons[8] = rho * p / rho ** gam          # DS (entropy backup)
            cons[4] = cons[10] - p - cons[0]        # TAU (unused by RAPTOR)
            blocks.append(cons)

with open(out, "wb") as fh:
    for cons in blocks:
        for w in range(nw):
            fh.write(np.ascontiguousarray(cons[w]).astype("<f8").tobytes())
        fh.write(np.zeros((NB + 1) ** 3 * nws, "<f8").tobytes())
    fh.write(np.ones(ng ** 3, "<i4").tobytes())           # all level-1 leaves
    fh.write(np.array([NB, NB, NB], "<i4").tobytes())
    eq = np.zeros(neqpar); eq[0] = gam; eq[3] = spin
    fh.write(eq.astype("<f8").tobytes())
    fh.write(np.array([ng ** 3, 1, 3, 3, nw, nws, neqpar, 0], "<i4").tobytes())
    fh.write(np.array([0.0], "<f8").tobytes())

with open("grid_cks.in", "w") as fh:
    for n in (1, 2, 3):
        fh.write(f"nxlone{n} = {NL}\n")
    for n in (1, 2, 3):
        fh.write(f"xprobminx{n} = {-L}\n")
    for n in (1, 2, 3):
        fh.write(f"xprobmax{n} = {L}\n")
    fh.write("hslope = 1.0\n")
print(out, len(blocks), "leaves")
