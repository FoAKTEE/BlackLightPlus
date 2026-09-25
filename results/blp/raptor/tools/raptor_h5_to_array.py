"""Reassemble a RAPTOR img_data_<n>.h5 (datasets [block][pixel] with per-pixel
alpha/beta impact parameters) into BlackLightPlus layout M[i_beta, i_alpha]
(row 0 = most negative beta, column 0 = most negative alpha), in Jy per pixel.

RAPTOR stores I, Q, U, V in erg s^-1 cm^-2 Hz^-1 sr^-1 times 1e23 (Jy sr^-1),
one dataset per Stokes parameter and frequency ("I2.300000e+11", ...), plus
optical depths ("tau...", "tauF..."). The pixel solid angle is
(FOV/NPIX R_g / D)^2 with R_g = G M / c^2. Pixels are located by rounding the
stored (alpha, beta) onto the uniform grid, which avoids the truncation in
write_uniform_camera/find_pixel (camera.c:261-262).

Constants: `raptor` is the upstream definitions.h set (G 6.674e-8,
M_sun 1.989e33, kpc 3.086e21); `blp` is the BlackLightPlus set of the BBH
driver (G 6.6743e-8, M_sun 1.98847e33, pc 3.0856775814913673e18). Use the set
the RAPTOR build was compiled with (definitions.h), so that the solid angle
matches the R_g of the run.

The .npz holds every dataset under its RAPTOR name, plus
  IQUV        [nfreq, 4, npix, npix]  Jy per pixel (absent Stokes = 0)
  frequencies [nfreq]                 Hz
  alpha, beta [npix]                  pixel-centre impact parameters [R_g]

usage: python raptor_h5_to_array.py IMG.h5 FOV_M NPIX MBH_MSUN DIST_KPC [OUT.npz]
                                     [--constants raptor|blp]
"""
import argparse
import re

import h5py
import numpy as np

C = 2.99792458e10
CONSTANTS = {
    "raptor": {"G": 6.674e-8, "MSUN": 1.989e33, "KPC": 3.086e21},
    "blp": {"G": 6.6743e-8, "MSUN": 1.98847e33, "KPC": 3.0856775814913673e21},
}
_KEY = re.compile(r"^(I|Q|U|V|tauF|tau)([0-9.eE+-]+)$")


def pixel_solid_angle(fov, npix, mbh_msun, dist_kpc, constants="raptor"):
    k = CONSTANTS[constants]
    rg = k["G"] * mbh_msun * k["MSUN"] / C ** 2
    return (fov / npix * rg / (dist_kpc * k["KPC"])) ** 2


def load(path, fov, npix, mbh_msun, dist_kpc, constants="raptor"):
    dom = pixel_solid_angle(fov, npix, mbh_msun, dist_kpc, constants)
    out = {}
    with h5py.File(path, "r") as f:
        a = f["alpha"][...].ravel()
        b = f["beta"][...].ravel()
        ia = np.rint((a + fov / 2) / (fov / npix) - 0.5).astype(int)
        ib = np.rint((b + fov / 2) / (fov / npix) - 0.5).astype(int)
        assert ia.min() >= 0 and ia.max() < npix and ib.min() >= 0 and ib.max() < npix
        assert len(set(zip(ia, ib))) == npix * npix, "non-uniform camera (AMR/SMR levels > 1)"
        for key in f.keys():
            if key in ("alpha", "beta"):
                continue
            img = np.zeros((npix, npix))
            img[ib, ia] = f[key][...].ravel()
            m = _KEY.match(key)
            if m and m.group(1) in "IQUV":
                img *= dom
            out[key] = img
    freqs = sorted({float(m.group(2)) for k in out if (m := _KEY.match(k))})
    iquv = np.zeros((len(freqs), 4, npix, npix))
    for k, img in out.items():
        m = _KEY.match(k)
        if m and m.group(1) in "IQUV":
            iquv[freqs.index(float(m.group(2))), "IQUV".index(m.group(1))] = img
    out["IQUV"] = iquv
    out["frequencies"] = np.array(freqs)
    centres = (np.arange(npix) + 0.5) * (fov / npix) - fov / 2
    out["alpha"] = centres
    out["beta"] = centres.copy()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path")
    ap.add_argument("fov", type=float, help="field of view [R_g]")
    ap.add_argument("npix", type=int)
    ap.add_argument("mbh", type=float, help="black-hole mass [M_sun] as in model.in")
    ap.add_argument("dist", type=float, help="distance [kpc] as in model.in")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--constants", choices=sorted(CONSTANTS), default="raptor")
    args = ap.parse_args()
    imgs = load(args.path, args.fov, args.npix, args.mbh, args.dist, args.constants)
    for k, v in sorted(imgs.items()):
        if _KEY.match(k):
            print(f"{k:16s} sum={v.sum():+.6e} finite={bool(np.isfinite(v).all())}")
    if args.out:
        np.savez(args.out, **imgs)


if __name__ == "__main__":
    main()
