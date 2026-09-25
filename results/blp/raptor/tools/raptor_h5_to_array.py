"""Reassemble a RAPTOR img_data_<n>.h5 (datasets [block][pixel] with per-pixel
alpha/beta impact parameters) into BlackLightPlus layout M[i_beta, i_alpha]
(row 0 = most negative beta), in Jy per pixel.

RAPTOR stores I in erg s^-1 cm^-2 Hz^-1 sr^-1 times 1e23 (Jy sr^-1); the pixel
solid angle is (dx R_g / D)^2. Pixels are located by rounding the stored
(alpha, beta) onto the uniform grid, which avoids the truncation in
write_uniform_camera/find_pixel (camera.c:261-262).

usage: python raptor_h5_to_array.py img_data_0.h5 FOV_M NPIX MBH_MSUN DIST_KPC [out.npz]
"""
import sys
import h5py
import numpy as np

MSUN, G, C, KPC = 1.989e33, 6.674e-8, 2.99792458e10, 3.086e21   # RAPTOR definitions.h values


def load(path, fov, npix, mbh_msun, dist_kpc, G=G, MSUN=MSUN, KPC=KPC):
    rg = G * mbh_msun * MSUN / C ** 2
    dom = (fov / npix * rg / (dist_kpc * KPC)) ** 2
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
            if key[0] in "IQUV":
                img *= dom
            out[key] = img
    return out


if __name__ == "__main__":
    path, fov, npix, mbh, dist = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    imgs = load(path, fov, npix, mbh, dist)
    for k, v in sorted(imgs.items()):
        print(f"{k:16s} sum={v.sum():+.6e}")
    if len(sys.argv) > 6:
        np.savez(sys.argv[6], **imgs)
