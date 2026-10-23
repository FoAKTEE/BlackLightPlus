#!/usr/bin/env python3
"""Rewrite an AthenaK .athdf dump in the Athena++ layout that Blacklight's `athena` reader
parses: datasets `prim` (rho, press, vel1, vel2, vel3) and `B` (Bcc1, Bcc2, Bcc3) in
float32, press = (γ − 1) eint, float32 coordinate arrays, fixed-type attributes without the
variable-length `Header`. Usage: athdf_for_blacklight.py <in.athdf> <out.athdf> <gamma>."""
import sys

import h5py
import numpy as np

src, dst, gamma = sys.argv[1], sys.argv[2], float(sys.argv[3])
with h5py.File(src) as a, h5py.File(dst, "w", libver="earliest") as b:
    names = [n.decode() for n in a.attrs["VariableNames"]]
    uov = a["uov"][...]
    prim = np.empty_like(uov)
    prim[0] = uov[names.index("dens")]
    prim[1] = (gamma - 1) * uov[names.index("eint")]
    prim[2] = uov[names.index("velx")]
    prim[3] = uov[names.index("vely")]
    prim[4] = uov[names.index("velz")]
    for k in a.attrs:
        if k in ("Header", "DatasetNames", "VariableNames"):
            continue
        v = a.attrs[k]
        if isinstance(v, (bytes, np.bytes_)):
            b.attrs.create(k, v, dtype=f"|S{len(v)}")
        elif isinstance(v, np.ndarray) and v.dtype.kind == "f":
            b.attrs.create(k, v.astype(np.float32), dtype=np.float32)
        elif isinstance(v, np.ndarray) and v.dtype.kind == "i":
            b.attrs.create(k, v.astype(np.int32), dtype=np.int32)
        elif isinstance(v, np.floating):
            b.attrs.create(k, np.float32(v), dtype=np.float32)
        elif isinstance(v, np.integer):
            b.attrs.create(k, np.int32(v), dtype=np.int32)
        else:
            b.attrs[k] = v
    b.attrs.create("DatasetNames", np.array([b"prim", b"B"], dtype="|S21"), dtype="|S21")
    b.attrs.create("VariableNames", np.array([b"rho", b"press", b"vel1", b"vel2", b"vel3", b"Bcc1", b"Bcc2", b"Bcc3"],
                                             dtype="|S21"), dtype="|S21")
    b.create_dataset("prim", data=prim.astype(np.float32))
    b.create_dataset("B", data=a["B"][...].astype(np.float32))
    for k in ("Levels", "LogicalLocations", "x1f", "x1v", "x2f", "x2v", "x3f", "x3v"):
        d = a[k][...]
        d = d.astype(np.float32) if d.dtype.kind == "f" else d.astype(np.int32 if k == "Levels" else np.int64)
        b.create_dataset(k, data=d)
print(dst)
