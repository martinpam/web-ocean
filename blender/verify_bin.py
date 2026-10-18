#!/usr/bin/env python3
"""Verify assets/ship.bin (plain Python, no numpy).

    /usr/bin/python3 blender/verify_bin.py [path/to/ship.bin]

Prints counts, bbox, material histogram, AO / extra stats and checks that every index < vertexCount,
that the bbox in the header matches the vertices, and that normals are unit length.
"""
import os
import struct
import sys
from array import array

MAT_NAMES = ["hull", "deck", "sail", "rope", "iron", "gold", "paint", "glass", "flag", "spar", "copper"]


def half(h):
    return struct.unpack("<e", struct.pack("<H", h))[0]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(here), "assets", "ship.bin")
    data = open(path, "rb").read()
    assert data[:4] == b"SHP1", "bad magic"
    nv, ni, res = struct.unpack_from("<III", data, 4)
    bb = struct.unpack_from("<6f", data, 16)
    pad = struct.unpack_from("<6f", data, 40)
    expect = 64 + nv * 28 + ni * 4
    print("file      %s  (%d bytes, expected %d)" % (path, len(data), expect))
    assert len(data) == expect, "size mismatch"
    assert res == 0 and all(p == 0 for p in pad), "reserved fields not zero"
    assert ni % 3 == 0, "index count not a multiple of 3"
    print("vertices  %d" % nv)
    print("indices   %d  (%d triangles)" % (ni, ni // 3))
    print("bbox hdr  min (%.3f, %.3f, %.3f)  max (%.3f, %.3f, %.3f)" % bb)

    mn = [1e30] * 3
    mx = [-1e30] * 3
    hist = [0] * 256
    ao_sum = [0.0] * 256
    ao_min = [1.0] * 256
    ex_max = [0] * 256
    bad_n = 0
    off = 64
    rec = struct.Struct("<3f4b2H4B4B")
    for i in range(nv):
        x, y, z, nx, ny, nz, nw, u, v, r, g, b, a, mid, rough, extra, w = rec.unpack_from(data, off + i * 28)
        for k, c in enumerate((x, y, z)):
            if c < mn[k]:
                mn[k] = c
            if c > mx[k]:
                mx[k] = c
        ln = (nx * nx + ny * ny + nz * nz) ** 0.5 / 127.0
        if abs(ln - 1.0) > 0.03 or nw != 0:
            bad_n += 1
        hist[mid] += 1
        ao_sum[mid] += a / 255.0
        ao_min[mid] = min(ao_min[mid], a / 255.0)
        ex_max[mid] = max(ex_max[mid], extra)
    print("bbox data min (%.3f, %.3f, %.3f)  max (%.3f, %.3f, %.3f)" % (*mn, *mx))
    assert all(abs(mn[k] - bb[k]) < 1e-4 and abs(mx[k] - bb[3 + k]) < 1e-4 for k in range(3)), "bbox mismatch"

    idx = array("I")
    idx.frombytes(data[64 + nv * 28:])
    if sys.byteorder != "little":
        idx.byteswap()
    imax = max(idx)
    print("max index %d  (< vertexCount: %s)" % (imax, imax < nv))
    assert imax < nv, "index out of range"
    tri_mat = [0] * 256
    degenerate = 0
    for t in range(0, ni, 3):
        a_, b_, c_ = idx[t], idx[t + 1], idx[t + 2]
        if a_ == b_ or b_ == c_ or a_ == c_:
            degenerate += 1
        tri_mat[data[64 + a_ * 28 + 24]] += 1
    print("bad normals %d, degenerate triangles %d" % (bad_n, degenerate))
    print("\n id  material   vertices  triangles  meanAO  minAO  maxExtra")
    for m in range(256):
        if hist[m]:
            name = MAT_NAMES[m] if m < len(MAT_NAMES) else "?"
            print(" %2d  %-8s %10d %10d   %.3f  %.3f  %5d" % (m, name, hist[m], tri_mat[m], ao_sum[m] / hist[m],
                                                           ao_min[m], ex_max[m]))
    print("\nOK")


if __name__ == "__main__":
    main()
