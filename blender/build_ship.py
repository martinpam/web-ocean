"""
build_ship.py -- procedural 18th-century three-masted pirate ship for the WebGPU ocean.

Rebuild everything (geometry, AO, assets/ship.bin, blender/ship.blend):
    /opt/homebrew/bin/blender -b --factory-startup --python blender/build_ship.py
Same, plus preview renders to blender/renders/<TAG>_<view>.png:
    /opt/homebrew/bin/blender -b --factory-startup --python blender/build_ship.py -- --render final
Options after "--":
    --render TAG      render preview views with this file prefix
    --views a,b,c     subset of views (front34, side, sternlow, deck, sil)
    --samples N       Cycles samples (default 96)
    --fast            skip the AO ray casts (AO = 1), for quick geometry checks
    --check           also render a Workbench image with back-face culling (normal check)

Coordinates
    Build space (Blender): Z up, bow toward +Y, starboard +X, design waterline z = 0.
    Export space (ship.bin): Y up, bow toward -Z, starboard +X, waterline y = 0.
        export = (bx, bz, -by)   -- a proper rotation, so CCW winding is kept.
    Before export the ship is moved along Blender Y so that the centre of buoyancy
    (design draft) is at export z = 0.

Everything is deterministic (fixed seeds, no random state from Blender).
"""
import bpy
import math
import os
import struct
import sys
import time

import numpy as np
from mathutils import Vector
from mathutils import geometry as mgeo
from mathutils.bvhtree import BVHTree

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN_PATH = os.path.join(ROOT, "assets", "ship.bin")
BLEND_PATH = os.path.join(HERE, "ship.blend")
RENDER_DIR = os.path.join(HERE, "renders")

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default=None):
    if name in ARGV:
        i = ARGV.index(name)
        if i + 1 < len(ARGV) and not ARGV[i + 1].startswith("--"):
            return ARGV[i + 1]
        return True
    return default


RENDER_TAG = opt("--render")
VIEWS = opt("--views", "front34,side,sternlow,deck,sil")
SAMPLES = int(opt("--samples", 96))
FAST = bool(opt("--fast", False))
CHECK = bool(opt("--check", False))


def log(*a):
    print("[ship %6.1fs]" % (time.time() - T0), *a, flush=True)


# ----------------------------------------------------------------------------
# materials / colours (sRGB 0..1)
# ----------------------------------------------------------------------------
def hexc(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


M_HULL, M_DECK, M_SAIL, M_ROPE, M_IRON, M_GOLD, M_PAINT, M_GLASS, M_FLAG, M_SPAR, M_COPPER = range(11)
ROUGH = {0: 0.82, 1: 0.88, 2: 0.92, 3: 0.9, 4: 0.55, 5: 0.35, 6: 0.6, 7: 0.08, 8: 0.9, 9: 0.72, 10: 0.5}

HULL = hexc("#2b1c0f")
HULL_D = hexc("#24180e")
DECK = hexc("#95805f")
DECK_D = hexc("#5b4a36")
SAIL = hexc("#e2d6ba")
TAR = hexc("#231b14")
HEMP = hexc("#7a664a")
IRON = hexc("#1d1c1b")
GOLD = hexc("#b48a34")
GOLD_D = hexc("#8a6a2a")
BLACK = hexc("#131211")
HOLE = hexc("#060504")
RED = hexc("#5a140f")
GLASS = hexc("#ffc77a")
FLAGC = hexc("#0c0c0c")
BONE = hexc("#e6dfcc")
MAST = hexc("#634629")
YARD = hexc("#3b2b1c")
COPPER = hexc("#7c4b2b")
PATINA = hexc("#56705d")

RNG = np.random.default_rng(1717)

# ----------------------------------------------------------------------------
# small math helpers
# ----------------------------------------------------------------------------


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def nrm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def V3(*a):
    return np.array(a, float)


def face_normals(P, T):
    return np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])


def vertex_normals(P, T):
    fn = face_normals(P, T)
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    return nrm(N)


def orient(P, T, N):
    """Flip each triangle so its winding agrees with the given vertex normals."""
    fn = face_normals(P, T)
    s = np.einsum("ij,ij->i", fn, N[T[:, 0]] + N[T[:, 1]] + N[T[:, 2]])
    T = T.copy()
    f = s < 0
    T[f] = T[f][:, ::-1]
    return T


def grid_tris(nu, nv):
    i, j = np.meshgrid(np.arange(nu - 1), np.arange(nv - 1), indexing="ij")
    a = (i * nv + j).ravel()
    b = ((i + 1) * nv + j).ravel()
    c = ((i + 1) * nv + j + 1).ravel()
    d = (i * nv + j + 1).ravel()
    return np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


def frame_axes(x, up=(0, 0, 1)):
    """rows: x (given), y, z (close to up)."""
    x = nrm(x)
    up = np.asarray(up, float)
    if abs(np.dot(up, x)) > 0.98:
        up = np.array([0.0, 1.0, 0.0]) if abs(x[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    z = nrm(up - np.dot(up, x) * x)
    y = np.cross(z, x)
    return np.array([x, y, z])


def _hash3(ix, iy, iz, seed):
    h = (ix.astype(np.uint64) * np.uint64(73856093)) ^ (iy.astype(np.uint64) * np.uint64(19349663)) \
        ^ (iz.astype(np.uint64) * np.uint64(83492791)) ^ np.uint64((seed * 2654435761) & 0xFFFFFFFF)
    h = (h ^ (h >> np.uint64(13))) * np.uint64(1274126177)
    h = h ^ (h >> np.uint64(16))
    return (h & np.uint64(0xFFFF)).astype(float) / 65535.0


def vnoise(p, seed=0):
    p = np.asarray(p, float)
    i = np.floor(p).astype(np.int64)
    f = p - i
    u = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (u[..., 0] if dx else 1 - u[..., 0]) * (u[..., 1] if dy else 1 - u[..., 1]) * \
                    (u[..., 2] if dz else 1 - u[..., 2])
                out = out + w * _hash3(i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz, seed)
    return out


def fbm(p, octaves=4, seed=0):
    p = np.asarray(p, float)
    s, a, tot = 0.0, 1.0, 0.0
    for k in range(octaves):
        s = s + a * vnoise(p * (2.0 ** k), seed + 17 * k)
        tot += a
        a *= 0.5
    return s / tot


def hash1(*a, seed=0):
    a = [np.asarray(x, np.int64) for x in a] + [np.zeros_like(np.asarray(a[0], np.int64))] * (3 - len(a))
    return _hash3(a[0], a[1], a[2], seed)


# ----------------------------------------------------------------------------
# geometry accumulator
# ----------------------------------------------------------------------------
class Acc:
    def __init__(self):
        self.parts = []
        self.n = 0
        self.tag_tris = {}
        self.tag = "misc"
        self.cur_w = 0      # material.w for everything added (0..255)

    def add(self, P, T, N=None, UV=None, col=(1, 1, 1), mat=0, rough=None, extra=0.0, expect=None):
        P = np.asarray(P, float).reshape(-1, 3)
        T = np.asarray(T, np.int64).reshape(-1, 3)
        n = len(P)
        if N is None:
            if expect is not None:
                fn = face_normals(P, T)
                if isinstance(expect, tuple) and isinstance(expect[0], str):
                    cen = (P[T[:, 0]] + P[T[:, 1]] + P[T[:, 2]]) / 3
                    s = np.einsum("ij,ij->i", fn, cen - np.asarray(expect[1], float)).sum()
                else:
                    s = fn.sum(0) @ np.asarray(expect, float)
                if s < 0:
                    T = T[:, ::-1].copy()
            N = vertex_normals(P, T)
        else:
            N = nrm(np.asarray(N, float).reshape(-1, 3))
            T = orient(P, T, N)
        UV = np.zeros((n, 2)) if UV is None else np.asarray(UV, float).reshape(-1, 2)
        col = np.asarray(col, float)
        C = np.broadcast_to(col.reshape(-1, 3) if col.size == 3 * n else col, (n, 3)).astype(float)
        extra = np.asarray(extra, float)
        E = np.broadcast_to(extra.reshape(-1) if extra.size == n else extra, (n,)).astype(float)
        R = np.full(n, ROUGH[mat] if rough is None else rough)
        M = np.full(n, mat)
        W = np.full(n, float(self.cur_w))
        self.parts.append((P, N, UV, C, M, R, E, W, T + self.n))
        self.n += n
        self.tag_tris[self.tag] = self.tag_tris.get(self.tag, 0) + len(T)

    def arrays(self):
        out = [np.concatenate([p[k] for p in self.parts]) for k in range(9)]
        return out


def add_grid(acc, G, UV=None, N=None, col=(1, 1, 1), extra=0.0, **kw):
    nu, nv = G.shape[:2]
    T = grid_tris(nu, nv)
    col = np.asarray(col, float)
    if col.ndim == 3:
        col = col.reshape(-1, 3)
    extra = np.asarray(extra, float)
    if extra.ndim == 2:
        extra = extra.ravel()
    acc.add(G.reshape(-1, 3), T, None if N is None else N.reshape(-1, 3),
            None if UV is None else UV.reshape(-1, 2), col=col, extra=extra, **kw)


def add_tube(acc, pts, radii, sides=6, caps=(False, False), col=TAR, mat=M_ROPE, rough=None,
             extra=0.0, cap_col=None, ref=None):
    pts = np.asarray(pts, float)
    k = len(pts)
    radii = np.broadcast_to(np.asarray(radii, float), (k,)).astype(float)
    d = np.gradient(pts, axis=0)
    t = nrm(d)
    if ref is None:
        ref = np.array([0, 0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    n0 = nrm(np.cross(t[0], ref))
    ns = [n0]
    for i in range(1, k):
        v = ns[-1] - np.dot(ns[-1], t[i]) * t[i]
        ns.append(nrm(v) if np.linalg.norm(v) > 1e-8 else ns[-1])
    ns = np.array(ns)
    bs = np.cross(t, ns)
    ang = np.linspace(0, 2 * np.pi, sides + 1)
    off = np.cos(ang)[None, :, None] * ns[:, None, :] + np.sin(ang)[None, :, None] * bs[:, None, :]
    G = pts[:, None, :] + radii[:, None, None] * off
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    UV = np.stack(np.broadcast_arrays(s[:, None], ang[None, :] * radii.mean()), -1)
    add_grid(acc, G, UV=UV, N=off, col=col, mat=mat, rough=rough, extra=extra)
    for e, sgn in ((0, -1), (-1, 1)):
        if not caps[0 if e == 0 else 1]:
            continue
        ring = G[e, :-1]
        P = np.vstack([pts[e], ring])
        T = [(0, 1 + i, 1 + (i + 1) % sides) for i in range(sides)]
        N = np.tile(sgn * t[e], (len(P), 1))
        uv = np.stack([(P - pts[e]) @ ns[e], (P - pts[e]) @ bs[e]], 1)
        acc.add(P, T, N=N, UV=uv, col=col if cap_col is None else cap_col, mat=mat, rough=rough, extra=extra)


def add_box(acc, c, ax, h, col=(1, 1, 1), mat=0, rough=None, extra=0.0):
    c = np.asarray(c, float)
    ax = np.asarray(ax, float)
    h = np.asarray(h, float)
    P, N, UV, T = [], [], [], []
    for k in range(3):
        a, b = [(1, 2), (2, 0), (0, 1)][k]
        if h[b] > h[a]:
            a, b = b, a
        for s in (-1, 1):
            n = s * ax[k]
            fc = c + n * h[k]
            base = len(P)
            for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                P.append(fc + sa * h[a] * ax[a] + sb * h[b] * ax[b])
                N.append(n)
                UV.append((sa * h[a], sb * h[b]))
            T += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    acc.add(np.array(P), np.array(T), N=np.array(N), UV=np.array(UV), col=col, mat=mat, rough=rough, extra=extra)


def add_beam(acc, p0, p1, w, h, up=(0, 0, 1), **kw):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    ax = frame_axes(p1 - p0, up)
    add_box(acc, (p0 + p1) / 2, ax, (np.linalg.norm(p1 - p0) / 2, w / 2, h / 2), **kw)


def add_slab(acc, poly, nvec, thick, col=(1, 1, 1), mat=0, rough=None, extra=0.0, side_col=None):
    poly = np.asarray(poly, float)
    n = nrm(np.asarray(nvec, float))
    k = len(poly)
    tris = np.array(mgeo.tessellate_polygon([[Vector(p) for p in poly]]))
    e1 = nrm(poly[1] - poly[0])
    e1 = nrm(e1 - np.dot(e1, n) * n)
    e2 = np.cross(n, e1)
    uvp = np.stack([(poly - poly[0]) @ e1, (poly - poly[0]) @ e2], 1)
    if abs(e2 @ np.array([0, 0, 1.0])) < abs(e1 @ np.array([0, 0, 1.0])):
        pass
    acc.add(poly + n * thick / 2, tris, N=np.tile(n, (k, 1)), UV=uvp, col=col, mat=mat, rough=rough, extra=extra)
    acc.add(poly - n * thick / 2, tris, N=np.tile(-n, (k, 1)), UV=uvp, col=col, mat=mat, rough=rough, extra=extra)
    area = sum(np.cross(poly[i], poly[(i + 1) % k]) @ n for i in range(k))
    P, N, T, UV = [], [], [], []
    run = 0.0
    for i in range(k):
        a = poly[i]
        b = poly[(i + 1) % k]
        en = nrm(np.cross(b - a, n)) * (1 if area > 0 else -1)
        base = len(P)
        L = np.linalg.norm(b - a)
        P += [a - n * thick / 2, b - n * thick / 2, b + n * thick / 2, a + n * thick / 2]
        N += [en] * 4
        UV += [(run, 0), (run + L, 0), (run + L, thick), (run, thick)]
        run += L
        T += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    acc.add(np.array(P), np.array(T), N=np.array(N), UV=np.array(UV),
            col=col if side_col is None else side_col, mat=mat, rough=rough, extra=extra)


def add_sweep(acc, path, profile, col=(1, 1, 1), mat=0, rough=None, side_hint=None, caps=True, extra=0.0):
    """Sweep a closed convex 2D profile (a = side, b = up) along a path; flat per profile edge."""
    path = np.asarray(path, float)
    prof = np.asarray(profile, float)
    k = len(path)
    t = nrm(np.gradient(path, axis=0))
    if side_hint is None:
        side = np.cross(t, np.array([0, 0, 1.0]))
        bad = np.linalg.norm(side, axis=1) < 1e-6
        side[bad] = np.array([1.0, 0, 0])
        side = nrm(side)
    else:
        h = np.asarray(side_hint, float)
        side = nrm(h[None, :] - (t @ h)[:, None] * t)
    up = np.cross(side, t)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    cen = prof.mean(0)
    m = len(prof)
    run = 0.0
    for i in range(m):
        pa = prof[i]
        pb = prof[(i + 1) % m]
        d = pb - pa
        n2 = nrm(np.array([d[1], -d[0]]))
        if n2 @ ((pa + pb) / 2 - cen) < 0:
            n2 = -n2
        G = np.stack([path + pa[0] * side + pa[1] * up, path + pb[0] * side + pb[1] * up], 1)
        N = np.repeat((n2[0] * side + n2[1] * up)[:, None, :], 2, 1)
        L = np.linalg.norm(d)
        UV = np.stack(np.broadcast_arrays(s[:, None], np.array([run, run + L])[None, :]), -1)
        run += L
        add_grid(acc, G, UV=UV, N=N, col=col, mat=mat, rough=rough, extra=extra)
    if caps:
        for e, sgn in ((0, -1), (k - 1, 1)):
            ring = np.array([path[e] + p[0] * side[e] + p[1] * up[e] for p in prof])
            P = np.vstack([path[e] + cen[0] * side[e] + cen[1] * up[e], ring])
            T = [(0, 1 + i, 1 + (i + 1) % m) for i in range(m)]
            acc.add(P, T, N=np.tile(sgn * t[e], (len(P), 1)), col=col, mat=mat, rough=rough, extra=extra)


def rect(w, h):
    return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]


def add_cloth(acc, P, T, UV, col, extra, mat, front_dir, thick=0.014):
    """Double-sided cloth: front sheet (normals toward front_dir) + back sheet (reversed)."""
    P = np.asarray(P, float)
    T = np.asarray(T, np.int64)
    fn = face_normals(P, T)
    if fn.sum(0) @ np.asarray(front_dir, float) < 0:
        T = T[:, ::-1].copy()
    N = vertex_normals(P, T)
    acc.add(P + N * thick / 2, T, N=N, UV=UV, col=col, mat=mat, extra=extra)
    acc.add(P - N * thick / 2, T[:, ::-1], N=-N, UV=UV, col=col, mat=mat, extra=extra)


def rope(acc, a, b, r=0.03, col=TAR, sides=5, sag=0.0, n=8, mat=M_ROPE):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if sag > 0:
        t = np.linspace(0, 1, n)
        pts = a + (b - a) * t[:, None]
        pts[:, 2] -= sag * 4 * t * (1 - t)
    else:
        pts = np.array([a, b])
    add_tube(acc, pts, r, sides, col=col, mat=mat)


# ----------------------------------------------------------------------------
# hull definition (Blender space, bow +Y, waterline z=0)
# ----------------------------------------------------------------------------
ZK = -3.15      # hull bottom at keel rabbet (keel timber goes to -3.5)
ZC = 0.18       # boot-top: copper below
ZM = 0.85       # height of max breadth
ZTOPMAX = 6.0
BMAX = 4.0
TH = 0.17       # tumblehome fraction at the top
YF0 = 1.5       # start of entrance
YA0 = -1.0      # start of run

Y_QD = -6.6     # quarterdeck break
Y_FC = 10.4     # forecastle break
QD_H = 1.35
FC_H = 1.05
BULW = 1.2
BW_T = 0.2      # bulwark thickness


def y_stem(z):
    z = np.asarray(z, float)
    up = 14.6 + 0.24 * z
    dz = np.clip(-1.0 - z, 0, None)
    r = 2.4
    low = (14.6 - 0.24) - (r - np.sqrt(np.maximum(r * r - dz * dz, 0)))
    return np.where(z >= -1.0, up, low)


def y_post(z):
    z = np.asarray(z, float)
    return -12.9 - 0.16 * (z - ZK) - 0.26 * np.clip(z - 0.9, 0, None)


def tr_ratio(z):
    z = np.asarray(z, float)
    return 0.6 * smoothstep(-1.4, 1.5, z) + 0.05 * np.clip(z - 1.5, 0, None) / 4.0


def hb(y, z):
    """Half breadth of the outer hull shell."""
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    e = smoothstep(7.0, 13.5, np.abs(y - 0.5))
    p = 2.5 - 0.8 * e
    q = 0.7 + 0.22 * e
    below = np.maximum(1.0 - np.clip((ZM - z) / (ZM - ZK), 0, 1) ** p, 0) ** q
    above = 1.0 - TH * np.clip((z - ZM) / (ZTOPMAX - ZM), 0, None) ** 1.5
    sec = np.where(z < ZM, below, above)
    ys = y_stem(z)
    yp = y_post(z)
    s = np.clip((y - YF0) / (ys - YF0), 0, 1)
    b = lerp(1.0, 0.72, smoothstep(-2.5, 1.5, z))
    Lf = np.maximum(1 - s ** 2, 0) ** b
    sa = np.clip((YA0 - y) / (YA0 - yp), 0, 1)
    k = lerp(1.6, 2.3, smoothstep(-2.5, 1.5, z))
    qq = lerp(1.6, 1.0, smoothstep(-2.5, 0.5, z))
    tr = tr_ratio(z)
    La = np.maximum(1 - (1 - tr) * sa ** k, 0) ** qq
    out = BMAX * sec * Lf * La
    inside = (y <= ys + 1e-9) & (y >= yp - 1e-9)
    return np.where(inside, out, 0.0)


def deck_main(y):
    return 1.75 + 0.85 * ((np.asarray(y, float) - 1.0) / 15.0) ** 2


def deck_level(y):
    y = np.asarray(y, float)
    return deck_main(y) + QD_H * (y < Y_QD) + FC_H * (y > Y_FC)


def z_top(y):
    y = np.asarray(y, float)
    return (deck_main(y) + BULW + QD_H * smoothstep(Y_QD + 0.5, Y_QD - 0.7, y)
            + FC_H * smoothstep(Y_FC - 0.5, Y_FC + 0.7, y) + 0.4 * smoothstep(-12.0, -15.8, y))


def hull_pn(y, z, side):
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    e = 0.01
    x = hb(y, z)
    fy = (hb(y + e, z) - hb(y - e, z)) / (2 * e)
    fz = (hb(y, z + e) - hb(y, z - e)) / (2 * e)
    n = nrm(np.stack([np.full_like(x, side), -fy, -fz], -1))
    p = np.stack([side * x, y, z], -1)
    return p, n


def hull_frame(y, z, side):
    p, n = hull_pn(y, z, side)
    up = nrm(np.array([0, 0, 1.0]) - n[2] * n)
    t = np.cross(up, n)
    return p, n, t, up


def transom_y(xn, z):
    return y_post(z) - 0.15 * (1 - np.asarray(xn) ** 2) * smoothstep(-1.0, 1.0, z)


def transom_pt(xn, z, off=0.0):
    xn = np.asarray(xn, float)
    z = np.asarray(z, float)
    yp = y_post(z)
    hE = hb(yp, z)
    y = transom_y(xn, z)
    d = 0.01
    dydz = (y_post(z + d) - y_post(z - d)) / (2 * d)
    xn_b, z_b = np.broadcast_arrays(xn, z)
    n = nrm(np.stack([np.zeros_like(xn_b), -np.ones_like(xn_b), np.broadcast_to(dydz, xn_b.shape)], -1))
    p = np.stack([xn_b * np.broadcast_to(hE, xn_b.shape), np.broadcast_to(y, xn_b.shape), z_b], -1)
    return p + n * off, n


# ----------------------------------------------------------------------------
# HULL
# ----------------------------------------------------------------------------
def build_hull(acc):
    acc.tag = "hull"
    NU = 150
    t = np.linspace(0, 1, NU)
    u = 0.55 * (0.5 - 0.5 * np.cos(np.pi * t)) + 0.45 * t
    zt = np.full(NU, 4.0)
    for _ in range(10):
        yy = y_post(zt) + u * (y_stem(zt) - y_post(zt))
        zt = z_top(yy)
    NB, NA = 24, 34
    tb = np.linspace(0, 1, NB)
    zb = ZK + (ZC - ZK) * tb ** 1.25
    ta = np.linspace(0, 1, NA)
    Z = np.concatenate([np.tile(zb, (NU, 1)), ZC + ta[None, 1:] * (zt[:, None] - ZC)], axis=1)
    Y = y_post(Z) + u[:, None] * (y_stem(Z) - y_post(Z))
    X = hb(Y, Z)
    nrow = Z.shape[1]
    jc = NB - 1
    for side in (1, -1):
        G = np.stack([side * X, Y, Z], -1)
        T = grid_tris(NU, nrow)
        Pf = G.reshape(-1, 3)
        if face_normals(Pf, T).sum(0)[0] * side < 0:
            T = T[:, ::-1]
        N = vertex_normals(Pf, T).reshape(NU, nrow, 3)
        gir = np.concatenate([np.zeros((NU, 1)), np.cumsum(np.linalg.norm(np.diff(G, axis=1), axis=2), 1)], 1)
        UV = np.stack([Y, gir], -1)
        # colours
        stain = 1 - 0.28 * np.exp(-((Z - 0.3) / 0.35) ** 2)
        var = 0.93 + 0.07 * np.sin(Y * 0.9 + Z * 2.7) * np.sin(Y * 2.3 + 1.0)
        cw = HULL[None, None, :] * (stain * var)[..., None]
        pat = np.clip(0.12 + 0.3 * np.exp(-((Z + 0.1) / 0.7) ** 2) + 0.1 * np.sin(Y * 1.7) * np.sin(Z * 3.1), 0, 1)
        cc = COPPER[None, None, :] * (1 - pat[..., None]) + PATINA[None, None, :] * pat[..., None]
        sl = slice(0, jc + 1)
        add_grid(acc, G[:, sl], UV=UV[:, sl], N=N[:, sl], col=cc[:, sl], mat=M_COPPER)
        sl = slice(jc, nrow)
        add_grid(acc, G[:, sl], UV=UV[:, sl], N=N[:, sl], col=cw[:, sl], mat=M_HULL)

    # transom (closes the stern); its side edges are the hull's u=0 column
    zr = Z[0]
    hr = X[0]
    nz = np.nonzero(hr > 1e-4)[0]
    j0 = max(int(nz[0]) - 1, 0)
    zr, hr = zr[j0:], hr[j0:]
    NX = 25
    xn = np.linspace(-1, 1, NX)
    G = np.stack(np.broadcast_arrays(xn[None, :] * hr[:, None], transom_y(xn[None, :], zr[:, None]),
                                     zr[:, None]), -1)  # (rows, NX, 3)
    G = G.transpose(1, 0, 2)  # (NX, rows)
    T = grid_tris(NX, len(zr))
    Pf = G.reshape(-1, 3)
    if face_normals(Pf, T).sum(0)[1] > 0:
        T = T[:, ::-1]
    N = vertex_normals(Pf, T).reshape(G.shape)
    UV = np.stack([G[..., 0], G[..., 2]], -1)
    jc2 = int(np.argmin(np.abs(zr - ZC)))
    add_grid(acc, G[:, :jc2 + 1], UV=UV[:, :jc2 + 1], N=N[:, :jc2 + 1], col=COPPER * 0.9, mat=M_COPPER)
    add_grid(acc, G[:, jc2:], UV=UV[:, jc2:], N=N[:, jc2:], col=HULL * 0.95, mat=M_HULL)
    return dict(zt_stern=float(zt[0]), zt=zt, u=u)


def build_keel_stem(acc):
    acc.tag = "hull"
    # keel
    y0 = float(y_post(ZK)) - 0.15
    y1 = float(y_stem(ZK)) + 0.05
    add_box(acc, ((0, (y0 + y1) / 2, (-3.5 + ZK + 0.05) / 2)), np.eye(3),
            (0.18, (y1 - y0) / 2, (ZK + 0.05 + 3.5) / 2), col=COPPER * 0.85, mat=M_COPPER)
    # stem post: sweep along stem profile
    zz = np.linspace(-3.45, 0.0, 14)
    zz2 = np.linspace(0.0, float(z_top(15.5)) - 0.05, 12)
    for zs, mat, col in ((zz, M_COPPER, COPPER * 0.85), (zz2, M_HULL, HULL_D)):
        path = np.stack([np.zeros_like(zs), y_stem(zs) + 0.06, zs], 1)
        add_sweep(acc, path, rect(0.26, 0.28), col=col, mat=mat, side_hint=(1, 0, 0))
    # sternpost
    for (za, zb_), mat, col in (((-3.5, 0.0), M_COPPER, COPPER * 0.85), ((0.0, 1.6), M_HULL, HULL_D)):
        zs = np.linspace(za, zb_, 6)
        path = np.stack([np.zeros_like(zs), y_post(zs) - 0.1, zs], 1)
        add_sweep(acc, path, rect(0.32, 0.26), col=col, mat=mat, side_hint=(1, 0, 0))
    # rudder (split at boot-top)
    def rud(z):
        return float(y_post(z)) - 0.24
    def chord(z):
        return float(lerp(1.4, 0.45, np.clip((z + 3.4) / 5.4, 0, 1) ** 1.3))
    for za, zb_, mat, col in ((-3.4, ZC, M_COPPER, COPPER * 0.85), (ZC, 2.0, M_HULL, HULL_D)):
        zs = np.linspace(za, zb_, 6)
        lead = [(0, rud(z), z) for z in zs]
        trail = [(0, rud(z) - chord(z), z) for z in zs[::-1]]
        add_slab(acc, lead + trail, (1, 0, 0), 0.26, col=col, mat=mat)
    # knee of the head / cutwater, head rails, gilded trail lines and a figurehead
    acc.tag = "bow"
    zk = np.linspace(0.2, 4.1, 14)
    fwd = 0.18 + 1.5 * ((zk - 0.2) / 3.9) ** 1.7
    front = [(0, float(y_stem(z)) + f, z) for z, f in zip(zk, fwd)]
    ys41 = float(y_stem(4.1))
    top = [(0, ys41 + 1.45, 4.32), (0, ys41 + 0.95, 4.45), (0, ys41 + 0.4, 4.55), (0, float(y_stem(4.6)) + 0.05, 4.6)]
    back = [(0, float(y_stem(z)) + 0.02, z) for z in np.linspace(4.5, 0.25, 10)]
    add_slab(acc, front + top + back, (1, 0, 0), 0.24, col=HULL_D, mat=M_HULL)
    for s in (1, -1):
        for (z0, z1, inset) in ((2.3, 3.95, 0.3), (2.9, 3.6, 0.6)):
            zz = np.linspace(z0, z1, 10)
            ff = 0.18 + 1.5 * ((zz - 0.2) / 3.9) ** 1.7 - inset
            pts = np.stack([np.full(10, s * 0.125), y_stem(zz) + np.maximum(ff, 0.12), zz], 1)
            add_tube(acc, pts, 0.03, sides=5, col=GOLD, mat=M_GOLD, caps=(True, True))
        # head rails: from the bow side sweeping forward/down to the knee head
        for (zs, ze, ye) in ((4.35, 4.05, 1.35), (3.75, 3.45, 1.15)):
            y0 = 13.6
            t = np.linspace(0, 1, 12)
            yy = lerp(y0, ys41 + ye, t)
            zz = lerp(zs, ze, t) + 0.25 * np.sin(np.pi * t)
            xh = np.array([float(hb(y, z)) for y, z in zip(np.minimum(yy, float(y_stem(4.0)) - 0.01), zz)])
            xx = lerp(xh[0] + 0.05, 0.15, smoothstep(0.0, 1.0, t))
            xx = np.maximum(xx, np.minimum(xh + 0.06, xh[0] + 0.05))
            xx[-3:] = np.linspace(xx[-4], 0.15, 4)[1:]
            add_tube(acc, np.stack([s * xx, yy, zz], 1), 0.065, sides=6, col=BLACK, mat=M_PAINT, caps=(True, True))
    # figurehead (stylised, gilded) leaning forward under the bowsprit
    base = V3(0, ys41 + 1.42, 3.95)
    lean = nrm(V3(0, 0.75, 1.0))
    tt = np.linspace(0, 1.5, 15)
    rr = 1.25 * np.interp(tt, [0, 0.15, 0.4, 0.7, 0.9, 1.02, 1.1, 1.22, 1.35, 1.45, 1.5],
                          [0.12, 0.2, 0.24, 0.2, 0.15, 0.08, 0.1, 0.15, 0.14, 0.08, 0.01])
    add_tube(acc, [base + lean * t * 1.2 for t in tt], rr, sides=12, col=GOLD, mat=M_GOLD, caps=(True, False))
    for s in (1, -1):
        sh = base + lean * 0.9 + V3(s * 0.2, 0, 0)
        add_tube(acc, [sh, sh + V3(s * 0.12, 0.25, -0.15), sh + V3(s * 0.08, 0.45, -0.38)], [0.07, 0.06, 0.045],
                 sides=6, col=GOLD_D, mat=M_GOLD, caps=(True, True))


def build_deck_and_bulwarks(acc, info):
    acc.tag = "deck"
    z_qd = float(deck_level(-14.5))
    y_aft = float(y_post(z_qd)) + BW_T
    ys = np.linspace(12, 16.5, 900)
    w = hb(ys, deck_level(ys)) - BW_T
    y_bow = float(ys[np.argmax(w < 0.14)])
    info["y_bow_deck"] = y_bow
    eps = 1e-4
    patches = [(y_aft, Y_QD - eps, 24), (Y_QD + eps, Y_FC - eps, 46), (Y_FC + eps, y_bow, 18)]
    NC = 17
    xn = np.sin(np.linspace(-np.pi / 2, np.pi / 2, NC))
    for (a, b, n) in patches:
        yv = np.linspace(a, b, n)
        zd = deck_level(yv)
        xin = np.maximum(hb(yv, zd) - BW_T, 0.02)
        Xg = xin[:, None] * xn[None, :]
        Zg = zd[:, None] + 0.09 * (1 - xn[None, :] ** 2)
        G = np.stack(np.broadcast_arrays(Xg, yv[:, None], Zg), -1)
        var = 0.95 + 0.05 * np.sin(Xg * 4.1 + yv[:, None] * 0.37)
        add_grid(acc, G, UV=np.stack([G[..., 1], G[..., 0]], -1), col=DECK[None, None, :] * var[..., None],
                 mat=M_DECK, expect=(0, 0, 1))
        # inner bulwarks (painted red inside)
        NR = 7
        tr_ = np.linspace(0, 1, NR)
        ztp = z_top(yv) - 0.03
        Zb = zd[:, None] + tr_[None, :] * (ztp - zd)[:, None]
        for side in (1, -1):
            Xb = side * np.maximum(hb(yv[:, None], Zb) - BW_T, 0.02)
            G = np.stack(np.broadcast_arrays(Xb, yv[:, None], Zb), -1)
            add_grid(acc, G, UV=np.stack([G[..., 1], G[..., 2]], -1), col=RED * 0.9, mat=M_PAINT,
                     expect=(-side, 0, 0))
    # bow closing wall
    zb_ = float(deck_level(y_bow))
    zz = np.linspace(zb_, float(z_top(y_bow)) - 0.03, 5)
    xi = np.maximum(hb(y_bow, zz) - BW_T, 0.02)
    G = np.stack([np.stack([-xi, np.full(5, y_bow), zz], 1), np.stack([xi, np.full(5, y_bow), zz], 1)], 0)
    add_grid(acc, G, col=RED * 0.9, mat=M_PAINT, expect=(0, -1, 0))
    # inner transom wall (straight), from quarterdeck up to the top
    zt0 = info["zt_stern"]
    zz = np.linspace(z_qd, zt0 - 0.03, 6)
    xin = np.maximum(hb(y_post(zz), zz) - BW_T, 0.02)
    xr = np.linspace(-1, 1, 15)
    G = np.stack(np.broadcast_arrays(xr[:, None] * xin[None, :], y_post(zz)[None, :] + BW_T, zz[None, :]), -1)
    add_grid(acc, G, UV=np.stack([G[..., 0], G[..., 2]], -1), col=RED * 0.9, mat=M_PAINT, expect=(0, 1, 0))
    # break bulkheads
    acc.tag = "deckhouse"
    for yb, za, zb2, face in ((Y_QD, float(deck_main(Y_QD)), float(deck_main(Y_QD)) + QD_H, 1),
                              (Y_FC, float(deck_main(Y_FC)), float(deck_main(Y_FC)) + FC_H, -1)):
        zz = np.linspace(za, zb2, 5)
        xin = np.maximum(hb(yb, zz) - BW_T, 0.02)
        G = np.stack(np.broadcast_arrays(xr[:, None] * xin[None, :], yb, zz[None, :]), -1)
        add_grid(acc, G, UV=np.stack([G[..., 0], G[..., 2]], -1), col=HULL * 1.15, mat=M_HULL, expect=(0, face, 0))
        # doors and small windows
        for dx in (-1.1, 1.1):
            add_box(acc, (dx, yb + face * 0.04, za + 0.6), np.eye(3), (0.38, 0.04, 0.6), col=HULL_D * 0.8, mat=M_HULL)
            add_box(acc, (dx, yb + face * 0.06, za + 0.6), np.eye(3), (0.44, 0.02, 0.66), col=BLACK, mat=M_PAINT)
        for dx in (-2.4, 0.0, 2.4):
            if abs(dx) < 1e-6 and face < 0:
                continue
            add_box(acc, (dx, yb + face * 0.05, za + 0.85), np.eye(3), (0.22, 0.03, 0.18), col=GLASS * 0.85,
                    mat=M_GLASS, extra=90)
    # cap rails along both sides
    acc.tag = "rails"
    yy = np.linspace(-15.0, 16.0, 2000)
    zc = z_top(yy)
    yv0 = yy[np.argmax(yy >= y_post(zc))]
    w = hb(yy, z_top(yy))
    yv1 = yy[(yy > 10) & (w > 0.12)].max()
    ys_ = np.linspace(yv0, yv1, 180)
    zc = z_top(ys_)
    for side in (1, -1):
        path = np.stack([side * np.maximum(hb(ys_, zc) - 0.09, 0.0), ys_, zc + 0.03], 1)
        add_sweep(acc, path, rect(0.34, 0.12), col=BLACK, mat=M_PAINT)
    # taffrail cap across the transom (slab)
    xs = np.linspace(-1, 1, 21)
    hE = float(hb(y_post(zt0), zt0))
    outer = np.stack([xs * (hE + 0.06), transom_y(xs, zt0) - 0.08, np.full_like(xs, zt0 + 0.03)], 1)
    inner = np.stack([xs[::-1] * max(hE - 0.25, 0.1), np.full_like(xs, float(y_post(zt0)) + 0.26),
                      np.full_like(xs, zt0 + 0.03)], 1)
    add_slab(acc, np.vstack([outer, inner]), (0, 0, 1), 0.12, col=BLACK, mat=M_PAINT)
    info["hE_top"] = hE


def build_hull_bands(acc):
    """Wales (black) and the painted gun-port strake (red), as raised strips on the hull."""
    acc.tag = "hull_trim"
    dm = deck_main
    BIG = (-99.0, 99.0)
    bands = [
        (lambda y: dm(y) - 1.25, lambda y: dm(y) - 0.97, 0.07, BLACK, M_PAINT, BIG),
        (lambda y: dm(y) - 0.46, lambda y: dm(y) - 0.08, 0.085, BLACK, M_PAINT, BIG),
        (lambda y: dm(y) + 0.27, lambda y: dm(y) + 1.03, 0.03, RED, M_PAINT, BIG),
        (lambda y: dm(y) + 1.05, lambda y: dm(y) + 1.15, 0.06, BLACK, M_PAINT, BIG),
        (lambda y: dm(y) + QD_H + 0.95, lambda y: dm(y) + QD_H + 1.03, 0.04, GOLD, M_GOLD, (-99.0, Y_QD - 0.6)),
        (lambda y: dm(y) + QD_H + 0.32, lambda y: dm(y) + QD_H + 0.88, 0.025, RED, M_PAINT, (-99.0, Y_QD - 0.7)),
        (lambda y: dm(y) + FC_H + 0.32, lambda y: dm(y) + FC_H + 0.85, 0.025, RED, M_PAINT, (Y_FC + 0.75, 99.0)),
        (lambda y: dm(y) + FC_H + 0.9, lambda y: dm(y) + FC_H + 0.98, 0.04, GOLD, M_GOLD, (Y_FC + 0.7, 99.0)),
    ]
    for bi, (flo, fhi, th, col, mat, yr) in enumerate(bands):
        ys = np.linspace(-16.5, 16.5, 3000)
        zlo, zhi = flo(ys), fhi(ys)
        ok = (ys > np.maximum(y_post(zlo), y_post(zhi)) + 0.03) & (hb(ys, zlo) > 0.18) & (hb(ys, zhi) > 0.18)
        ok &= (ys > yr[0]) & (ys < yr[1])
        if not ok.any():
            continue
        ya, yb = ys[ok].min(), ys[ok].max()
        n = 190 if bi < 4 else 80
        yv = np.linspace(ya, yb, n)
        zlo, zhi = flo(yv), fhi(yv)
        for side in (1, -1):
            rows = np.linspace(0, 1, 4)
            Zr = zlo[:, None] + rows[None, :] * (zhi - zlo)[:, None]
            p, nn = hull_pn(np.repeat(yv[:, None], 4, 1), Zr, side)
            outer = p + nn * th
            uvo = np.stack([np.repeat(yv[:, None], 4, 1), Zr], -1)
            add_grid(acc, outer, UV=uvo, col=col, mat=mat, expect=(side, 0, 0))
            for r, ez in ((0, -1), (3, 1)):
                G = np.stack([p[:, r] - nn[:, r] * 0.03, outer[:, r]], 1)
                add_grid(acc, G, col=col, mat=mat, expect=(0, 0, ez))
            # end caps
            for e, ey in ((0, -1), (-1, 1)):
                quad = np.array([p[e, 0] - nn[e, 0] * 0.03, outer[e, 0], outer[e, 3], p[e, 3] - nn[e, 3] * 0.03])
                tvec = V3(0, ey, 0)
                acc.add(quad, [(0, 1, 2), (0, 2, 3)], N=np.tile(tvec, (4, 1)), col=col, mat=mat)


# ----------------------------------------------------------------------------
# gunports, cannons, channels
# ----------------------------------------------------------------------------
PORTS_Y = [-5.3, -3.2, 2.2, 4.2, 6.2]
# (y, deck height, scale)
PORTS = [(y, None, 1.0) for y in PORTS_Y] + [(-7.9, QD_H, 0.8)]


def build_guns(acc):
    acc.tag = "guns"
    for yp, dh, sc in PORTS:
        dmv = float(deck_main(yp)) + (dh or 0.0)
        zc = dmv + 0.71 * sc
        for side in (1, -1):
            p, n, t, up = hull_frame(yp, zc, side)
            ax = np.array([t, up, n])
            add_box(acc, p + n * 0.045, ax, (0.30 * sc, 0.28 * sc, 0.012), col=HOLE, mat=M_PAINT)
            for s in (-1, 1):
                add_box(acc, p + up * s * 0.325 * sc + n * 0.05, ax, (0.38 * sc, 0.045, 0.028), col=BLACK, mat=M_PAINT)
                add_box(acc, p + t * s * 0.345 * sc + n * 0.05, ax, (0.045, 0.28 * sc, 0.028), col=BLACK, mat=M_PAINT)
            # lid hinged at the top, swung up and out
            alpha = math.radians(102)
            d = -up * math.cos(alpha) + n * math.sin(alpha)
            hinge = p + up * 0.36 * sc + n * 0.07
            add_box(acc, hinge + d * 0.31 * sc, frame_axes(t, np.cross(t, d)) if False else np.array([t, d, nrm(np.cross(t, d))]),
                    (0.33 * sc, 0.31 * sc, 0.04), col=RED, mat=M_PAINT)
            # cannon barrel
            out = nrm(V3(n[0], n[1], 0))
            tip = p + out * 0.6
            L = 2.15
            prof_d = np.array([0.0, 0.04, 0.12, 0.2, 0.7, 1.3, 1.95, 2.0, 2.08, 2.15])
            prof_r = np.array([0.07, 0.15, 0.185, 0.18, 0.165, 0.15, 0.135, 0.155, 0.155, 0.14]) * sc
            prof_d = prof_d * (0.85 if sc < 1 else 1.0)
            L = prof_d[-1]
            pts = tip - out * L + out[None, :] * prof_d[:, None]
            add_tube(acc, pts, prof_r, sides=10, caps=(True, True), col=IRON, mat=M_IRON, cap_col=IRON)
            add_tube(acc, [tip + out * 0.002, tip + out * 0.004], 0.075, sides=10, caps=(False, True),
                     col=HOLE, mat=M_IRON, cap_col=HOLE)
            # inner port hole on the bulwark
            pin = V3(side * (float(hb(yp, zc)) - BW_T - 0.012), yp, zc)
            add_box(acc, pin, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float), (0.30 * sc, 0.28 * sc, 0.01),
                    col=HOLE, mat=M_PAINT)
            # carriage + wheels
            dz = dmv
            cc = tip - out * 1.75 * sc
            cax = frame_axes(out)
            add_box(acc, V3(cc[0], cc[1], dz + 0.3 * sc), cax, (0.62 * sc, 0.32 * sc, 0.24 * sc), col=RED * 0.8, mat=M_PAINT)
            for a in (-0.42, 0.42):
                for b in (-0.36, 0.36):
                    wc = V3(cc[0], cc[1], dz + 0.15 * sc) + (cax[0] * a + cax[1] * b) * sc
                    add_tube(acc, [wc - cax[1] * 0.05, wc + cax[1] * 0.05], 0.15 * sc, sides=10, caps=(True, True),
                             col=HULL_D, mat=M_HULL)


def channel(acc, ya, yb, zch, side):
    ys = np.linspace(ya, yb, 8)
    inner = np.stack([side * (hb(ys, zch) - 0.08), ys, np.full(8, zch)], 1)
    outer = np.stack([side * (hb(ys, zch) + 0.6), ys, np.full(8, zch)], 1)[::-1]
    add_slab(acc, np.vstack([inner, outer]), (0, 0, 1), 0.11, col=BLACK, mat=M_PAINT)


# ----------------------------------------------------------------------------
# stern decoration
# ----------------------------------------------------------------------------
def build_stern(acc, info):
    acc.tag = "stern"
    zt0 = info["zt_stern"]
    lanterns = []

    def band(z, h, off, col, mat, x0=-1.0, x1=1.0):
        xs = np.linspace(x0, x1, 31)
        p, n = transom_pt(xs, np.full_like(xs, z), off)
        add_sweep(acc, p, rect(h, 0.05) if False else [(-0.03, -h / 2), (0.03, -h / 2), (0.03, h / 2), (-0.03, h / 2)],
                  col=col, mat=mat, side_hint=None, caps=True)

    # horizontal gold bands across transom
    for z, h in ((1.35, 0.12), (2.62, 0.14), (4.0, 0.12), (zt0 - 0.12, 0.12)):
        xs = np.linspace(-1, 1, 31)
        p, n = transom_pt(xs, np.full_like(xs, z), 0.02)
        # sweep with profile in (side=horizontal normal, up)
        path = p
        prof = [(-0.04, -h / 2), (0.04, -h / 2), (0.04, h / 2), (-0.04, h / 2)]
        add_sweep(acc, path, prof, col=GOLD, mat=M_GOLD)
    # name board
    xs = np.linspace(-0.55, 0.55, 2)
    zb0, zb1 = 4.3, 4.68
    corners = [transom_pt(-0.5, zb0, 0.03)[0], transom_pt(0.5, zb0, 0.03)[0],
               transom_pt(0.5, zb1, 0.03)[0], transom_pt(-0.5, zb1, 0.03)[0]]
    n0 = transom_pt(0.0, 4.5)[1]
    add_slab(acc, corners, n0, 0.04, col=BLACK, mat=M_PAINT)
    for z in (zb0 - 0.03, zb1 + 0.03):
        a = transom_pt(-0.53, z, 0.05)[0]
        b = transom_pt(0.53, z, 0.05)[0]
        add_beam(acc, a, b, 0.04, 0.05, col=GOLD, mat=M_GOLD)
    # stern windows with gold frames and pilasters
    zw0, zw1 = 3.15, 3.85
    zmid = (zw0 + zw1) / 2
    hE = float(hb(y_post(zmid), zmid))
    for xc in (-1.6, -0.8, 0.0, 0.8, 1.6):
        xa, xb = (xc - 0.27) / hE, (xc + 0.27) / hE
        cs = [transom_pt(xa, zw0, 0.03)[0], transom_pt(xb, zw0, 0.03)[0],
              transom_pt(xb, zw1, 0.03)[0], transom_pt(xa, zw1, 0.03)[0]]
        nn = transom_pt(xc / hE, zmid)[1]
        add_slab(acc, cs, nn, 0.03, col=GLASS, mat=M_GLASS, extra=150)
        # frame
        fa = np.array([nrm(cs[1] - cs[0]), nrm(cs[3] - cs[0]), nn])
        c = (cs[0] + cs[2]) / 2 + nn * 0.03
        for s in (-1, 1):
            add_box(acc, c + fa[1] * s * 0.39, fa, (0.33, 0.045, 0.03), col=GOLD, mat=M_GOLD)
            add_box(acc, c + fa[0] * s * 0.3, fa, (0.04, 0.35, 0.03), col=GOLD, mat=M_GOLD)
        add_box(acc, c, fa, (0.015, 0.35, 0.02), col=GOLD_D, mat=M_GOLD)
        add_box(acc, c, fa, (0.27, 0.015, 0.02), col=GOLD_D, mat=M_GOLD)
    for xc in (-2.15, -1.2, -0.4, 0.4, 1.2, 2.15):
        z0, z1 = 2.75, 4.05
        a, nn = transom_pt(xc / hE, z0, 0.04)
        b = transom_pt(xc / hE, z1, 0.04)[0]
        add_beam(acc, a, b, 0.1, 0.06, up=nn, col=GOLD_D, mat=M_GOLD)
    # stern gallery (balcony)
    zg = 2.72
    xs = np.linspace(-0.82, 0.82, 17)
    inner = transom_pt(xs, np.full_like(xs, zg), 0.0)[0]
    outer = inner + V3(0, -0.8, 0)
    add_slab(acc, np.vstack([inner, outer[::-1]]), (0, 0, 1), 0.12, col=HULL_D, mat=M_HULL, side_col=GOLD_D)
    rail_path = np.vstack([inner[0] + V3(0, -0.05, 0), outer[1:-1] + V3(0, 0.05, 0), inner[-1] + V3(0, -0.05, 0)])
    rail_path[1:-1, 1] = outer[1:-1, 1] + 0.06
    rp = np.vstack([inner[0] + V3(0, -0.03, 0), outer[0] + V3(0, 0.06, 0), outer[1:-1] + V3(0, 0.06, 0),
                    outer[-1] + V3(0, 0.06, 0), inner[-1] + V3(0, -0.03, 0)])
    for dz, h, col, mat in ((0.68, 0.07, GOLD, M_GOLD), (0.12, 0.06, HULL_D, M_HULL)):
        add_sweep(acc, rp + V3(0, 0, dz), rect(0.09, h), col=col, mat=mat)
    seg = np.linalg.norm(np.diff(rp, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    for s in np.arange(0.12, cum[-1] - 0.05, 0.2):
        i = min(np.searchsorted(cum, s) - 1, len(seg) - 1)
        f = (s - cum[i]) / seg[i]
        bp = rp[i] + (rp[i + 1] - rp[i]) * f
        zz = np.linspace(0, 0.56, 6)
        add_tube(acc, bp + np.outer(zz, [0, 0, 1]) + V3(0, 0, 0.12), [0.025, 0.04, 0.028, 0.045, 0.03, 0.025],
                 sides=6, col=HULL_D, mat=M_HULL)
    # gallery brackets
    for xn_ in (-0.7, 0.0, 0.7):
        a = transom_pt(xn_, zg - 0.65, 0.0)[0]
        b = transom_pt(xn_, zg - 0.06, 0.0)[0] + V3(0, -0.7, 0)
        add_beam(acc, a, b, 0.1, 0.12, col=GOLD_D, mat=M_GOLD)
    # taffrail crest (gilded)
    xs = np.linspace(-0.78, 0.78, 25)
    hEt = info["hE_top"]
    yb = transom_y(xs, zt0) + 0.04
    top = zt0 + 0.08 + 0.62 * np.clip(1 - (xs / 0.78) ** 2, 0, 1) ** 1.3
    crest = np.vstack([np.stack([xs * hEt, yb, top], 1), np.stack([xs[::-1] * hEt, yb[::-1], np.full(25, zt0 + 0.06)], 1)])
    add_slab(acc, crest, (0, -1, 0), 0.12, col=GOLD_D, mat=M_GOLD)
    path = np.stack([xs * hEt, yb - 0.07, top + 0.02], 1)
    add_tube(acc, path, 0.045, sides=6, col=GOLD, mat=M_GOLD, caps=(True, True))
    # lanterns: centre (large) on the crest, two on the quarters
    lanterns.append(lantern(acc, V3(0, float(yb[12]) - 0.55, zt0 + 0.85), 1.25,
                            mount=V3(0, float(yb[12]), zt0 + 0.55)))
    for s in (1, -1):
        x = s * (hEt - 0.3)
        y = float(transom_y(x / hEt, zt0)) - 0.42
        lanterns.append(lantern(acc, V3(x, y, zt0 + 0.42), 1.0, mount=V3(x, y + 0.42, zt0 + 0.1)))
    info["lanterns"] = lanterns

    # quarter galleries: rounded, lofted bays with windows, gilded bands, drop finial and dome
    acc.tag = "stern"
    yc = -12.75
    zz = np.concatenate([np.linspace(1.85, 2.6, 7), np.linspace(2.6, 4.45, 9)[1:], np.linspace(4.45, 5.05, 7)[1:]])
    i_lo, i_hi = 6, 6 + 8
    aa = np.linspace(0, np.pi, 19)
    k = smoothstep(4.45, 5.05, zz)
    Lz = (0.25 + 1.85 * smoothstep(1.85, 2.6, zz) ** 0.6) * (1 - 0.85 * k)
    Wz = (0.06 + 0.52 * smoothstep(1.85, 2.65, zz) ** 0.7) * (1 - 0.82 * k ** 0.8)

    def gal_pt(a_, z_, side, extra_w=0.0):
        a_ = np.asarray(a_, float)
        z_ = np.asarray(z_, float)
        L_ = np.interp(z_, zz, Lz)
        W_ = np.interp(z_, zz, Wz) + extra_w
        y_ = yc + L_ / 2 * np.cos(a_)
        x_ = hb(y_, z_) - 0.12 + W_ * np.sin(a_)
        return np.stack(np.broadcast_arrays(side * x_, y_, z_), -1)
    for side in (1, -1):
        G = gal_pt(aa[:, None], zz[None, :], side)
        T = grid_tris(len(aa), len(zz))
        Pf = G.reshape(-1, 3)
        cen = np.array([side * float(hb(yc, 3.4)) - side * 0.5, yc, 3.4])
        fn = face_normals(Pf, T)
        cc = (Pf[T[:, 0]] + Pf[T[:, 1]] + Pf[T[:, 2]]) / 3
        if np.einsum("ij,ij->i", fn, cc - cen).sum() < 0:
            T = T[:, ::-1]
        Nn = vertex_normals(Pf, T).reshape(G.shape)
        UV = np.stack([G[..., 1], G[..., 2]], -1)
        add_grid(acc, G[:, :i_lo + 1], UV=UV[:, :i_lo + 1], N=Nn[:, :i_lo + 1], col=GOLD_D, mat=M_GOLD)
        add_grid(acc, G[:, i_lo:i_hi + 1], UV=UV[:, i_lo:i_hi + 1], N=Nn[:, i_lo:i_hi + 1], col=HULL * 1.1, mat=M_HULL)
        add_grid(acc, G[:, i_hi:], UV=UV[:, i_hi:], N=Nn[:, i_hi:], col=BLACK, mat=M_PAINT)
        # gilded bands
        for z_, r_ in ((2.6, 0.05), (4.45, 0.06), (3.0, 0.03)):
            add_tube(acc, gal_pt(aa, np.full_like(aa, z_), side, 0.02), r_, sides=6, col=GOLD, mat=M_GOLD,
                     caps=(True, True))
        # windows
        for da in (-0.62, 0.0, 0.62):
            a0 = np.pi / 2 + da
            pc = gal_pt(a0, 3.6, side)
            pa = gal_pt(a0 - 0.12, 3.6, side)
            pb = gal_pt(a0 + 0.12, 3.6, side)
            tv = nrm(pb - pa)
            nv = nrm(np.cross(tv, [0, 0, 1])) * 1.0
            if np.dot(nv, pc - cen) < 0:
                nv = -nv
            cs = [pc - tv * 0.19 + V3(0, 0, -0.36), pc + tv * 0.19 + V3(0, 0, -0.36),
                  pc + tv * 0.19 + V3(0, 0, 0.36), pc - tv * 0.19 + V3(0, 0, 0.36)]
            add_slab(acc, [c + nv * 0.03 for c in cs], nv, 0.03, col=GLASS, mat=M_GLASS, extra=140)
            fa = np.array([tv, [0, 0, 1.0], nv])
            for sgn in (-1, 1):
                add_box(acc, pc + nv * 0.05 + V3(0, 0, sgn * 0.4), fa, (0.25, 0.04, 0.035), col=GOLD, mat=M_GOLD)
                add_box(acc, pc + nv * 0.05 + tv * sgn * 0.22, fa, (0.035, 0.42, 0.035), col=GOLD, mat=M_GOLD)
            add_box(acc, pc + nv * 0.045, fa, (0.012, 0.36, 0.02), col=GOLD_D, mat=M_GOLD)
        # finials
        bot = gal_pt(np.pi / 2, 1.85, side)
        add_tube(acc, [bot + V3(0, 0, 0.1), bot, bot + V3(0, 0, -0.35)], [0.1, 0.09, 0.01], sides=8, col=GOLD,
                 mat=M_GOLD, caps=(True, False))
        topp = gal_pt(np.pi / 2, 5.05, side)
        add_tube(acc, [topp + V3(0, 0, -0.1), topp + V3(0, 0, 0.05), topp + V3(0, 0, 0.3)], [0.13, 0.08, 0.01],
                 sides=8, col=GOLD, mat=M_GOLD, caps=(True, False))


def lantern(acc, base, s, mount=None):
    """Stern lantern. Returns glass centre (Blender space)."""
    b = np.asarray(base, float)
    add_tube(acc, [b, b + V3(0, 0, 0.08 * s)], [0.24 * s, 0.2 * s], sides=6, caps=(True, True), col=IRON, mat=M_IRON)
    g0 = b + V3(0, 0, 0.08 * s)
    g1 = g0 + V3(0, 0, 0.5 * s)
    add_tube(acc, [g0, (g0 + g1) / 2, g1], [0.17 * s, 0.2 * s, 0.17 * s], sides=6, caps=(False, False),
             col=GLASS, mat=M_GLASS, extra=255)
    for k in range(6):
        a = 2 * math.pi * k / 6
        off = V3(math.cos(a), math.sin(a), 0) * 0.195 * s
        add_tube(acc, [g0 + off * 0.88, (g0 + g1) / 2 + off * 1.02, g1 + off * 0.88], 0.013 * s, sides=4,
                 col=IRON, mat=M_IRON)
    add_tube(acc, [g1, g1 + V3(0, 0, 0.06 * s), g1 + V3(0, 0, 0.3 * s)], [0.22 * s, 0.22 * s, 0.03 * s], sides=6,
             caps=(True, False), col=IRON, mat=M_IRON)
    add_tube(acc, [g1 + V3(0, 0, 0.28 * s), g1 + V3(0, 0, 0.36 * s), g1 + V3(0, 0, 0.42 * s)],
             [0.02 * s, 0.06 * s, 0.01 * s], sides=6, col=GOLD, mat=M_GOLD)
    if mount is not None:
        m = np.asarray(mount, float)
        mid = V3(b[0], b[1], m[2]) * 0.5 + m * 0.5
        add_tube(acc, [m, mid + V3(0, 0, 0.05), b - V3(0, 0, 0.01)], 0.035, sides=6, col=IRON, mat=M_IRON)
    return (g0 + g1) / 2


# ----------------------------------------------------------------------------
# deck furniture
# ----------------------------------------------------------------------------
def balustrade(acc, path, h=0.85, pitch=0.24, rail_col=BLACK, rail_mat=M_PAINT):
    path = np.asarray(path, float)
    add_sweep(acc, path + V3(0, 0, h), rect(0.14, 0.08), col=rail_col, mat=rail_mat)
    add_sweep(acc, path + V3(0, 0, 0.06), rect(0.12, 0.08), col=HULL_D, mat=M_HULL)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    for s in np.arange(pitch / 2, cum[-1], pitch):
        i = min(np.searchsorted(cum, s) - 1, len(seg) - 1)
        i = max(i, 0)
        f = (s - cum[i]) / seg[i]
        bp = path[i] + (path[i + 1] - path[i]) * f
        zz = np.linspace(0.1, h - 0.04, 6)
        add_tube(acc, bp + np.outer(zz, [0, 0, 1]), [0.03, 0.045, 0.03, 0.05, 0.034, 0.028], sides=6,
                 col=HULL_D, mat=M_HULL)


def barrel(acc, c, r=0.3, h=0.8):
    zz = np.linspace(0, h, 7)
    rr = r * (0.86 + 0.14 * np.sin(np.pi * zz / h))
    add_tube(acc, np.asarray(c) + np.outer(zz, [0, 0, 1]), rr, sides=12, caps=(False, True), col=HULL * 1.5,
             mat=M_HULL)
    for z in (0.1, h - 0.1):
        rz = r * (0.86 + 0.14 * np.sin(np.pi * z / h)) + 0.008
        add_tube(acc, [np.asarray(c) + V3(0, 0, z - 0.035), np.asarray(c) + V3(0, 0, z + 0.035)], rz, sides=12,
                 col=IRON, mat=M_IRON)


def build_furniture(acc, info):
    acc.tag = "deck_items"
    # balustrades at the breaks
    for yb, zl, face in ((Y_QD - 0.1, float(deck_main(Y_QD)) + QD_H + 0.09, 1),
                         (Y_FC + 0.1, float(deck_main(Y_FC)) + FC_H + 0.09, -1)):
        xin = float(hb(yb, zl)) - BW_T - 0.05
        for xa, xb in ((-xin, -0.75), (0.75, xin)):
            path = np.stack([np.linspace(xa, xb, 6), np.full(6, yb), np.full(6, zl - 0.09)], 1)
            balustrade(acc, path)
        # ladders (stairs) from main deck at the centre opening
        zm = float(deck_main(Y_QD if face > 0 else Y_FC))
        zu = zl - 0.09
        for xs in (-0.45, 0.45):
            a = V3(xs, yb + face * 1.5, zm + 0.05)
            b = V3(xs, yb, zu)
            add_beam(acc, a, b, 0.06, 0.2, col=HULL_D, mat=M_HULL)
        for k in range(1, 6):
            f = k / 6
            p = V3(0, yb + face * 1.5 * (1 - f), zm + (zu - zm) * f)
            add_box(acc, p, np.eye(3), (0.42, 0.12, 0.025), col=DECK * 0.85, mat=M_DECK)
    # capstan
    yc = -4.3
    zd = float(deck_main(yc))
    add_tube(acc, [V3(0, yc, zd), V3(0, yc, zd + 0.15), V3(0, yc, zd + 0.65), V3(0, yc, zd + 0.85)],
             [0.6, 0.42, 0.36, 0.55], sides=14, caps=(False, True), col=HULL * 1.4, mat=M_HULL)
    for k in range(6):
        a = math.pi * k / 6
        d = V3(math.cos(a), math.sin(a), 0)
        add_tube(acc, [V3(0, yc, zd + 0.75) - d * 1.3, V3(0, yc, zd + 0.75) + d * 1.3], 0.04, sides=6,
                 col=DECK * 0.8, mat=M_DECK)
    # hatches
    for y, hx, hy in ((-1.9, 0.75, 0.65), (4.6, 1.05, 1.0)):
        zd = float(deck_main(y)) + 0.09
        add_box(acc, (0, y, zd + 0.16), np.eye(3), (hx + 0.08, hy + 0.08, 0.16), col=HULL * 1.3, mat=M_HULL)
        add_box(acc, (0, y, zd + 0.33), np.eye(3), (hx, hy, 0.02), col=DECK_D * 0.6, mat=M_DECK)
        for k in np.linspace(-hx + 0.1, hx - 0.1, int(hx * 8)):
            add_box(acc, (k, y, zd + 0.355), np.eye(3), (0.025, hy, 0.012), col=DECK_D, mat=M_DECK)
    # ship's boat on the main hatch
    build_boat(acc, V3(0, 4.6, float(deck_main(4.6)) + 0.75))
    # barrels
    for (x, y) in ((-2.6, 9.6), (-2.0, 9.75), (-2.3, 9.2), (2.5, 9.6), (2.7, -5.9), (2.1, -6.0)):
        barrel(acc, V3(x, y, float(deck_level(y)) + 0.08))
    # fife rails / bitts around masts
    for m in ("fore", "main"):
        M = MASTS[m]
        y = M["y"]
        zd = float(deck_level(y)) + 0.08
        for dx in (-0.8, 0.8):
            for dy in (-0.55, 0.55):
                add_box(acc, (dx, y + dy, zd + 0.45), np.eye(3), (0.07, 0.07, 0.45), col=HULL_D, mat=M_HULL)
        for dy in (-0.55, 0.55):
            add_box(acc, (0, y + dy, zd + 0.85), np.eye(3), (0.92, 0.07, 0.05), col=HULL_D, mat=M_HULL)
        for dx in (-0.8, 0.8):
            add_box(acc, (dx, y, zd + 0.85), np.eye(3), (0.07, 0.62, 0.05), col=HULL_D, mat=M_HULL)
    # catheads + anchors
    for side in (1, -1):
        yc = 13.5
        zc = float(z_top(yc)) - 0.15
        a = V3(side * (float(hb(yc, zc)) - 0.5), yc - 0.3, zc)
        b = V3(side * (float(hb(yc, zc)) + 0.95), yc + 0.35, zc + 0.15)
        add_beam(acc, a, b, 0.28, 0.28, col=BLACK, mat=M_PAINT)
        anchor(acc, b + V3(0, 0, -0.25), side)
    # bitts at bowsprit heel
    add_box(acc, (0, BS_HEEL[1] - 0.3, BS_HEEL[2] - 0.1), np.eye(3), (0.5, 0.2, 0.45), col=HULL_D, mat=M_HULL)


def anchor(acc, top, side):
    top = np.asarray(top, float)
    bot = top + V3(0, 0, -2.4)
    add_tube(acc, [top, bot], 0.07, sides=6, caps=(True, True), col=IRON, mat=M_IRON)
    add_tube(acc, [top - V3(0.9, 0, 0) + V3(0, 0, -0.3), top + V3(0.9, 0, 0) + V3(0, 0, -0.3)], [0.06, 0.09][0],
             sides=6, caps=(True, True), col=HULL * 1.3, mat=M_HULL)
    ang = np.linspace(0, 1, 8)
    for s in (-1, 1):
        pts = np.stack([np.zeros(8) + top[0], top[1] + s * 0.9 * np.sin(ang * 1.2), bot[2] + 0.95 * (1 - np.cos(ang * 1.2))], 1)
        add_tube(acc, pts, 0.06, sides=6, col=IRON, mat=M_IRON, caps=(False, True))
        tip = pts[-1]
        add_box(acc, tip - V3(0, s * 0.08, 0.12), frame_axes(pts[-1] - pts[-3], (1, 0, 0)), (0.2, 0.16, 0.025),
                col=IRON, mat=M_IRON)
    ring_c = top + V3(0, 0, 0.12)
    a = np.linspace(0, 2 * np.pi, 13)
    add_tube(acc, ring_c + np.stack([np.zeros(13), 0.12 * np.cos(a), 0.12 * np.sin(a)], 1), 0.025, sides=4,
             col=IRON, mat=M_IRON)


def build_boat(acc, c):
    L, B, D = 4.6, 1.55, 0.62
    nu, nv = 26, 9
    u = np.linspace(-1, 1, nu)
    w = (1 - np.abs(u) ** 2.2) ** 0.6 * B / 2
    w[0] = w[-1] = 0.0
    sheer = 0.12 * u ** 2
    a = np.linspace(0, np.pi / 2, nv)
    X = w[:, None] * np.sin(a)[None, :]
    Zs = -D * (np.cos(a)[None, :]) * (1 - 0.35 * u[:, None] ** 2) + sheer[:, None]
    Y = np.repeat((u * L / 2)[:, None], nv, 1)
    for side in (1, -1):
        G = np.stack([side * X, Y, Zs], -1) + c
        add_grid(acc, G, UV=np.stack([Y, Zs], -1), col=hexc("#6e4e2c"), mat=M_PAINT, expect=(side, 0, 0))
        Gi = np.stack([side * np.maximum(X - 0.05, 0), Y * 0.98, Zs + 0.05], -1) + c
        add_grid(acc, Gi, UV=np.stack([Y, Zs], -1), col=HULL * 1.3, mat=M_HULL, expect=(-side, 0, 0))
        rim = np.stack([np.stack([side * X[:, -1], Y[:, -1], Zs[:, -1]], -1),
                        np.stack([side * np.maximum(X[:, -1] - 0.05, 0), Y[:, -1] * 0.98, Zs[:, -1] + 0.05], -1)],
                       1) + c
        add_tube(acc, np.stack([side * (X[:, -1] - 0.02), Y[:, -1], Zs[:, -1] + 0.03], -1) + c, 0.04, sides=5,
                 col=BLACK, mat=M_PAINT)
    for yy in (-1.2, 0.0, 1.2):
        add_box(acc, c + V3(0, yy, -0.12), np.eye(3), (B / 2 * 0.85, 0.12, 0.025), col=DECK, mat=M_DECK)
    for yy in (-1.0, 1.0):
        add_box(acc, c + V3(0, yy, -0.55), np.eye(3), (0.5, 0.12, 0.2), col=HULL_D, mat=M_HULL)


# ----------------------------------------------------------------------------
# masts, spars, sails, rigging
# ----------------------------------------------------------------------------
MASTS = {
    "fore": dict(y=8.6, rake=1.5, rl=0.30, low_top=13.2, top_z=12.7, topw=3.0,
                 tm=(11.3, 20.6, 0.21), tg=(19.7, 27.2, 0.13), tg_rig=25.9, ctw=1.9, nshroud=4),
    "main": dict(y=0.4, rake=2.5, rl=0.35, low_top=15.0, top_z=14.5, topw=3.4,
                 tm=(13.0, 23.7, 0.23), tg=(22.5, 31.0, 0.14), tg_rig=28.9, ctw=2.2, nshroud=5),
    "mizzen": dict(y=-9.4, rake=4.0, rl=0.25, low_top=12.9, top_z=12.4, topw=2.5,
                   tm=(11.4, 20.4, 0.16), tg=None, tg_rig=None, ctw=1.5, nshroud=3),
}
for _k, _M in MASTS.items():
    _M["zd"] = float(deck_level(_M["y"]))
    _M["off1"] = 0.8 * _M["rl"] + 0.85 * _M["tm"][2] + 0.06
    _M["off2"] = _M["off1"] + (0.75 * _M["tm"][2] + 0.9 * _M["tg"][2] + 0.05 if _M["tg"] else 0)

BRACE = math.radians(16.0)
YDIR = np.array([math.cos(BRACE), math.sin(BRACE), 0.0])
YFWD = np.array([-math.sin(BRACE), math.cos(BRACE), 0.0])

YARDS = {
    "fore": [("course", 11.9, 14.6, 0.20, 0), ("topsail", 19.0, 11.0, 0.15, 1), ("tgallant", 24.6, 7.4, 0.10, 2)],
    "main": [("course", 13.7, 16.6, 0.22, 0), ("topsail", 21.9, 12.6, 0.17, 1), ("tgallant", 28.0, 8.4, 0.11, 2)],
    "mizzen": [("crossjack", 11.6, 11.0, 0.15, 0), ("topsail", 17.9, 8.2, 0.12, 1)],
}

BS_HEEL = None


def ax(m, z, level=0):
    M = MASTS[m]
    off = [0.0, M["off1"], M["off2"]][level]
    return np.array([0.0, M["y"] - math.tan(math.radians(M["rake"])) * (z - M["zd"]) + off, z])


def mast_r(m, level):
    M = MASTS[m]
    return [M["rl"] * 0.8, M["tm"][2] * 0.8, (M["tg"][2] if M["tg"] else 0.1) * 0.7][level]


def yard_geo(m, i):
    name, z, L, r, lev = YARDS[m][i]
    c = ax(m, z, lev) + YFWD * (mast_r(m, lev) + r + 0.06)
    return dict(name=name, z=z, L=L, r=r, lev=lev, c=c,
                arm=lambda s, d=0.0: c + YDIR * s * (L / 2 - d))


def build_masts(acc, info):
    acc.tag = "masts"
    tops = {}
    for m, M in MASTS.items():
        zd = M["zd"]
        # lower mast
        zz = np.linspace(zd - 0.8, M["low_top"], 8)
        rr = M["rl"] * (1 - 0.22 * (zz - zd) / (M["low_top"] - zd))
        add_tube(acc, [ax(m, z, 0) for z in zz], rr, sides=14, caps=(False, True), col=MAST, mat=M_SPAR)
        # partner ring at deck
        add_tube(acc, [ax(m, zd + 0.05, 0), ax(m, zd + 0.25, 0)], [M["rl"] + 0.12, M["rl"] + 0.06], sides=14,
                 caps=(False, True), col=HULL_D, mat=M_HULL)
        for z in np.arange(zd + 1.6, M["top_z"] - 2.5, 1.5):
            r = M["rl"] * (1 - 0.22 * (z - zd) / (M["low_top"] - zd)) + 0.018
            add_tube(acc, [ax(m, z - 0.06, 0), ax(m, z + 0.06, 0)], r, sides=14, col=BLACK, mat=M_IRON)
        # topmast
        a, b, r = M["tm"]
        zz = np.linspace(a, b, 6)
        add_tube(acc, [ax(m, z, 1) for z in zz], np.linspace(r, r * 0.68, 6), sides=12, caps=(True, True),
                 col=MAST, mat=M_SPAR)
        # topgallant (with pole) or truck
        if M["tg"]:
            a2, b2, r2 = M["tg"]
            zz = np.linspace(a2, b2, 6)
            add_tube(acc, [ax(m, z, 2) for z in zz], np.linspace(r2, r2 * 0.45, 6), sides=10, caps=(True, True),
                     col=MAST, mat=M_SPAR)
            truck = ax(m, b2, 2)
            info.setdefault("trucks", {})[m] = truck
        else:
            truck = ax(m, b, 1)
            info.setdefault("trucks", {})[m] = truck
        add_tube(acc, [truck, truck + V3(0, 0, 0.12)], 0.13, sides=10, caps=(True, True), col=MAST * 0.8, mat=M_SPAR)
        # caps
        zc = M["low_top"] - 0.15
        p0, p1 = ax(m, zc, 0), ax(m, zc, 1)
        add_box(acc, (p0 + p1) / 2, np.eye(3), (M["rl"] * 0.95, (p1[1] - p0[1]) / 2 + M["rl"] * 0.9, 0.16),
                col=BLACK, mat=M_PAINT)
        if M["tg"]:
            zc = M["tm"][1] - 0.12
            p0, p1 = ax(m, zc, 1), ax(m, zc, 2)
            add_box(acc, (p0 + p1) / 2, np.eye(3), (M["tm"][2] * 0.85, (p1[1] - p0[1]) / 2 + M["tm"][2] * 0.8, 0.12),
                    col=BLACK, mat=M_PAINT)
        # top platform
        W = M["topw"]
        Ln = 0.72 * W
        tc = ax(m, M["top_z"], 0) + V3(0, 0.25, 0)
        arc = [(W / 2 * math.cos(t), 0.08 * Ln + 0.5 * Ln * math.sin(t)) for t in np.linspace(0, np.pi, 15)]
        poly = [(-W / 2, -0.42 * Ln), (W / 2, -0.42 * Ln)] + arc
        poly = [tc + V3(x, y, -0.07) for x, y in poly]
        add_slab(acc, poly, (0, 0, 1), 0.14, col=MAST * 0.9, mat=M_SPAR)
        # aft rail on the top
        add_beam(acc, tc + V3(-W / 2 + 0.1, -0.4 * Ln, 0.55), tc + V3(W / 2 - 0.1, -0.4 * Ln, 0.55), 0.06, 0.06,
                 col=BLACK, mat=M_PAINT)
        for x in np.linspace(-W / 2 + 0.1, W / 2 - 0.1, 5):
            add_beam(acc, tc + V3(x, -0.4 * Ln, 0.05), tc + V3(x, -0.4 * Ln, 0.55), 0.04, 0.04, col=BLACK, mat=M_PAINT)
        tops[m] = dict(tc=tc, W=W, Ln=Ln)
        # crosstrees
        ctz = M["tm"][1] - 0.75
        cc = ax(m, ctz, 1)
        for dy in (-0.3, 0.3):
            add_box(acc, cc + V3(0, dy, 0), np.eye(3), (M["ctw"] / 2, 0.06, 0.06), col=MAST * 0.85, mat=M_SPAR)
        for dx in (-0.2, 0.2):
            add_box(acc, cc + V3(dx, 0.1, 0.1), np.eye(3), (0.06, 0.6, 0.07), col=MAST * 0.85, mat=M_SPAR)
        M["ctz"] = ctz
    # crow's nest on the main topmast crosstrees
    M = MASTS["main"]
    cn = ax("main", M["ctz"] + 0.07, 2)
    zz = np.linspace(0, 1.1, 6)
    rr = 0.66 + 0.05 * np.sin(np.pi * zz / 1.1)
    add_tube(acc, cn + np.outer(zz, [0, 0, 1]), rr, sides=16, col=DECK * 0.75, mat=M_DECK)
    add_tube(acc, cn + np.outer(zz[::-1], [0, 0, 1]), rr - 0.05, sides=16, col=DECK_D, mat=M_DECK)
    # flip inner normals: build explicitly
    acc.parts[-1] = flip_part(acc.parts[-1])
    add_tube(acc, [cn + V3(0, 0, 1.1), cn + V3(0, 0, 1.14)], 0.69, sides=16, caps=(False, True), col=BLACK, mat=M_PAINT)
    acc.add(*disc(cn + V3(0, 0, 0.02), 0.63, 16, V3(0, 0, 1)), col=DECK_D, mat=M_DECK)
    acc.add(*disc(cn + V3(0, 0, -0.01), 0.68, 16, V3(0, 0, -1)), col=DECK_D, mat=M_DECK)
    for z in (0.15, 0.95):
        add_tube(acc, [cn + V3(0, 0, z - 0.03), cn + V3(0, 0, z + 0.03)],
                 0.66 + 0.05 * math.sin(math.pi * z / 1.1) + 0.012, sides=16, col=IRON, mat=M_IRON)
    info["tops"] = tops
    info["crow"] = cn


def flip_part(part):
    P, N, UV, C, M, R, E, W, T = part
    return (P, -N, UV, C, M, R, E, W, T[:, ::-1])


def disc(c, r, n, nv):
    nv = nrm(nv)
    a1 = nrm(np.cross(nv, [1, 0, 0] if abs(nv[0]) < 0.9 else [0, 1, 0]))
    a2 = np.cross(nv, a1)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = c + r * (np.cos(ang)[:, None] * a1 + np.sin(ang)[:, None] * a2)
    P = np.vstack([c, ring])
    T = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    return P, np.array(T), np.tile(nv, (n + 1, 1))


def build_yards(acc):
    acc.tag = "spars"
    for m in YARDS:
        for i in range(len(YARDS[m])):
            g = yard_geo(m, i)
            s = np.linspace(-1, 1, 13)
            pts = g["c"] + np.outer(s * g["L"] / 2, YDIR)
            rr = g["r"] * (1 - 0.55 * np.abs(s) ** 1.6)
            add_tube(acc, pts, rr, sides=8, caps=(True, True), col=YARD, mat=M_SPAR)
            # sling / truss
            add_box(acc, g["c"], np.array([YDIR, YFWD, [0, 0, 1]]), (g["r"] * 1.6, g["r"] * 1.3, g["r"] * 1.3),
                    col=BLACK, mat=M_PAINT)


# --- sails ---------------------------------------------------------------
SAIL_BLACK = hexc("#29231e")
SAIL_BLEACH = hexc("#4a443b")
SAIL_SALT = hexc("#5c5850")
SAIL_GRIME = hexc("#17130f")


def sail_color(U, V, seed, patches=4):
    """Weathered black canvas. U, V in 0..1 (V down from the head)."""
    U = np.asarray(U, float).reshape(-1)
    V = np.asarray(V, float).reshape(-1)
    rng = np.random.default_rng(100 + seed)
    p3 = np.stack([U * 3.0, V * 3.0, np.full_like(U, seed * 1.37)], 1)
    c = np.tile(SAIL_BLACK * (1.0 + 0.06 * math.sin(seed * 2.1)), (len(U), 1))
    # sun-bleached areas (stronger high up)
    b = smoothstep(0.5, 0.78, fbm(p3, 4, seed=11)) * (0.55 + 0.45 * (1 - V))
    c = c + (SAIL_BLEACH - c) * b[:, None] * 0.85
    # salt stains: vertical streaks, more toward the foot
    st = smoothstep(0.62, 0.85, fbm(np.stack([U * 14.0, V * 1.6, np.full_like(U, seed)], 1), 3, seed=12)) * V ** 0.7
    c = c + (SAIL_SALT - c) * st[:, None] * 0.45
    # grime near the foot and along the edges (bolt ropes)
    c = c + (SAIL_GRIME - c) * (0.75 * V ** 3)[:, None]
    edge = np.minimum(np.minimum(U, 1 - U), V)
    c = c * (1 - 0.18 * np.exp(-edge / 0.03))[:, None]
    # sewn-on repair patches (rectangles of another tone)
    for _ in range(patches):
        cu, cv = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)
        hu, hv = rng.uniform(0.04, 0.09), rng.uniform(0.035, 0.08)
        m = smoothstep(1.0, 0.8, np.maximum(np.abs(U - cu) / hu, np.abs(V - cv) / hv))
        tone = SAIL_BLACK * rng.choice([0.7, 1.45, 1.75]) * np.array([1.0, rng.uniform(0.95, 1.03), rng.uniform(0.9, 1.0)])
        c = c + (tone - c) * m[:, None]
    return np.clip(c, 0, 1)


def square_sail(acc, TL, TR, BL, BR, wind_n, depth, nu=24, nv=18, roach=0.0, seed=0, reefs=(), torn=False):
    u = np.linspace(0, 1, nu)
    v = np.linspace(0, 1, nv)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    if torn:
        rng = np.random.default_rng(500 + seed)
        r = rng.uniform(0.0, 0.07, nu) * np.where(np.arange(nu) % 2 == 0, 1.0, 0.35)
        for k in rng.choice(np.arange(3, nu - 3), 3, replace=False):
            r[k] = rng.uniform(0.12, 0.2)
            r[k - 1] = max(r[k - 1], r[k] * 0.5)
        r[:2] = 0.0
        r[-2:] = 0.0
        Vv = Vv - r[:, None] * Vv ** 6
    uc = 2 * U - 1
    head = TL[None, None] * (1 - U)[..., None] + TR[None, None] * U[..., None]
    foot = BL[None, None] * (1 - U)[..., None] + BR[None, None] * U[..., None]
    P = head * (1 - Vv)[..., None] + foot * Vv[..., None]
    P[..., 2] += roach * (1 - uc ** 2) * Vv ** 5
    A = 0.28 + 0.72 * (1 - uc ** 2) ** 1.2
    B = np.sin(np.pi * 0.8 * Vv)
    wr = 0.04 * np.sin(7.0 * Vv + 2.5 * uc + seed) * np.sin(5.0 * uc + seed * 1.3) * Vv
    D = depth * A * B + depth * wr
    P = P + wind_n[None, None, :] * D[..., None]
    # leeches pulled in a little by the belly
    P = P - YDIR[None, None, :] * (uc * 0.06 * depth * B)[..., None]
    wtop = np.linalg.norm(TR - TL)
    wbot = np.linalg.norm(BR - BL)
    hgt = np.linalg.norm((TL + TR) / 2 - (BL + BR) / 2)
    UV = np.stack([(U - 0.5) * (wtop * (1 - Vv) + wbot * Vv), Vv * hgt], -1)
    f = np.sqrt(Vv) * (1 - np.abs(uc) ** 4 * Vv ** 2) * (0.75 + 0.25 * (1 - uc ** 2))
    extra = np.clip(f, 0, 1) * 255
    T = grid_tris(nu, nv)
    col = sail_color(U, Vv, seed).reshape(nu, nv, 3)
    for rv in reefs:
        col = col * (1 - 0.25 * np.exp(-((Vv - rv) / 0.025) ** 2))[..., None]
    add_cloth(acc, P.reshape(-1, 3), T, UV.reshape(-1, 2), col.reshape(-1, 3), extra.ravel(), M_SAIL, wind_n)
    return P, hgt


def tri_sail(acc, H, Tk, C, wind_n, depth, n=18, seed=0):
    pts, uvw = [], []
    idx = {}
    for i in range(n + 1):
        for j in range(i + 1):
            a = 1 - i / n
            c = j / n
            b = (i - j) / n
            idx[(i, j)] = len(pts)
            pts.append(a * H + b * Tk + c * C)
            uvw.append((a, b, c))
    P = np.array(pts)
    W = np.array(uvw)
    a, b, c = W[:, 0], W[:, 1], W[:, 2]
    D = depth * 2.8 * c ** 0.75 * (1 - c) * (1 - 0.35 * b) + 0.03 * np.sin(9 * c + seed) * c * (1 - c)
    P = P + wind_n[None, :] * D[:, None]
    T = []
    for i in range(n):
        for j in range(i + 1):
            T.append((idx[(i, j)], idx[(i + 1, j)], idx[(i + 1, j + 1)]))
            if j < i:
                T.append((idx[(i, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
    hd = nrm(V3(*(C - Tk)[:2], 0))
    UV = np.stack([(P - Tk) @ hd - ((H - Tk) @ hd) * a, H[2] - P[:, 2]], 1)
    UV[:, 0] -= 0.5 * (UV[:, 0].min() + UV[:, 0].max())
    extra = np.clip(np.minimum(1, 2.2 * c) * (1 - c ** 4) * (0.6 + 0.4 * (1 - b)), 0, 1) * 255
    Vv = (H[2] - P[:, 2]) / max(H[2] - min(Tk[2], C[2]), 1e-3)
    add_cloth(acc, P, np.array(T), UV, sail_color(c, Vv, seed, patches=2), extra, M_SAIL, wind_n)


def quad_sail_fa(acc, throat, peak, tack, clew, wind_n, depth, nu=20, nv=18, seed=0):
    """Fore-and-aft four-sided sail (spanker). u: luff->leech, v: head->foot."""
    u = np.linspace(0, 1, nu)
    v = np.linspace(0, 1, nv)
    U, Vv = np.meshgrid(u, v, indexing="ij")
    head = throat[None, None] * (1 - U)[..., None] + peak[None, None] * U[..., None]
    foot = tack[None, None] * (1 - U)[..., None] + clew[None, None] * U[..., None]
    P = head * (1 - Vv)[..., None] + foot * Vv[..., None]
    D = depth * U ** 0.7 * np.sin(np.pi * Vv) * (1 - 0.25 * U) + 0.03 * np.sin(6 * U + 5 * Vv + seed) * U
    P = P + wind_n[None, None, :] * D[..., None]
    hl = np.linalg.norm(clew - tack)
    hh = np.linalg.norm(throat - tack)
    UV = np.stack([(U - 0.5) * hl, Vv * hh], -1)
    extra = np.clip(U ** 0.8 * np.sin(np.pi * Vv) ** 0.5, 0, 1) * 255
    add_cloth(acc, P.reshape(-1, 3), grid_tris(nu, nv), UV.reshape(-1, 2), sail_color(U, Vv, seed, patches=3), extra.ravel(),
              M_SAIL, wind_n)


SKULL_SAILS = set()   # no skull sails any more: all sails carry material.w = 0
TORN_SAILS = {("main", "course"), ("fore", "topsail"), ("mizzen", "topsail")}


def build_sails(acc, info):
    acc.tag = "sails"
    seed = 0
    course_foot = {"fore": 5.5, "main": 4.7}
    depth = {"course": 1.5, "topsail": 1.75, "tgallant": 1.0}
    sails = {}
    for m in ("fore", "main", "mizzen"):
        for i, (name, z, L, r, lev) in enumerate(YARDS[m]):
            if name == "crossjack":
                continue
            g = yard_geo(m, i)
            TL = g["c"] - YDIR * (L / 2 - 0.35) + V3(0, 0, -r - 0.06) + YFWD * 0.05
            TR = g["c"] + YDIR * (L / 2 - 0.35) + V3(0, 0, -r - 0.06) + YFWD * 0.05
            if name == "course":
                zf = course_foot[m]
                hw = L / 2 - 0.35 + 0.3
                cz = g["c"].copy()
                cz[2] = zf
                BL = cz - YDIR * hw + YFWD * 0.35
                BR = cz + YDIR * hw + YFWD * 0.35
                roach, d = 0.55, depth["course"]
            else:
                lo = yard_geo(m, i - 1)
                BL = lo["c"] - YDIR * (lo["L"] / 2 - 0.55) + V3(0, 0, lo["r"] + 0.08) + YFWD * 0.3
                BR = lo["c"] + YDIR * (lo["L"] / 2 - 0.55) + V3(0, 0, lo["r"] + 0.08) + YFWD * 0.3
                roach = 0.25
                d = depth[name] * (0.75 if m == "mizzen" else 1.0)
            nu, nv = (32, 34) if name != "tgallant" else (18, 16)
            key = (m, name)
            # height for the skull sails (decimetres in material.w)
            hgt_est = float(np.linalg.norm((TL + TR) / 2 - (BL + BR) / 2))
            acc.cur_w = int(np.clip(round(hgt_est * 10), 1, 254)) if key in SKULL_SAILS else 0
            P, hgt = square_sail(acc, TL, TR, BL, BR, YFWD, d, nu=nu, nv=nv, roach=roach, seed=seed,
                                 reefs=(0.12, 0.24, 0.36) if name == "topsail" else ((0.15,) if name == "course" else ()),
                                 torn=key in TORN_SAILS)
            if key in SKULL_SAILS:
                info.setdefault("skull_sails", []).append((m + " " + name, hgt, acc.cur_w))
            acc.cur_w = 0
            sails[(m, name)] = dict(TL=TL, TR=TR, BL=P[0, -1], BR=P[-1, -1])
            if name in ("course", "topsail"):
                nu_ = P.shape[0]
                for fu in (0.3, 0.7):
                    col_ = P[int(round(fu * (nu_ - 1)))]
                    line = col_ + YFWD[None, :] * 0.035
                    line = np.vstack([g["c"] + YDIR * (fu - 0.5) * (L - 0.7) + YFWD * (r + 0.05), line])
                    acc.tag = "running"
                    add_tube(acc, line, 0.018, sides=4, col=HEMP, mat=M_ROPE)
                    acc.tag = "sails"
            seed += 1
    info["sails"] = sails
    # headsails
    port = V3(-1, 0, 0)
    for k, (stay_top, stay_bot, ft, fh, cfac, xoff, dpt) in enumerate(info["headstays"]):
        Tk = stay_bot + (stay_top - stay_bot) * ft
        H = stay_bot + (stay_top - stay_bot) * fh
        C = Tk + V3(-xoff, -(Tk[1] - H[1]) * cfac, (H[2] - Tk[2]) * (0.07 + 0.05 * k))
        tri_sail(acc, H, Tk, C, port, dpt, n=18, seed=seed)
        info.setdefault("jib_clews", []).append(C)
        seed += 1
    # spanker
    M = MASTS["mizzen"]
    phi = math.radians(11)
    bdir = V3(-math.sin(phi), -math.cos(phi), 0)
    gdir = nrm(V3(-math.sin(phi) * math.cos(math.radians(40)), -math.cos(phi) * math.cos(math.radians(40)),
                  math.sin(math.radians(40))))
    zb = M["zd"] + 3.0
    zg = 11.8
    boom0 = ax("mizzen", zb, 0) + bdir * (M["rl"] + 0.1)
    gaff0 = ax("mizzen", zg, 0) + gdir * (M["rl"] + 0.1)
    boom1 = boom0 + bdir * 9.4
    gaff1 = gaff0 + gdir * 6.8
    add_tube(acc, [boom0 + bdir * s for s in np.linspace(0, 9.4, 6)], np.linspace(0.16, 0.11, 6), sides=8,
             caps=(True, True), col=YARD, mat=M_SPAR)
    add_tube(acc, [gaff0 + gdir * s for s in np.linspace(0, 6.8, 6)], np.linspace(0.13, 0.08, 6), sides=8,
             caps=(True, True), col=YARD, mat=M_SPAR)
    bn = nrm(np.cross(bdir, V3(0, 0, 1)))
    if bn[0] > 0:
        bn = -bn
    quad_sail_fa(acc, gaff0 + V3(0, 0, -0.12), gaff1 + V3(0, 0, -0.12) - gdir * 0.25, boom0 + V3(0, 0, 0.14),
                 boom1 + V3(0, 0, 0.14) - bdir * 0.3, bn, 0.75, seed=seed)
    info["spanker"] = dict(boom0=boom0, boom1=boom1, gaff0=gaff0, gaff1=gaff1)


# --- flag ----------------------------------------------------------------
def sd_circle(px, py, cx, cy, r):
    return np.hypot(px - cx, py - cy) - r


def sd_capsule(px, py, ax_, ay, bx, by, r):
    pax, pay = px - ax_, py - ay
    bax, bay = bx - ax_, by - ay
    h = np.clip((pax * bax + pay * bay) / (bax * bax + bay * bay), 0, 1)
    return np.hypot(pax - bax * h, pay - bay * h) - r


def sd_rbox(px, py, cx, cy, hx, hy, r):
    qx = np.abs(px - cx) - hx + r
    qy = np.abs(py - cy) - hy + r
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def jolly_roger(u, v):
    cx = 1.25
    bones = None
    for (x0, y0, x1, y1) in ((cx - 0.5, 0.9, cx + 0.5, 1.36), (cx + 0.5, 0.9, cx - 0.5, 1.36)):
        d = sd_capsule(u, v, x0, y0, x1, y1, 0.042)
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        px, py = -dy / L, dx / L
        for (ex, ey) in ((x0, y0), (x1, y1)):
            for s in (-1, 1):
                d = np.minimum(d, sd_circle(u, v, ex + s * px * 0.045, ey + s * py * 0.045, 0.055))
        bones = d if bones is None else np.minimum(bones, d)
    skull = np.minimum(sd_circle(u, v, cx, 0.52, 0.28), sd_rbox(u, v, cx, 0.8, 0.17, 0.11, 0.05))
    holes = np.minimum(sd_circle(u, v, cx - 0.105, 0.58, 0.072), sd_circle(u, v, cx + 0.105, 0.58, 0.072))
    holes = np.minimum(holes, sd_circle(u, v, cx, 0.7, 0.033))
    for tx in (-0.07, 0.0, 0.07):
        holes = np.minimum(holes, sd_rbox(u, v, cx + tx, 0.88, 0.009, 0.045, 0.004))
    white = np.minimum(bones, np.maximum(skull, -holes))
    return white


def build_flag(acc, info):
    acc.tag = "flag"
    truck = info["trucks"]["main"]
    d = nrm(V3(-0.35, 1.0, 0))
    perp = nrm(np.cross(d, V3(0, 0, 1)))
    NU, NV = 80, 52
    Lf, Hf = 2.4, 1.55
    u = np.linspace(0, Lf, NU)
    v = np.linspace(0, Hf, NV)
    Uu, Vv = np.meshgrid(u, v, indexing="ij")
    top = truck + V3(0, 0, -0.15)
    f = Uu / Lf
    rip = (0.14 * f + 0.02) * np.sin(2 * np.pi * Uu / 0.95 - 2.2 * Vv / Hf) + 0.04 * f * np.sin(2 * np.pi * Uu / 0.45 + 1.0 + Vv)
    droop = 0.18 * f ** 2
    P = (top[None, None, :] + (Uu + 0.07)[..., None] * d[None, None, :]
         - (Vv + droop)[..., None] * V3(0, 0, 1)[None, None, :] + rip[..., None] * perp[None, None, :])
    sd = jolly_roger(Uu, Vv)
    h = 0.022
    wmix = np.clip(0.5 - sd / (2 * h), 0, 1)
    col = FLAGC[None, None, :] * (1 - wmix[..., None]) + BONE[None, None, :] * wmix[..., None]
    extra = np.clip(f ** 0.8, 0, 1) * 255
    UV = np.stack([Uu, Vv], -1)
    add_cloth(acc, P.reshape(-1, 3), grid_tris(NU, NV), UV.reshape(-1, 2), col.reshape(-1, 3), extra.ravel(),
              M_FLAG, perp, thick=0.01)
    # pennant on the fore truck (red streamer)
    ft = info["trucks"]["fore"] + V3(0, 0, -0.1)
    NU2, NV2 = 36, 3
    L2 = 4.2
    u2 = np.linspace(0, L2, NU2)
    v2 = np.linspace(0, 1, NV2)
    U2, V2 = np.meshgrid(u2, v2, indexing="ij")
    hgt = 0.32 * (1 - U2 / L2 * 0.92)
    rp = 0.25 * (U2 / L2) * np.sin(2 * np.pi * U2 / 1.6)
    P2 = ft[None, None, :] + (U2 + 0.05)[..., None] * d + (-(V2 * hgt) - 0.25 * (U2 / L2) ** 2)[..., None] * V3(0, 0, 1) \
        + rp[..., None] * perp
    add_cloth(acc, P2.reshape(-1, 3), grid_tris(NU2, NV2), np.stack([U2, V2 * hgt], -1).reshape(-1, 2),
              hexc("#101010"), (np.clip(U2 / L2, 0, 1) * 255).ravel(), M_FLAG, perp, thick=0.008)


# --- bowsprit -------------------------------------------------------------
def build_bowsprit(acc, info):
    global BS_HEEL
    acc.tag = "spars"
    y0 = 11.8
    H0 = V3(0, y0, float(deck_level(y0)) + 0.45)
    BS_HEEL = H0
    db = V3(0, math.cos(math.radians(24)), math.sin(math.radians(24)))
    Lb = 10.0
    add_tube(acc, [H0 + db * s for s in np.linspace(0, Lb, 7)], np.linspace(0.31, 0.21, 7), sides=12,
             caps=(True, True), col=MAST, mat=M_SPAR)
    J0 = H0 + db * 6.4 + V3(0, 0, 0.36)
    dj = V3(0, math.cos(math.radians(20)), math.sin(math.radians(20)))
    Lj = 7.6
    add_tube(acc, [J0 + dj * s for s in np.linspace(0, Lj, 6)], np.linspace(0.16, 0.09, 6), sides=10,
             caps=(True, True), col=MAST, mat=M_SPAR)
    capc = H0 + db * (Lb - 0.15) + V3(0, 0, 0.2)
    add_box(acc, capc, np.array([[1, 0, 0], db, nrm(np.cross([1, 0, 0], db))]), (0.2, 0.12, 0.3), col=BLACK,
            mat=M_PAINT)
    for s in (2.6, 3.0, 3.4):
        add_tube(acc, [H0 + db * (s - 0.05), H0 + db * (s + 0.05)], 0.31 - 0.01 * s + 0.02, sides=12, col=TAR,
                 mat=M_ROPE)
    info["bs"] = lambda f: H0 + db * Lb * f
    info["jb"] = lambda f: J0 + dj * Lj * f
    # dolphin striker
    ds0 = H0 + db * (Lb - 0.4)
    ds1 = ds0 + V3(0, 0.1, -1.4)
    add_tube(acc, [ds0, ds1], 0.06, sides=6, caps=(False, True), col=MAST, mat=M_SPAR)
    info["dstriker"] = ds1


def build_rigging(acc, info):
    acc.tag = "rigging"
    bs, jb = info["bs"], info["jb"]
    tops = info["tops"]
    rr = 0.045
    # ---- shrouds, ratlines, channels, deadeyes
    for m, M in MASTS.items():
        n = M["nshroud"]
        my = M["y"]
        for side in (1, -1):
            yv = my + 0.35 - 0.62 * np.arange(n)
            if m == "mizzen":
                zch_f = lambda y: float(deck_main(y)) + QD_H + 0.98
            else:
                zch_f = lambda y: float(deck_main(y)) + 1.0
            zch = zch_f(my)
            acc.tag = "hull_trim"
            channel(acc, yv[0] + 0.45, yv[-1] - 0.45, zch, side)
            acc.tag = "rigging"
            head = ax(m, M["top_z"] - 0.5, 0) + V3(side * M["rl"] * 0.85, -0.05, 0)
            uppers = []
            for y in yv:
                L0 = V3(side * (float(hb(y, zch)) + 0.45), y, zch + 0.06)
                dvec = nrm(head - L0)
                d_lo = L0 + dvec * 0.12
                d_up = L0 + dvec * 0.55
                for dc in (d_lo, d_up):
                    add_tube(acc, [dc - dvec * 0.045, dc + dvec * 0.045], 0.12, sides=8, caps=(True, True),
                             col=HULL_D * 0.8, mat=M_HULL)
                for k in (-1, 1):
                    off = nrm(np.cross(dvec, [0, 1, 0])) * 0.05 * k
                    rope(acc, d_lo + off, d_up + off, 0.016, HEMP, 4)
                # chainplate
                hp = V3(side * (float(hb(y, zch - 0.85)) + 0.02), y, zch - 0.85)
                rope(acc, L0, hp, 0.022, IRON, 4, mat=M_IRON)
                rope(acc, d_up, head, rr, TAR, 5)
                uppers.append((d_up, head))
            ratlines(acc, uppers, zch + 0.95, M["top_z"] - 1.2)
            # topmast shrouds + futtocks
            t = tops[m]
            tmh = ax(m, M["tm"][1] - 0.95, 1) + V3(side * M["tm"][2] * 0.85, 0, 0)
            lows = []
            for dy in (0.3, -0.2, -0.7)[: (3 if m != "mizzen" else 2)]:
                lp = V3(side * (t["W"] / 2 - 0.1), t["tc"][1] + dy * t["W"] / 3.4, M["top_z"] + 0.05)
                rope(acc, lp, tmh, 0.032, TAR, 5)
                rope(acc, lp, ax(m, M["top_z"] - 1.7, 0) + V3(side * M["rl"] * 0.8, 0, 0), 0.028, TAR, 4)
                lows.append((lp, tmh))
            ratlines(acc, lows, M["top_z"] + 0.45, M["tm"][1] - 1.7)
            # topgallant shrouds
            if M["tg"]:
                tgh = ax(m, M["tg_rig"], 2) + V3(side * 0.09, 0, 0)
                cc = ax(m, M["ctz"], 1)
                for dy in (0.2, -0.25):
                    rope(acc, cc + V3(side * (M["ctw"] / 2 - 0.05), dy, 0.03), tgh, 0.022, TAR, 4)
            # backstays to the rail
            for k in range(2):
                yb = my - 3.0 - 0.6 * k
                zb = float(z_top(yb))
                p = V3(side * (float(hb(yb, zb)) + 0.12), yb, zb - 0.08)
                rope(acc, tmh + V3(0, -0.05, 0.1), p, 0.035, TAR, 5)
            if M["tg"]:
                yb = my - 4.3
                zb = float(z_top(yb))
                p = V3(side * (float(hb(yb, zb)) + 0.12), yb, zb - 0.08)
                rope(acc, ax(m, M["tg_rig"] - 0.1, 2) + V3(side * 0.08, 0, 0), p, 0.028, TAR, 5)
    # ---- stays
    fm = MASTS["fore"]
    mm = MASTS["main"]
    zm = MASTS["mizzen"]
    fore_low = ax("fore", fm["low_top"] - 0.6, 0) + V3(0, 0.28, 0)
    fore_tm = ax("fore", fm["tm"][1] - 0.85, 1) + V3(0, 0.18, 0)
    fore_tm2 = ax("fore", fm["tm"][1] - 0.5, 1) + V3(0, 0.18, 0)
    fore_tg = ax("fore", fm["tg_rig"], 2) + V3(0, 0.1, 0)
    rope(acc, fore_low, bs(0.62), 0.07, TAR, 6)
    rope(acc, fore_low + V3(0, 0, -0.25), bs(0.58), 0.05, TAR, 5)
    stays = [(fore_tm, bs(0.97)), (fore_tm2, jb(0.6)), (fore_tg, jb(0.98))]
    for a, b in stays:
        rope(acc, a, b, 0.045, TAR, 5)
    info["headstays"] = [
        (stays[0][0], stays[0][1], 0.06, 0.6, 0.86, 0.9, 0.65),
        (stays[1][0], stays[1][1], 0.05, 0.62, 0.72, 1.25, 0.75),
        (stays[2][0], stays[2][1], 0.04, 0.6, 0.6, 1.55, 0.65),
    ]
    rope(acc, ax("main", mm["low_top"] - 0.6, 0) + V3(0, 0.3, 0), ax("fore", fm["zd"] + 0.6, 0) + V3(0, -0.38, 0),
         0.075, TAR, 6)
    rope(acc, ax("main", mm["tm"][1] - 0.85, 1) + V3(0, 0.15, 0), ax("fore", fm["top_z"] - 0.3, 0) + V3(0, -0.3, 0),
         0.05, TAR, 5)
    rope(acc, ax("main", mm["tg_rig"] - 0.1, 2) + V3(0, 0.1, 0), ax("fore", fm["tm"][1] - 0.9, 1) + V3(0, -0.2, 0),
         0.035, TAR, 5)
    rope(acc, ax("mizzen", zm["low_top"] - 0.6, 0) + V3(0, 0.25, 0), ax("main", mm["zd"] + 2.4, 0) + V3(0, -0.4, 0),
         0.06, TAR, 5)
    rope(acc, ax("mizzen", zm["tm"][1] - 0.9, 1) + V3(0, 0.12, 0), ax("main", mm["top_z"] - 0.4, 0) + V3(0, -0.35, 0),
         0.04, TAR, 5)
    # bobstay, martingale, bowsprit shrouds
    stem_pt = V3(0, float(y_stem(0.8)) + 0.35, 0.8)
    rope(acc, bs(0.93), stem_pt, 0.055, TAR, 5)
    rope(acc, bs(0.75), stem_pt + V3(0, -0.1, 0.5), 0.045, TAR, 5)
    rope(acc, jb(0.98), info["dstriker"], 0.035, TAR, 5)
    rope(acc, info["dstriker"], bs(0.6) + V3(0, 0, -0.1), 0.035, TAR, 5)
    for side in (1, -1):
        yy = 14.6
        zz = float(z_top(yy)) - 0.4
        rope(acc, bs(0.8), V3(side * (float(hb(yy, zz)) + 0.05), yy, zz), 0.04, TAR, 5)
        rope(acc, jb(0.95), bs(0.9) + V3(side * 1.2, 0.2, -0.2), 0.022, TAR, 4)
        rope(acc, bs(0.9) + V3(side * 1.2, 0.2, -0.2), V3(side * 1.6, 15.4, float(z_top(15.4)) - 0.2), 0.022, TAR, 4)
    # ---- running rigging on the yards
    acc.tag = "running"
    brace_to = {
        ("fore", 0): lambda s: ax("main", 8.0, 0) + V3(s * 0.3, -0.1, 0),
        ("fore", 1): lambda s: tops["main"]["tc"] + V3(s * 1.1, 0.4, 0.1),
        ("fore", 2): lambda s: ax("main", mm["tm"][1] - 1.0, 1) + V3(s * 0.25, 0, 0),
        ("main", 0): lambda s: V3(s * (float(hb(-7.6, float(z_top(-7.6)))) + 0.1), -7.6, float(z_top(-7.6)) - 0.1),
        ("main", 1): lambda s: tops["mizzen"]["tc"] + V3(s * 0.9, -0.2, 0.1),
        ("main", 2): lambda s: ax("mizzen", zm["tm"][1] - 1.0, 1) + V3(s * 0.15, 0, 0),
        ("mizzen", 0): lambda s: ax("main", 9.5, 0) + V3(s * 0.35, -0.2, 0),
        ("mizzen", 1): lambda s: ax("main", mm["top_z"] - 0.6, 0) + V3(s * 0.4, -0.25, 0),
    }
    for m in YARDS:
        M = MASTS[m]
        for i, (name, z, L, r, lev) in enumerate(YARDS[m]):
            g = yard_geo(m, i)
            lift_to = [ax(m, M["low_top"] - 0.1, 0), ax(m, M["tm"][1] - 0.2, 1),
                       ax(m, (M["tg_rig"] or z + 1) + 0.15, 2)][lev]
            for s in (1, -1):
                arm = g["arm"](s, 0.25)
                rope(acc, arm, lift_to + V3(s * 0.15, 0, 0), 0.022, HEMP, 4)
                rope(acc, arm, brace_to[(m, i)](s), 0.022, HEMP, 4, sag=0.15)
                # footrope
                pts = []
                for t in np.linspace(0, 1, 9):
                    sd = 0.5 + t * (L / 2 - 0.75)
                    p = g["c"] + YDIR * s * sd + V3(0, 0, -0.85 * math.sin(math.pi * min(1, t * 1.15)) - 0.05) \
                        + YFWD * -0.12
                    pts.append(p)
                pts[-1] = g["arm"](s, 0.25) + V3(0, 0, -0.05)
                add_tube(acc, np.array(pts), 0.018, sides=4, col=HEMP, mat=M_ROPE)
            if lev > 0 or m == "mizzen":
                rope(acc, g["c"] + V3(0, -0.12, 0.1), lift_to + V3(0, -0.1, 0), 0.03, HEMP, 4)


def build_sheets(acc, info):
    acc.tag = "running"
    zm = MASTS["mizzen"]
    mm = MASTS["main"]
    # course sheets / tacks
    sails = info["sails"]
    for m in ("fore", "main"):
        S = sails[(m, "course")]
        for s, corner in ((-1, S["BL"]), (1, S["BR"])):
            yt = corner[1] - 3.2
            zt_ = float(z_top(yt))
            rope(acc, corner, V3(s * (float(hb(yt, zt_)) - 0.1), yt, zt_ + 0.05), 0.028, HEMP, 4)
            yt2 = corner[1] + 2.6
            zt2 = float(z_top(yt2))
            rope(acc, corner, V3(s * (float(hb(yt2, zt2)) - 0.1), yt2, zt2 + 0.05), 0.024, HEMP, 4)
    # jib sheets
    for C in info["jib_clews"]:
        yt = C[1] - 2.4
        zt_ = float(z_top(yt))
        rope(acc, C, V3(-(float(hb(yt, zt_)) - 0.1), yt, zt_ + 0.05), 0.022, HEMP, 4, sag=0.12)
    # spanker gear
    sp = info["spanker"]
    zt0 = info["zt_stern"]
    rope(acc, sp["gaff1"], ax("mizzen", zm["tm"][1] - 1.3, 1), 0.025, HEMP, 4)
    rope(acc, sp["gaff0"] + (sp["gaff1"] - sp["gaff0"]) * 0.5, ax("mizzen", zm["low_top"] - 0.2, 0), 0.025, HEMP, 4)
    rope(acc, sp["boom1"], ax("mizzen", zm["low_top"] - 0.3, 0) + V3(0, -0.2, 0), 0.02, HEMP, 4)
    rope(acc, sp["boom1"] + V3(0, 0.6, 0), V3(-0.6, float(y_post(zt0)) + 0.3, zt0 + 0.1), 0.03, HEMP, 4)
    for s in (1, -1):
        x = s * (info["hE_top"] - 0.2)
        rope(acc, sp["gaff1"], V3(x, float(y_post(zt0)) + 0.4, zt0 + 0.1), 0.02, HEMP, 4, sag=0.2)
    # flag halyard
    tr = info["trucks"]["main"]
    rope(acc, tr + V3(0.02, 0.06, 0), ax("main", mm["tg"][0] + 1.0, 2) + V3(0.05, 0.1, 0), 0.012, HEMP, 4)




def ratlines(acc, lines, z0, z1, step=0.4):
    for h in np.arange(z0, z1, step):
        pts = []
        for (a, b) in lines:
            if not (min(a[2], b[2]) < h < max(a[2], b[2])):
                pts = []
                break
            f = (h - a[2]) / (b[2] - a[2])
            pts.append(a + (b - a) * f)
        if len(pts) >= 2:
            for p, q in zip(pts[:-1], pts[1:]):
                rope(acc, p, q, 0.014, TAR, 4)


# ----------------------------------------------------------------------------
# generic smooth primitives (used for the helmsman and the wheel)
# ----------------------------------------------------------------------------
def add_ellipsoid(acc, c, radii, axes=None, col=(1, 1, 1), mat=M_PAINT, rough=None, nu=20, nv=14,
                  bump=0.0, bump_freq=6.0, seed=0, zcut=None):
    """Ellipsoid (optionally bumpy). axes rows = local x, y, z. zcut: keep only local z >= zcut*rz (dome)."""
    c = np.asarray(c, float)
    axes = np.eye(3) if axes is None else np.asarray(axes, float)
    radii = np.asarray(radii, float)
    th0 = 0.0
    th1 = np.pi if zcut is None else math.acos(max(-1.0, min(1.0, zcut)))
    th = np.linspace(th0, th1, nv)
    ph = np.linspace(0, 2 * np.pi, nu + 1)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], -1)
    sc = 1.0
    if bump:
        sc = 1.0 + bump * (2 * fbm(d * bump_freq + seed * 3.1, 3, seed=seed) - 1)
    G = c + (d * radii * (sc[..., None] if bump else 1.0)) @ axes
    T = grid_tris(nv, nu + 1)
    Pf = G.reshape(-1, 3)
    fn = face_normals(Pf, T)
    cen = (Pf[T[:, 0]] + Pf[T[:, 1]] + Pf[T[:, 2]]) / 3
    if np.einsum("ij,ij->i", fn, cen - c).sum() < 0:
        T = T[:, ::-1]
    N = vertex_normals(Pf, T).reshape(G.shape)
    seam = nrm(N[:, 0] + N[:, -1])
    N[:, 0] = seam
    N[:, -1] = seam
    N[0, :] = nrm(N[0, :].sum(0))
    if zcut is None:
        N[-1, :] = nrm(N[-1, :].sum(0))
    UV = np.stack([PH * radii.mean(), TH * radii.mean()], -1)
    add_grid(acc, G, UV=UV, N=N, col=col, mat=mat, rough=rough)


def add_loft(acc, pts, rx, ry, ref, sides=16, col=(1, 1, 1), mat=M_PAINT, rough=None, caps=(False, False),
             a0=0.0, a1=2 * np.pi, double=None, inset=0.012):
    """Elliptic loft along pts. rx along ref, ry along cross(t, ref). a0/a1 may be per-ring arrays
    (open sheet). double = colour of the inner sheet for open sheets."""
    pts = np.asarray(pts, float)
    k = len(pts)
    rx = np.broadcast_to(np.asarray(rx, float), (k,))
    ry = np.broadcast_to(np.asarray(ry, float), (k,))
    a0 = np.broadcast_to(np.asarray(a0, float), (k,))
    a1 = np.broadcast_to(np.asarray(a1, float), (k,))
    t = nrm(np.gradient(pts, axis=0))
    ref = np.asarray(ref, float)
    n = nrm(ref[None, :] - (t @ ref)[:, None] * t)
    b = np.cross(t, n)
    f = np.linspace(0, 1, sides + 1)
    ang = a0[:, None] + (a1 - a0)[:, None] * f[None, :]
    ca, sa = np.cos(ang)[..., None], np.sin(ang)[..., None]
    off = ca * rx[:, None, None] * n[:, None, :] + sa * ry[:, None, None] * b[:, None, :]
    nn = nrm(ca / rx[:, None, None] * n[:, None, :] + sa / ry[:, None, None] * b[:, None, :])
    G = pts[:, None, :] + off
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    UV = np.stack(np.broadcast_arrays(s[:, None], ang * 0.5 * (rx + ry)[:, None]), -1)
    closed = np.allclose(a1 - a0, 2 * np.pi)
    if closed:
        N = nn.copy()
        add_grid(acc, G, UV=UV, N=N, col=col, mat=mat, rough=rough)
    else:
        add_grid(acc, G, UV=UV, N=nn, col=col, mat=mat, rough=rough)
        if double is not None:
            add_grid(acc, G - nn * inset, UV=UV, N=-nn, col=double, mat=mat, rough=rough)
    for e, sgn in ((0, -1), (k - 1, 1)):
        if caps[0 if e == 0 else 1]:
            ring = G[e, :-1] if closed else G[e]
            m = len(ring)
            Pc = np.vstack([pts[e], ring])
            Tc = [(0, 1 + i, 1 + (i + 1) % m) for i in range(m if closed else m - 1)]
            acc.add(Pc, Tc, N=np.tile(sgn * t[e], (len(Pc), 1)), col=col if np.size(col) == 3 else col[0],
                    mat=mat, rough=rough)


# ----------------------------------------------------------------------------
# ship's wheel (rotating part has material.w = 255)
# ----------------------------------------------------------------------------
WHEEL_Y = -11.6          # Blender y of the wheel plane (before the CB shift)
WHEEL_H = 0.95           # axle height above the deck
WHEEL_ROT = math.pi / 10 # spoke phase: handles at 18 + 36k degrees
GRIP_A = math.radians(54)
GRIP_R = 0.575           # hands grip the handle roots just outside the rim
WHEEL_HANDLE_R = 0.665   # radius of the handle grips


def wheel_centre():
    zq = float(deck_level(WHEEL_Y)) + 0.09
    return V3(0, WHEEL_Y, zq + WHEEL_H), zq


def build_helm(acc, info):
    acc.tag = "helm"
    wc, zq = wheel_centre()
    yw = WHEEL_Y
    wood = hexc("#5a3b22")
    # --- stand (static): base plank, two turned posts, drum with tiller ropes, iron axle
    add_box(acc, (0, yw + 0.45, zq + 0.04), np.eye(3), (0.26, 0.42, 0.04), col=HULL_D, mat=M_HULL)
    for yy in (yw + 0.16, yw + 0.74):
        zz = np.linspace(zq + 0.06, wc[2] - 0.15, 9)
        rr = np.interp(zz - zq, [0, 0.1, 0.2, 0.35, 0.55, 0.75, 0.95], [0.11, 0.09, 0.065, 0.08, 0.055, 0.075, 0.09])
        add_tube(acc, [V3(0, yy, z) for z in zz], rr, sides=12, caps=(False, True), col=wood, mat=M_HULL)
    add_tube(acc, [wc + V3(0, 0.1, 0), wc + V3(0, 0.8, 0)], 0.16, sides=16, caps=(True, True), col=wood * 1.1,
             mat=M_HULL)
    for yy in (0.13, 0.77):
        add_tube(acc, [wc + V3(0, yy - 0.025, 0), wc + V3(0, yy + 0.025, 0)], 0.172, sides=16, col=IRON, mat=M_IRON)
    for k in range(5):
        a = np.linspace(0, 2 * np.pi, 13)
        yy = 0.25 + 0.1 * k
        add_tube(acc, wc + np.stack([0.168 * np.cos(a), np.full(13, yy), 0.168 * np.sin(a)], 1), 0.016, sides=4,
                 col=HEMP, mat=M_ROPE)
    add_tube(acc, [wc + V3(0, -0.09, 0), wc + V3(0, 0.85, 0)], 0.03, sides=8, caps=(True, True), col=IRON, mat=M_IRON)
    # --- rotating wheel
    acc.cur_w = 255
    ang = np.linspace(0, 2 * np.pi, 49)
    for rc, w_, h_ in ((0.5, 0.055, 0.07),):
        ring = wc + np.stack([rc * np.sin(ang), np.zeros_like(ang), rc * np.cos(ang)], 1)
        add_sweep(acc, ring, rect(w_, h_), col=wood, mat=M_SPAR, side_hint=(0, 1, 0), caps=False)
    ring = wc + np.stack([0.5 * np.sin(ang), np.full_like(ang, -0.04), 0.5 * np.cos(ang)], 1)
    add_tube(acc, ring, 0.012, sides=5, col=GOLD, mat=M_GOLD)
    for k in range(10):
        a = 2 * np.pi * k / 10 + WHEEL_ROT
        d = V3(math.sin(a), 0, math.cos(a))
        # spoke inside the rim
        tt = np.linspace(0.09, 0.47, 6)
        add_tube(acc, [wc + d * t_ for t_ in tt], [0.03, 0.022, 0.028, 0.02, 0.022, 0.024], sides=8, col=wood,
                 mat=M_SPAR)
        # turned handle outside the rim
        tt = np.linspace(0.52, 0.775, 9)
        rr = np.interp(tt, [0.52, 0.56, 0.6, 0.64, 0.67, 0.7, 0.735, 0.76, 0.775],
                       [0.026, 0.02, 0.03, 0.026, 0.031, 0.026, 0.024, 0.02, 0.008])
        add_tube(acc, [wc + d * t_ for t_ in tt], rr, sides=8, col=wood * 1.15, mat=M_SPAR)
    # brass hub with bosses
    add_tube(acc, [wc + V3(0, -0.075, 0), wc + V3(0, -0.06, 0), wc + V3(0, 0.06, 0), wc + V3(0, 0.075, 0)],
             [0.07, 0.1, 0.1, 0.07], sides=16, caps=(True, True), col=GOLD, mat=M_GOLD)
    add_tube(acc, [wc + V3(0, -0.11, 0), wc + V3(0, -0.075, 0)], [0.035, 0.05], sides=12, caps=(True, False),
             col=GOLD, mat=M_GOLD)
    acc.cur_w = 0
    # binnacle ahead of the stand
    yb = yw + 1.45
    add_box(acc, (0, yb, zq + 0.5), np.eye(3), (0.28, 0.28, 0.5), col=HULL * 1.3, mat=M_HULL)
    add_tube(acc, [V3(0, yb, zq + 1.0), V3(0, yb, zq + 1.2), V3(0, yb, zq + 1.3)], [0.22, 0.22, 0.05], sides=12,
             caps=(True, False), col=GOLD, mat=M_GOLD)
    info["wheel_centre"] = wc
    info["deck_helm"] = zq


# ----------------------------------------------------------------------------
# the helmsman (material.w = 200)
# ----------------------------------------------------------------------------
SKIN = hexc("#a86e4b")
SKIN_D = hexc("#8e5a3c")
COAT = hexc("#4d1611")
COAT_IN = hexc("#2a0e0b")
SHIRT = hexc("#cfc5b0")
TROUSER = hexc("#2b231c")
BOOT = hexc("#1a110b")
BELT = hexc("#25170e")
HATC = hexc("#141110")
BEARD = hexc("#140f0c")
HAIRC = hexc("#1a130f")


def solve_elbow(S, W, L1, L2, pole):
    d = np.linalg.norm(W - S)
    d = min(d, L1 + L2 - 0.005)
    u = nrm(W - S)
    a = (L1 * L1 - L2 * L2 + d * d) / (2 * d)
    h = math.sqrt(max(L1 * L1 - a * a, 0))
    pp = nrm(pole - np.dot(pole, u) * u)
    return S + u * a + pp * h, S + u * d


def arm_follow_weights(acc, part0, chain):
    """material.z (extra) = arm-follow weight: 0 at the shoulder, ~60 mid upper arm, ~120 elbow,
    ~180 mid forearm, 255 on wrist/hand. Arc-length parameter of the nearest point on the chain."""
    chain = [np.asarray(c, float) for c in chain]
    seg_len = [np.linalg.norm(b - a) for a, b in zip(chain[:-1], chain[1:])]
    cum = np.concatenate([[0], np.cumsum(seg_len)])
    L1, L2 = cum[1], cum[2] - cum[1]
    knots_s = [0.0, 0.07, 0.5 * L1, L1, L1 + 0.5 * L2, cum[2] - 0.03, cum[2] + 0.02]
    knots_w = [0.0, 0.0, 60.0, 120.0, 180.0, 240.0, 255.0]
    for pi in range(part0, len(acc.parts)):
        P, N, UV, C, M, R, E, W, T = acc.parts[pi]
        best_d = np.full(len(P), 1e9)
        best_s = np.zeros(len(P))
        for k, (a, b) in enumerate(zip(chain[:-1], chain[1:])):
            ab = b - a
            t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
            d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
            better = d < best_d
            best_d[better] = d[better]
            best_s[better] = cum[k] + t[better] * seg_len[k]
        E2 = np.interp(best_s, knots_s, knots_w)
        acc.parts[pi] = (P, N, UV, C, M, R, E2, W, T)


def build_pirate(acc, info):
    acc.tag = "pirate"
    acc.cur_w = 200
    wc = info["wheel_centre"]
    zq = info["deck_helm"]
    O = V3(0, WHEEL_Y - 0.40, zq)          # foot pivot: between the feet, on the deck
    info["pirate_pivot"] = O

    def L(x, y, z):
        return O + V3(x, y, z)
    # ---- legs and boots
    for s in (1, -1):
        hip = L(s * 0.095, 0.0, 0.93)
        knee = L(s * 0.11, 0.045, 0.5)
        ank = L(s * 0.12, 0.0, 0.1)
        pts = [hip + (knee - hip) * t for t in np.linspace(-0.1, 1.15, 8)]
        add_loft(acc, pts, np.linspace(0.088, 0.06, 8), np.linspace(0.095, 0.066, 8), (1, 0, 0), sides=14,
                 col=TROUSER, rough=0.85)
        pts = [L(s * 0.112, 0.04, z) for z in np.linspace(0.6, 0.12, 8)]
        pts = [p + V3(0, -0.04 * (0.6 - p[2] + O[2]) / 0.5, 0) for p in pts]
        rr = np.interp(np.linspace(0, 1, 8), [0, 0.12, 0.25, 0.6, 1], [0.088, 0.085, 0.07, 0.06, 0.05])
        add_loft(acc, pts, rr, rr * 1.05, (1, 0, 0), sides=14, col=BOOT, rough=0.45, caps=(True, False))
        add_loft(acc, [L(s * 0.112, 0.04, z) for z in (0.6, 0.52)], [0.093, 0.09], [0.1, 0.097], (1, 0, 0),
                 sides=14, col=BOOT * 1.3, rough=0.5)
        fp = [L(s * 0.122, y, 0.0) for y in np.linspace(-0.08, 0.2, 8)]
        hw = np.interp(np.linspace(0, 1, 8), [0, 0.2, 0.6, 0.85, 1], [0.04, 0.05, 0.05, 0.04, 0.02])
        hh = np.interp(np.linspace(0, 1, 8), [0, 0.25, 0.5, 0.85, 1], [0.07, 0.07, 0.05, 0.04, 0.02])
        fp = [p + V3(0, 0, h_) for p, h_ in zip(fp, hh)]
        add_loft(acc, fp, hw, hh, (1, 0, 0), sides=14, col=BOOT, rough=0.45, caps=(True, True))
    add_ellipsoid(acc, L(0, -0.005, 0.93), (0.165, 0.115, 0.1), col=TROUSER, rough=0.85)
    # ---- torso (coat), shirt V, belt, buttons
    zz = np.linspace(0.93, 1.47, 10)
    tx = np.interp(zz, [0.93, 1.0, 1.12, 1.27, 1.4, 1.47], [0.165, 0.152, 0.16, 0.18, 0.19, 0.15])
    ty = np.interp(zz, [0.93, 1.0, 1.12, 1.27, 1.4, 1.47], [0.118, 0.11, 0.12, 0.13, 0.115, 0.085])
    tc = [L(0, 0.012 * (z - 0.93), z) for z in zz]
    add_loft(acc, tc, tx, ty, (1, 0, 0), sides=22, col=COAT, rough=0.85, caps=(False, True))
    gam = np.radians(np.interp(zz, [0.93, 1.02, 1.47], [3, 6, 34]))
    sel = zz > 1.0
    add_loft(acc, np.array(tc)[sel], tx[sel] + 0.007, ty[sel] + 0.007, (1, 0, 0), sides=8, col=SHIRT, rough=0.9,
             a0=np.pi / 2 - gam[sel], a1=np.pi / 2 + gam[sel])
    for sgn in (1, -1):
        edge = [np.array(tc)[i] + V3(sgn * math.cos(np.pi / 2 - gam[i]) * (tx[i] + 0.012), math.sin(np.pi / 2 - gam[i]) * (ty[i] + 0.012), 0)
                for i in range(len(zz)) if zz[i] > 1.02]
        add_tube(acc, edge, 0.012, sides=6, col=COAT * 1.25, mat=M_PAINT, rough=0.8)
        for e in edge[1::2]:
            add_ellipsoid(acc, e + V3(sgn * 0.022, 0.006, 0), (0.012, 0.008, 0.012), col=GOLD, mat=M_GOLD, nu=8, nv=6)
    add_loft(acc, [L(0, 0.0, 0.955), L(0, 0.0, 1.02)], [0.172, 0.165], [0.127, 0.12], (1, 0, 0), sides=22,
             col=BELT, rough=0.55)
    add_box(acc, L(0, 0.13, 0.988), np.eye(3), (0.05, 0.012, 0.04), col=GOLD, mat=M_GOLD)
    add_box(acc, L(0, 0.137, 0.988), np.eye(3), (0.028, 0.008, 0.02), col=BELT, mat=M_PAINT)
    # sash/baldric across the chest
    bal = [L(0.16 * math.cos(t), 0.125 * math.sin(t) + 0.004, 1.0 + 0.42 * (t - 0.35) / 2.6) for t in np.linspace(0.35, 2.95, 10)]
    add_tube(acc, bal, 0.022, sides=6, col=BELT * 1.2, mat=M_PAINT, rough=0.55)
    # ---- coat skirt (open at the front, lined)
    zz = np.linspace(0.97, 0.38, 9)
    sx = np.interp(zz, [0.38, 0.6, 0.97], [0.29, 0.24, 0.17])
    sy = np.interp(zz, [0.38, 0.6, 0.97], [0.23, 0.19, 0.125])
    gam = np.radians(np.interp(zz, [0.38, 0.7, 0.97], [42, 30, 12]))
    sc = [L(0, -0.03 * (0.97 - z), z) for z in zz]
    add_loft(acc, sc, sx, sy, (1, 0, 0), sides=26, col=COAT, rough=0.85, a0=np.pi / 2 + gam,
             a1=np.pi / 2 + 2 * np.pi - gam, double=COAT_IN, inset=0.01)
    hem = [sc[-1] + V3(sx[-1] * math.cos(a), sy[-1] * math.sin(a), 0)
           for a in np.linspace(np.pi / 2 + gam[-1], np.pi / 2 + 2 * np.pi - gam[-1], 30)]
    add_tube(acc, hem, 0.012, sides=5, col=GOLD_D, mat=M_GOLD)
    # ---- neck, cravat, head
    add_loft(acc, [L(0, 0.01, z) for z in (1.43, 1.5, 1.58)], [0.052, 0.05, 0.05], [0.055, 0.052, 0.05], (1, 0, 0),
             sides=14, col=SKIN, rough=0.6)
    add_ellipsoid(acc, L(0, 0.065, 1.465), (0.055, 0.035, 0.04), col=SHIRT, rough=0.9, nu=12, nv=8)
    H = L(0, 0.012, 1.665)
    add_ellipsoid(acc, H, (0.084, 0.097, 0.115), col=SKIN, rough=0.6, nu=24, nv=16)
    add_ellipsoid(acc, H + V3(0, 0.088, -0.008), (0.018, 0.03, 0.032),
                  axes=frame_axes((1, 0, 0), (0, 0.35, 1)), col=SKIN_D, rough=0.6, nu=10, nv=8)
    for s in (1, -1):
        add_ellipsoid(acc, H + V3(s * 0.078, -0.01, 0.0), (0.012, 0.022, 0.032), col=SKIN_D, rough=0.6, nu=8, nv=6)
        add_tube(acc, [H + V3(s * 0.013, 0.09, 0.047), H + V3(s * 0.035, 0.087, 0.052), H + V3(s * 0.058, 0.074, 0.046)],
                 0.009, sides=5, col=BEARD, mat=M_ROPE, rough=0.95)
    add_ellipsoid(acc, H + V3(0.033, 0.079, 0.025), (0.012, 0.01, 0.009), col=hexc("#0e0b09"), rough=0.3, nu=8, nv=6)
    pa = frame_axes((0.94, 0.33, 0), (0, 0, 1))
    add_ellipsoid(acc, H + V3(-0.034, 0.086, 0.024), (0.027, 0.009, 0.022), axes=pa, col=hexc("#0b0908"), rough=0.5,
                  nu=10, nv=6)
    phi = np.linspace(-np.pi, np.pi, 33)
    sz = 0.03 + 0.03 * np.sin(phi + 0.0)
    strap = []
    for f_, z_ in zip(phi, sz):
        zz_ = z_
        k = math.sqrt(max(1 - (zz_ / 0.112) ** 2, 0))
        strap.append(H + V3(-0.083 * k * math.sin(f_), 0.098 * k * math.cos(f_), zz_))
    add_tube(acc, strap, 0.005, sides=4, col=hexc("#0b0908"), mat=M_PAINT, rough=0.5)
    # ---- hair, beard, moustache, braids
    add_ellipsoid(acc, H + V3(0, -0.05, -0.07), (0.088, 0.07, 0.13), col=HAIRC, mat=M_ROPE, rough=0.95, bump=0.08,
                  bump_freq=9, seed=3)
    add_ellipsoid(acc, H + V3(0, 0.045, -0.118), (0.097, 0.074, 0.088), col=BEARD, mat=M_ROPE, rough=0.95,
                  bump=0.1, bump_freq=8, seed=5, nu=26, nv=16)
    add_ellipsoid(acc, H + V3(0, 0.068, -0.215), (0.09, 0.06, 0.12), col=BEARD, mat=M_ROPE, rough=0.95,
                  bump=0.14, bump_freq=7, seed=6, nu=24, nv=16)
    for s in (1, -1):
        add_ellipsoid(acc, H + V3(s * 0.072, 0.01, -0.075), (0.028, 0.048, 0.055), col=BEARD, mat=M_ROPE, rough=0.95,
                      bump=0.1, bump_freq=9, seed=7 + s)
    must = [H + V3(-0.07, 0.085, -0.075), H + V3(-0.035, 0.112, -0.03), H + V3(0, 0.122, -0.022),
            H + V3(0.035, 0.112, -0.03), H + V3(0.07, 0.085, -0.075)]
    add_tube(acc, must, [0.01, 0.017, 0.016, 0.017, 0.01], sides=8, col=BEARD, mat=M_ROPE, rough=0.95)
    for s in (1, -1):
        bt = np.linspace(0, 1, 14)
        bp = [H + V3(s * (0.035 - 0.01 * t), 0.115 + 0.01 * t, -0.27 - 0.17 * t) for t in bt]
        br = 0.017 * (1 - 0.4 * bt) * (1 + 0.25 * np.abs(np.sin(bt * 22)))
        add_tube(acc, bp, br, sides=8, col=BEARD, mat=M_ROPE, rough=0.95, caps=(False, True))
        e = bp[-2]
        add_tube(acc, [e + V3(0, 0, 0.02), e + V3(0, 0, -0.01)], 0.016, sides=8, caps=(True, True), col=GOLD,
                 mat=M_GOLD)
    # ---- tricorn hat
    hc = H + V3(0, -0.006, 0.075)
    add_ellipsoid(acc, hc, (0.096, 0.108, 0.085), col=HATC, rough=0.9, zcut=-0.15, nu=24, nv=10)
    nphi = 72
    phs = np.linspace(0, 2 * np.pi, nphi + 1)
    Rc = 0.29
    corners = np.radians([90, 210, 330])
    rows = 7
    G = np.zeros((nphi + 1, rows, 3))
    top = []
    for i, f_ in enumerate(phs):
        rel = ((f_ - corners[0]) % (2 * np.pi / 3)) - np.pi / 3
        rtri = Rc * 0.5 / max(math.cos(rel), 0.5)
        cornerness = math.cos(rel * 3) * 0.5 + 0.5      # 1 at corners
        rtri = rtri * (1 - 0.08 * cornerness ** 8)
        hup = 0.035 + 0.075 * (1 - cornerness) ** 0.7
        rin_x, rin_y = 0.1, 0.112
        for j, t in enumerate(np.linspace(0, 1, rows)):
            px = lerp(rin_x * math.cos(f_), rtri * math.cos(f_), t)
            py = lerp(rin_y * math.sin(f_), rtri * math.sin(f_), t)
            G[i, j] = hc + V3(px, py, -0.065 + hup * t ** 1.8)
        top.append(G[i, -1])
    T = grid_tris(nphi + 1, rows)
    Pf = G.reshape(-1, 3)
    fn = face_normals(Pf, T)
    if fn.sum(0)[2] < 0:
        T = T[:, ::-1]
    Nn = vertex_normals(Pf, T)
    acc.add(Pf + Nn * 0.004, T, N=Nn, col=HATC, mat=M_PAINT, rough=0.9)
    acc.add(Pf - Nn * 0.004, T[:, ::-1], N=-Nn, col=HATC, mat=M_PAINT, rough=0.9)
    add_tube(acc, np.array(top), 0.009, sides=5, col=GOLD, mat=M_GOLD)
    # ---- arms reaching the handles
    info["eye"] = H + V3(0, 0.088, 0.025)
    for s in (1, -1):
        part0 = len(acc.parts)
        a = s * GRIP_A
        rdir = V3(math.sin(a), 0, math.cos(a))
        hd = wc + rdir * GRIP_R
        palm = hd + V3(0, -0.045, 0) + rdir * 0.0
        S = L(s * 0.185, 0.0, 1.415)
        Wr = palm - nrm(palm - S) * 0.075
        E, Wr2 = solve_elbow(S, Wr, 0.3, 0.28, V3(s * 0.3, -0.55, -0.8))
        if np.linalg.norm(Wr2 - Wr) > 0.02:
            log("WARN arm cannot reach: short by %.3f" % np.linalg.norm(Wr2 - Wr))
        add_ellipsoid(acc, S + V3(0, 0, 0.005), (0.078, 0.078, 0.072), col=COAT, rough=0.85, nu=16, nv=10)
        up = [S + (E - S) * t for t in np.linspace(0, 1, 6)]
        lo = [E + (Wr - E) * t for t in np.linspace(0, 1, 6)][1:]
        pts = np.array(up + lo)
        rr = np.concatenate([np.linspace(0.066, 0.054, 6), np.linspace(0.052, 0.045, 5)])
        add_tube(acc, pts, rr, sides=14, col=COAT, mat=M_PAINT, rough=0.85)
        add_ellipsoid(acc, E, (0.055, 0.055, 0.055), col=COAT, rough=0.85, nu=12, nv=8)
        fu = nrm(Wr - E)
        add_tube(acc, [Wr - fu * 0.14, Wr - fu * 0.12, Wr - fu * 0.025, Wr - fu * 0.01], [0.05, 0.07, 0.07, 0.05],
                 sides=14, col=COAT * 1.15, mat=M_PAINT, rough=0.8)
        for k in (0.05, 0.09):
            ob = nrm(np.cross(fu, V3(0, 0, 1))) * s
            add_ellipsoid(acc, Wr - fu * k + ob * 0.071, (0.011, 0.011, 0.011), col=GOLD, mat=M_GOLD, nu=8, nv=6)
        add_tube(acc, [Wr - fu * 0.015, Wr + fu * 0.012], [0.05, 0.056], sides=12, col=SHIRT, mat=M_PAINT,
                 rough=0.9, caps=(False, True))
        add_tube(acc, [Wr, palm], [0.033, 0.04], sides=12, col=SKIN, mat=M_PAINT, rough=0.6)
        hax = frame_axes(palm - Wr, rdir)
        add_ellipsoid(acc, palm, (0.048, 0.03, 0.045), axes=hax, col=SKIN, rough=0.6, nu=12, nv=8)
        # curled fingers around the handle
        e1 = V3(0, -1, 0)
        e2 = nrm(np.cross(rdir, e1))
        fing = [hd + 0.038 * (math.cos(t) * e1 + math.sin(t) * e2) for t in np.linspace(-0.6 * s, 0.6 * s + 4.2 * s, 12)]
        for off in (-0.022, 0.0, 0.022):
            add_tube(acc, [f + rdir * off for f in fing], 0.0145, sides=8, col=SKIN, mat=M_PAINT, rough=0.6,
                     caps=(True, True))
        th = [palm + rdir * 0.035, hd + rdir * 0.04 + V3(0, -0.01, 0) + e2 * 0.025 * s]
        add_tube(acc, th, 0.016, sides=8, col=SKIN, mat=M_PAINT, rough=0.6, caps=(True, True))
        arm_follow_weights(acc, part0, [S, E, Wr, palm, hd])
    acc.cur_w = 0


# ----------------------------------------------------------------------------
# weathering pass on the final vertex colours
# ----------------------------------------------------------------------------
ALGAE = hexc("#3b4524")
ALGAE_B = hexc("#4a3f25")
SALT = hexc("#8e897d")
RUST = hexc("#5a2e17")
TARS = hexc("#231c15")


def weather(P, N, C, M, info):
    C = C.copy()
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    ax_ = np.abs(x)
    hbv = hb(y, z)
    inlen = (y > y_post(z) - 0.8) & (y < y_stem(z) + 2.2)
    hullish = np.isin(M, [M_HULL, M_PAINT, M_COPPER, M_GOLD]) & inlen
    skip = np.zeros(len(P), bool)
    if "pirate_pivot" in info:
        skip = np.linalg.norm(P - info["pirate_pivot"] - V3(0, 0, 1), axis=1) < 1.4
    outer = hullish & (ax_ >= hbv - 0.08) & (np.abs(N[:, 0]) > 0.2) & (z < z_top(y) + 0.25) & ~skip
    transom = hullish & (y < y_post(z) + 0.12) & (np.abs(N[:, 1]) > 0.5) & ~skip
    ext = outer | transom
    p3 = np.stack([x * 0.0 + ax_, y, z], 1)
    # general grime and darker weathered planks on the outer hull
    g = fbm(p3 * 0.8, 4, seed=21)
    C[ext] *= (0.78 + 0.32 * g[ext])[:, None]
    wood = ext & (M == M_HULL)
    plank = np.floor((z + 3.5) / 0.25).astype(np.int64)
    seg = np.floor((y + 6 * hash1(plank, seed=3)) / 5.0).astype(np.int64)
    f = hash1(plank, seg, seed=4)
    C[wood] *= (0.8 + 0.32 * f[wood])[:, None]
    # grime / rust streaks below gunports, scuppers and chainplates
    src = []
    rng = np.random.default_rng(77)
    for yp, dh, sc in PORTS:
        src.append((yp, float(deck_main(yp)) + (dh or 0) + 0.4 * sc, rng.uniform(0.12, 0.2), rng.uniform(1.6, 2.6), 0))
        src.append((yp + rng.uniform(-0.25, 0.25), float(deck_main(yp)) + (dh or 0) + 0.4 * sc,
                    rng.uniform(0.06, 0.1), rng.uniform(0.8, 1.8), 1))
    for ys_ in np.arange(-12.5, 12.0, 1.55):
        ys2 = ys_ + rng.uniform(-0.3, 0.3)
        src.append((ys2, float(deck_level(ys2)) + 0.02, rng.uniform(0.08, 0.16), rng.uniform(1.0, 2.2), 0))
    for m, Mm in MASTS.items():
        for k in range(Mm["nshroud"]):
            yv = Mm["y"] + 0.35 - 0.62 * k
            zc = float(deck_main(yv)) + (QD_H if m == "mizzen" else 0) + 0.15
            src.append((yv, zc, 0.07, rng.uniform(0.8, 1.4), 1))
    dark = np.zeros(len(P))
    rust = np.zeros(len(P))
    idx = np.nonzero(outer & (z > -0.3))[0]
    yo, zo = y[idx], z[idx]
    wob = 0.04 * np.sin(zo * 5.0 + yo)
    for (ys_, zs_, w_, L_, kind) in src:
        dz = zs_ - zo
        s_ = np.exp(-((yo - ys_ + wob) / w_) ** 2) * np.where(dz > 0, np.exp(-dz / L_), 0) * (1 - np.exp(-dz / 0.05))
        if kind:
            rust[idx] = np.maximum(rust[idx], s_)
        else:
            dark[idx] = np.maximum(dark[idx], s_)
    C *= (1 - 0.55 * dark)[:, None]
    C += (RUST - C) * (0.55 * rust)[:, None]
    # worn paint on the red strake (and other red paint)
    red = (M == M_PAINT) & (C[:, 0] > 2.3 * C[:, 1] + 0.02) & ~skip
    wm = smoothstep(0.5, 0.72, fbm(p3 * 2.4, 4, seed=5))
    C[red] += ((HULL * 0.85 - C[red]) * (0.85 * wm[red])[:, None])
    C[red] *= 0.85
    # algae / weed band around the waterline (export y -0.4 .. +0.35)
    nwl = fbm(np.stack([y * 0.9, ax_ * 0.5, np.zeros_like(y)], 1), 3, seed=8)
    top = 0.35 + 0.12 * (nwl - 0.5) * 2
    bot = -0.4 - 0.1 * (nwl - 0.5) * 2
    wa = smoothstep(top + 0.06, top - 0.06, z) * smoothstep(bot - 0.08, bot + 0.06, z)
    alg = (np.isin(M, [M_HULL, M_PAINT, M_COPPER]) & inlen & ~skip) & ((ax_ >= hbv - 0.12) | transom | (ax_ < 0.4))
    tone = fbm(p3 * 3.0, 3, seed=9)
    acol = ALGAE[None, :] * (1 - tone[:, None]) + ALGAE_B[None, :] * tone[:, None]
    k = (wa * (0.55 + 0.4 * tone))[alg]
    C[alg] += (acol[alg] - C[alg]) * k[:, None]
    # salt crust near the upper edges and on upward-facing outer ledges
    sh = smoothstep(z_top(y) - 0.4, z_top(y) + 0.05, z)
    ledge = (N[:, 2] > 0.55) & (ax_ >= hbv - 0.35)
    sm = np.isin(M, [M_HULL, M_PAINT]) & inlen & ~skip & (outer | ledge | transom)
    sk = (np.maximum(sh * 0.7, ledge * 0.55) * (0.35 + 0.65 * fbm(p3 * 3.5, 3, seed=13)))[sm]
    C[sm] += (SALT - C[sm]) * (0.6 * sk)[:, None]
    # deck: plank tone variation and tar stains
    dk = (M == M_DECK) & (N[:, 2] > 0.7) & ~skip
    pl = np.floor(x / 0.25).astype(np.int64)
    sg = np.floor((y + 5 * hash1(pl, seed=31)) / 6.0).astype(np.int64)
    fv = hash1(pl, sg, seed=32)
    C[dk] *= (0.68 + 0.3 * fv[dk])[:, None]
    ts = smoothstep(0.53, 0.63, fbm(np.stack([x * 0.9, y * 0.9, z * 0.0], 1), 4, seed=33))
    C[dk] += (TARS - C[dk]) * (0.8 * ts[dk])[:, None]
    return np.clip(C, 0, 1)


# ----------------------------------------------------------------------------
# hydrostatics + reporting
# ----------------------------------------------------------------------------
def hydrostatics():
    zs = np.linspace(ZK, 0.0, 120)
    ys = np.linspace(-16.5, 16.5, 1200)
    Yg, Zg = np.meshgrid(ys, zs, indexing="ij")
    W = 2 * hb(Yg, Zg)
    dA = (ys[1] - ys[0]) * (zs[1] - zs[0])
    vol = W.sum() * dA
    # keel timber
    vol_k = 0.36 * 0.35 * (float(y_stem(ZK)) - float(y_post(ZK)))
    cb_y = (W * Yg).sum() * dA / vol
    cb_z = (W * Zg).sum() * dA / vol
    wpa = 2 * hb(ys, 0.0).sum() * (ys[1] - ys[0])
    return dict(vol=vol + vol_k, cb_y=cb_y, cb_z=cb_z, wpa=wpa)


# ----------------------------------------------------------------------------
# AO via BVH ray casts
# ----------------------------------------------------------------------------
def compute_ao(P, N, T, M, K=48, D=4.0):
    t0 = time.time()
    bvh = BVHTree.FromPolygons(P.tolist(), T.tolist(), all_triangles=True)
    gold = math.pi * (3 - math.sqrt(5))
    i = np.arange(K)
    r = np.sqrt((i + 0.5) / K)
    phi = i * gold
    loc = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - r * r)], 1)
    a = np.where(np.abs(N[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = nrm(np.cross(N, a))
    t2 = np.cross(N, t1)
    rot = (np.arange(len(P)) * 2.399963) % (2 * np.pi)
    c, s = np.cos(rot)[:, None], np.sin(rot)[:, None]
    t1r = c * t1 + s * t2
    t2r = -s * t1 + c * t2
    ao = np.ones(len(P))
    rope_mask = (M == M_ROPE)
    for v in range(len(P)):
        k = 10 if rope_mask[v] else K
        o = Vector(P[v] + N[v] * 0.1)
        dirs = loc[:k, 0:1] * t1r[v] + loc[:k, 1:2] * t2r[v] + loc[:k, 2:3] * N[v]
        occ = 0.0
        for dvec in dirs:
            hit = bvh.ray_cast(o, Vector(dvec), D)
            if hit[0] is not None:
                occ += 1.0 - 0.25 * (hit[3] / D)
        ao[v] = 1.0 - occ / k
    log("AO: %d verts in %.1fs" % (len(P), time.time() - t0))
    ao = np.clip(ao, 0, 1) ** 1.8
    ao[M == M_GLASS] = 1.0
    ao[rope_mask] = np.maximum(ao[rope_mask], 0.75)
    return ao


# ----------------------------------------------------------------------------
# export
# ----------------------------------------------------------------------------
HEADER_EXTRA = [0.0, 0.0, 0.0, 0.0]   # eye x, y, z (export space), neck-bottom height above the foot pivot
NECK_BOTTOM = 1.43                     # local height (m) above the helmsman's foot pivot


def export_bin(path, P, N, UV, C, AO, M, R, E, W, T):
    n = len(P)
    Pe = np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1).astype("<f4")
    Ne = np.stack([N[:, 0], N[:, 2], -N[:, 1]], 1)
    nb = np.clip(np.round(nrm(Ne) * 127), -127, 127).astype(np.int8)
    dt = np.dtype([("pos", "<f4", 3), ("nrm", "i1", 4), ("uv", "<f2", 2), ("col", "u1", 4), ("mat", "u1", 4)])
    assert dt.itemsize == 28
    Vb = np.zeros(n, dt)
    Vb["pos"] = Pe
    Vb["nrm"][:, :3] = nb
    Vb["uv"] = UV.astype("<f2")
    Vb["col"][:, :3] = np.clip(np.round(C * 255), 0, 255).astype(np.uint8)
    Vb["col"][:, 3] = np.clip(np.round(AO * 255), 0, 255).astype(np.uint8)
    Vb["mat"][:, 0] = M.astype(np.uint8)
    Vb["mat"][:, 1] = np.clip(np.round(R * 255), 0, 255).astype(np.uint8)
    Vb["mat"][:, 2] = np.clip(np.round(E), 0, 255).astype(np.uint8)
    Vb["mat"][:, 3] = np.clip(np.round(W), 0, 255).astype(np.uint8)
    bmin = Pe.min(0)
    bmax = Pe.max(0)
    hdr = b"SHP1" + struct.pack("<III", n, T.size, 0) + struct.pack("<6f", *bmin, *bmax) \
        + struct.pack("<4f", *HEADER_EXTRA) + struct.pack("<2f", 0.0, 0.0)
    assert len(hdr) == 64
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr)
        f.write(Vb.tobytes())
        f.write(T.astype("<u4").tobytes())
    return bmin, bmax


# ----------------------------------------------------------------------------
# Blender preview scene
# ----------------------------------------------------------------------------
def srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def make_mesh_object(P, N, UV, C, AO, M, R, E, W, T):
    me = bpy.data.meshes.new("ShipMesh")
    n, m = len(P), len(T)
    me.vertices.add(n)
    me.vertices.foreach_set("co", P.astype(np.float32).ravel())
    me.loops.add(3 * m)
    me.loops.foreach_set("vertex_index", T.astype(np.int32).ravel())
    me.polygons.add(m)
    me.polygons.foreach_set("loop_start", np.arange(0, 3 * m, 3, dtype=np.int32))
    me.update(calc_edges=True)
    lin = srgb_to_lin(C)
    a = me.attributes.new("albedo", "FLOAT_COLOR", "POINT")
    a.data.foreach_set("color", np.concatenate([lin, np.ones((n, 1))], 1).astype(np.float32).ravel())
    metal = np.select([M == M_GOLD, M == M_COPPER, M == M_IRON], [1.0, 0.8, 0.5], 0.0)
    emit = np.where(M == M_GLASS, E / 255.0, 0.0)
    transl = np.select([M == M_SAIL, M == M_FLAG], [0.3, 0.12], 0.0)
    for name, arr in (("ao", AO), ("rough", R), ("metal", metal), ("emit", emit), ("transl", transl),
                      ("matid", M.astype(float)), ("extra", E / 255.0), ("wflag", W)):
        at = me.attributes.new(name, "FLOAT", "POINT")
        at.data.foreach_set("value", arr.astype(np.float32))
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", UV[T.ravel()].astype(np.float32).ravel())
    try:
        me.shade_smooth()
    except Exception:
        me.polygons.foreach_set("use_smooth", np.ones(m, bool))
    me.normals_split_custom_set_from_vertices([tuple(v) for v in N.astype(float)])
    ob = bpy.data.objects.new("PirateShip", me)
    bpy.context.scene.collection.objects.link(ob)
    me.materials.append(make_ship_material())
    return ob


def make_ship_material():
    mat = bpy.data.materials.new("ShipVertexColor")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes["Principled BSDF"]

    def attr(name):
        a = nodes.new("ShaderNodeAttribute")
        a.attribute_type = "GEOMETRY"
        a.attribute_name = name
        return a
    alb = attr("albedo")
    links.new(alb.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(attr("rough").outputs["Fac"], bsdf.inputs["Roughness"])
    links.new(attr("metal").outputs["Fac"], bsdf.inputs["Metallic"])
    links.new(alb.outputs["Color"], bsdf.inputs["Emission Color"])
    mul = nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 12.0
    links.new(attr("emit").outputs["Fac"], mul.inputs[0])
    links.new(mul.outputs[0], bsdf.inputs["Emission Strength"])
    links.new(attr("transl").outputs["Fac"], bsdf.inputs["Transmission Weight"])
    try:
        bsdf.inputs["Thin Wall"].default_value = True
    except Exception:
        pass
    return mat


def setup_scene(info):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = "GPU"
    except Exception as ex:
        log("GPU setup failed", ex)
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.resolution_x = 1280
    sc.render.resolution_y = 720
    sc.render.film_transparent = False
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Medium High Contrast"
    except Exception as ex:
        log("colour mgmt", ex)
    sc.view_settings.exposure = 0.0
    # world: physical sky, low sun
    w = bpy.data.worlds.new("SunsetSky")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.radians(4.0)
    sky.sun_rotation = math.radians(SUN_ROT_DEG)
    sky.sun_disc = True
    sky.altitude = 10.0
    try:
        sky.aerosol_density = 2.2
        sky.air_density = 1.2
    except Exception:
        pass
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.35
    # ocean
    water = bpy.data.materials.new("Water")
    water.use_nodes = True
    wb = water.node_tree.nodes["Principled BSDF"]
    wb.inputs["Base Color"].default_value = (0.006, 0.03, 0.028, 1)
    wb.inputs["Roughness"].default_value = 0.06
    wb.inputs["IOR"].default_value = 1.333
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
    oc = bpy.context.active_object
    oc.name = "Ocean"
    mod = oc.modifiers.new("Ocean", "OCEAN")
    mod.geometry_mode = "GENERATE"
    mod.spatial_size = 120
    mod.resolution = 14
    mod.repeat_x = 5
    mod.repeat_y = 5
    mod.wave_scale = 0.45
    mod.choppiness = 1.0
    mod.wind_velocity = 7.0
    mod.random_seed = 3
    oc.data.materials.append(water)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = oc.evaluated_get(dg)
    bb = np.array([ev.matrix_world @ Vector(c) for c in ev.bound_box])
    cen = (bb.min(0) + bb.max(0)) / 2
    oc.location = (-cen[0], -cen[1], 0)
    bpy.ops.object.shade_smooth()
    bpy.ops.mesh.primitive_plane_add(size=40000, location=(0, 0, -0.5))
    far = bpy.context.active_object
    far.name = "OceanFar"
    far.data.materials.append(water)


SUN_ROT_DEG = 53.0  # sky sun horizontal dir = (sin r, cos r): starboard bow


def sun_vector():
    e = math.radians(4.0)
    r = math.radians(SUN_ROT_DEG)
    return V3(math.sin(r) * math.cos(e), math.cos(r) * math.cos(e), math.sin(e))


def add_camera(name, loc, target, lens):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.clip_start = 0.2
    cd.clip_end = 30000
    ob = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = Vector(loc)
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return ob


VIEW_DEFS = {
    "front34": ((-38, 46, 7.0), (0, 2, 10.5), 35),
    "side": ((84, -1, 9.0), (0, 3, 12.5), 42),
    "sternlow": ((-19, -37, 1.3), (0, -6, 7.5), 30),
    "deck": ((15, -26, 17.5), (0, 2, 3.5), 30),
    "sil": ((-80, -95, 4.0), (0, 2, 11.0), 50),
    "bow": ((15, 31, 4.5), (0, 13.5, 3.8), 35),
    "sternq": ((16, -31, 5.0), (0, -12.5, 3.8), 35),
}


def render_views(tag):
    os.makedirs(RENDER_DIR, exist_ok=True)
    sc = bpy.context.scene
    for v in VIEWS.split(","):
        if v not in VIEW_DEFS:
            continue
        loc, tgt, lens = VIEW_DEFS[v]
        cam = bpy.data.objects.get("Cam_" + v) or add_camera("Cam_" + v, loc, tgt, lens)
        sc.camera = cam
        sc.render.filepath = os.path.join(RENDER_DIR, "%s_%s.png" % (tag, v))
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        log("rendered", sc.render.filepath, "%.1fs" % (time.time() - t0))


def ao_render(tag):
    sc = bpy.context.scene
    mat = bpy.data.materials.new("AOView")
    mat.use_nodes = True
    nt = mat.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    at = nt.nodes.new("ShaderNodeAttribute")
    at.attribute_type = "GEOMETRY"
    at.attribute_name = "ao"
    nt.links.new(at.outputs["Fac"], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs[0])
    ob = bpy.data.objects["PirateShip"]
    old = ob.data.materials[0]
    ob.data.materials[0] = mat
    smp = sc.cycles.samples
    sc.cycles.samples = 8
    look = sc.view_settings.view_transform
    sc.view_settings.view_transform = "Standard"
    for v in ("front34", "deck", "sternq"):
        sc.camera = bpy.data.objects["Cam_" + v]
        sc.render.filepath = os.path.join(RENDER_DIR, "%s_ao_%s.png" % (tag, v))
        bpy.ops.render.render(write_still=True)
    ob.data.materials[0] = old
    sc.cycles.samples = smp
    sc.view_settings.view_transform = look


def albedo_render(tag, views):
    sc = bpy.context.scene
    eng = sc.render.engine
    me = bpy.data.objects["PirateShip"].data
    try:
        me.color_attributes.active_color = me.color_attributes["albedo"]
    except Exception as ex:
        log("albedo attr", ex)
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "FLAT"
    sc.display.shading.color_type = "VERTEX"
    vt = sc.view_settings.view_transform
    sc.view_settings.view_transform = "Standard"
    for v in views:
        sc.camera = bpy.data.objects["Cam_" + v]
        sc.render.filepath = os.path.join(RENDER_DIR, "%s_albedo_%s.png" % (tag, v))
        bpy.ops.render.render(write_still=True)
    sc.view_settings.view_transform = vt
    sc.render.engine = eng


def check_render(tag):
    sc = bpy.context.scene
    eng = sc.render.engine
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.show_backface_culling = True
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "SINGLE"
    for v in ("front34", "sternlow"):
        loc, tgt, lens = VIEW_DEFS[v]
        cam = bpy.data.objects.get("Cam_" + v) or add_camera("Cam_" + v, loc, tgt, lens)
        sc.camera = cam
        sc.render.filepath = os.path.join(RENDER_DIR, "%s_check_%s.png" % (tag, v))
        bpy.ops.render.render(write_still=True)
    sc.render.engine = eng


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    acc = Acc()
    info = {}
    info.update(build_hull(acc))
    build_keel_stem(acc)
    build_deck_and_bulwarks(acc, info)
    build_hull_bands(acc)
    build_guns(acc)
    build_stern(acc, info)
    build_bowsprit(acc, info)
    build_furniture(acc, info)
    build_helm(acc, info)
    build_pirate(acc, info)
    build_masts(acc, info)
    build_yards(acc)
    build_rigging(acc, info)
    build_sails(acc, info)
    build_sheets(acc, info)
    build_flag(acc, info)
    P, N, UV, C, M, R, E, W, T = acc.arrays()
    C = weather(P, N, C, M, info)
    # drop degenerate triangles
    area = np.linalg.norm(face_normals(P, T), axis=1)
    T = T[area > 1e-10]
    log("verts %d  tris %d" % (len(P), len(T)))
    for k, v in sorted(acc.tag_tris.items(), key=lambda kv: -kv[1]):
        log("   %-12s %7d tris" % (k, v))
    # centre of buoyancy -> origin
    hs = hydrostatics()
    shift = -hs["cb_y"]
    P = P.copy()
    P[:, 1] += shift
    AO = np.ones(len(P)) if FAST else compute_ao(P, N, T, M)
    eye = info["eye"] + V3(0, shift, 0)
    HEADER_EXTRA[:] = [float(eye[0]), float(eye[2]), float(-eye[1]), NECK_BOTTOM]
    bmin, bmax = export_bin(BIN_PATH, P, N, UV, C, AO, M, R, E, W, T)
    log("wrote", BIN_PATH, os.path.getsize(BIN_PATH), "bytes")
    report(info, hs, shift, bmin, bmax, len(P), len(T))
    # preview scene
    make_mesh_object(P, N, UV, C, AO, M, R, E, W, T)
    setup_scene(info)
    P0 = info["pirate_pivot"] + V3(0, shift, 0)
    VIEW_DEFS["helm"] = (tuple(P0 + V3(1.3, 3.6, 1.6)), tuple(P0 + V3(0, 0.1, 1.3)), 40)
    VIEW_DEFS["helm2"] = (tuple(P0 + V3(-2.7, 2.4, 1.7)), tuple(P0 + V3(0, 0.15, 1.25)), 40)
    VIEW_DEFS["helm3"] = (tuple(P0 + V3(1.5, -2.3, 2.1)), tuple(P0 + V3(0, 0.4, 1.25)), 40)
    for v, (loc, tgt, lens) in VIEW_DEFS.items():
        add_camera("Cam_" + v, loc, tgt, lens)
    bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH, compress=True)
    log("saved", BLEND_PATH)
    if RENDER_TAG:
        if CHECK:
            check_render(RENDER_TAG)
        if opt("--albedo", False):
            albedo_render(RENDER_TAG, ("side", "sternq", "deck", "bow"))
        if opt("--aoview", False):
            ao_render(RENDER_TAG)
        render_views(RENDER_TAG)


def report(info, hs, shift, bmin, bmax, nv, nt):
    def ex(p):
        return (p[0], p[2], -(p[1] + shift))
    print("\n================ SHIP REPORT (export space: Y up, bow -Z, m) ================")
    print("vertices %d  triangles %d" % (nv, nt))
    print("bbox min (%.2f, %.2f, %.2f)  max (%.2f, %.2f, %.2f)" % (*bmin, *bmax))
    print("displacement volume %.1f m^3 (~%.0f t in sea water), CB height y=%.2f, waterplane area %.1f m^2"
          % (hs["vol"], hs["vol"] * 1.025, hs["cb_z"], hs["wpa"]))
    print("Blender->export: shift along Blender Y = %+.3f (CB at export z=0)" % shift)
    print("main deck height (midship, export y) = %.2f ; quarterdeck %.2f ; forecastle %.2f"
          % (float(deck_main(-shift)), float(deck_main(-10 - shift * 0) + QD_H), float(deck_main(12) + FC_H)))
    print("stem at waterline: export z = %.2f ; sternpost at waterline: export z = %.2f"
          % (-(float(y_stem(0)) + shift), -(float(y_post(0)) + shift)))
    print("\n  z(exp) | half-width @y=0 | deck y | half-width @deck | rail top y")
    zs = np.arange(math.floor(-(float(y_stem(0)) + shift) / 2) * 2, -(float(y_post(0)) + shift) + 2.01, 2.0)
    for ze in zs:
        by = -ze - shift
        dl = float(deck_level(by))
        print("  %6.1f | %15.2f | %6.2f | %16.2f | %6.2f" % (ze, float(hb(by, 0.0)), dl, float(hb(by, dl)),
                                                          float(z_top(by))))
    for m in ("fore", "main", "mizzen"):
        t = info["trucks"][m]
        print("mast %-6s truck: (%.2f, %.2f, %.2f)" % ((m,) + ex(t)))
    print("main flag: top of hoist at y=%.2f" % (info["trucks"]["main"][2] - 0.15))
    wc = info["wheel_centre"]
    print("wheel pivot (axle centre): (%.3f, %.3f, %.3f)   axis: (0, 0, -1) (export -Z = toward the bow)" % ex(wc))
    print("helmsman foot pivot: (%.3f, %.3f, %.3f)" % ex(info["pirate_pivot"]))
    print("helmsman eye (between the eyes): (%.3f, %.3f, %.3f)" % ex(info["eye"]))
    print("helmsman neck bottom: local y %.3f above foot pivot (export y %.3f)"
          % (NECK_BOTTOM, info["pirate_pivot"][2] + NECK_BOTTOM))
    for i, L in enumerate(info["lanterns"]):
        print("stern lantern %d glass centre: (%.2f, %.2f, %.2f)" % ((i,) + ex(L)))
    print("=============================================================================\n", flush=True)


main()
log("done")
