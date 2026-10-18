// Crude stand-in ship mesh in the SHP1 format, for engine tests without Blender.
//   node tools/make_test_ship.mjs  ->  assets/ship_test.bin
import fs from 'node:fs';
const V = [], I = [];
function f16(v) { const f = new Float32Array([v]); const x = new Uint32Array(f.buffer)[0];
  const s = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112, m = (x >>> 13) & 0x3ff;
  return e <= 0 ? s : e >= 31 ? s | 0x7c00 : s | (e << 10) | m; }
function vert(p, n, uv, col, mat) { V.push({ p, n, uv, col, mat }); return V.length - 1; }
function quad(a, b, c, d, n, mat, col, uvs) {
  const i = [a, b, c, d].map((p, k) => vert(p, n, uvs[k], col, mat));
  I.push(i[0], i[1], i[2], i[0], i[2], i[3]);
}
// hull: lofted from sections
const L = 28, B = 4, D = 3.2, top = 2.6, ns = 24;
const sec = (t) => { const k = Math.sin(Math.PI * Math.min(1, Math.max(0, t))); return Math.pow(k, 0.6); };
for (let i = 0; i < ns; i++) {
  const t0 = i / ns, t1 = (i + 1) / ns;
  const z0 = -L / 2 + t0 * L, z1 = -L / 2 + t1 * L;
  const w0 = B * sec(t0) + 0.05, w1 = B * sec(t1) + 0.05;
  for (const s of [-1, 1]) {
    const a = [s * w0, top, z0], b = [s * w1, top, z1], c = [s * w1 * 0.4, -D, z1], d = [s * w0 * 0.4, -D, z0];
    const n = [s, -0.3, 0];
    const q = s > 0 ? [a, d, c, b] : [a, b, c, d];
    quad(...q, n, [0, 128, 255], [70, 45, 28, 255], [[0, 0], [0, 6], [1, 6], [1, 0]].map(([u, v]) => [u + z0, v]));
  }
  quad([-w0, top - 0.4, z0], [-w1, top - 0.4, z1], [w1, top - 0.4, z1], [w0, top - 0.4, z0], [0, 1, 0], [1, 170, 0], [150, 120, 85, 255],
    [[z0, -w0], [z1, -w1], [z1, w1], [z0, w0]]);
}
// mast + sail
for (const [mz, h] of [[-6, 24], [3, 27]]) {
  const r = 0.3;
  for (let k = 0; k < 6; k++) {
    const a0 = k / 6 * 2 * Math.PI, a1 = (k + 1) / 6 * 2 * Math.PI;
    const p = (a, y) => [Math.cos(a) * r, y, mz + Math.sin(a) * r];
    quad(p(a0, 0), p(a1, 0), p(a1, h), p(a0, h), [Math.cos(a0), 0, Math.sin(a0)], [9, 180, 0], [90, 60, 35, 255], [[0, 0], [0, 1], [h, 1], [h, 0]]);
  }
  const g = 10;
  for (let j = 0; j < g; j++) for (let i = 0; i < g; i++) {
    const P = (i, j) => { const u = i / g, v = j / g; return [(u - 0.5) * 12, h - 4 - v * 14, mz - 0.6 - Math.sin(u * Math.PI) * Math.sin(v * Math.PI) * 1.6]; };
    quad(P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1), [0, 0, -1], [2, 230, 200], [225, 210, 180, 255],
      [[i * 1.2, j * 1.4], [(i + 1) * 1.2, j * 1.4], [(i + 1) * 1.2, (j + 1) * 1.4], [i * 1.2, (j + 1) * 1.4]]);
  }
}
quad([-0.3, 3, 13.5], [0.3, 3, 13.5], [0.3, 3.6, 13.5], [-0.3, 3.6, 13.5], [0, 0, 1], [7, 0, 255], [255, 200, 120, 255], [[0, 0], [1, 0], [1, 1], [0, 1]]);
const buf = Buffer.alloc(64 + V.length * 28 + I.length * 4);
buf.write('SHP1', 0); buf.writeUInt32LE(V.length, 4); buf.writeUInt32LE(I.length, 8);
const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
V.forEach((v, k) => {
  const o = 64 + k * 28;
  for (let c = 0; c < 3; c++) { buf.writeFloatLE(v.p[c], o + c * 4); mn[c] = Math.min(mn[c], v.p[c]); mx[c] = Math.max(mx[c], v.p[c]); }
  const l = Math.hypot(...v.n);
  for (let c = 0; c < 3; c++) buf.writeInt8(Math.round(v.n[c] / l * 127), o + 12 + c);
  buf.writeUInt16LE(f16(v.uv[0]), o + 16); buf.writeUInt16LE(f16(v.uv[1]), o + 18);
  for (let c = 0; c < 4; c++) buf.writeUInt8(v.col[c], o + 20 + c);
  for (let c = 0; c < 3; c++) buf.writeUInt8(v.mat[c], o + 24 + c);
});
for (let c = 0; c < 3; c++) { buf.writeFloatLE(mn[c], 16 + c * 4); buf.writeFloatLE(mx[c], 28 + c * 4); }
I.forEach((x, k) => buf.writeUInt32LE(x, 64 + V.length * 28 + k * 4));
fs.writeFileSync(new URL('../assets/ship_test.bin', import.meta.url), buf);
console.log('verts', V.length, 'tris', I.length / 3);
