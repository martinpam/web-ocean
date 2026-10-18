// Pirate ship: lit mesh, sun shadow map, and the multi-point water height probe that
// drives its buoyancy.
import { FRAME_WGSL, ATMO_CORE_WGSL, ENV_MIPS, CASCADES, NOISE_WGSL } from './common.js';
import { SKY_RENDER_FUNCS } from './sky.js';

export const SHIP_PROBES = 16;
export const SHADOW_RES = 2048;
export const MAX_LANTERNS = 4;

// Ship uniform block, shared by the ship, shadow, ocean and foam shaders.
// Layout must match the Float32Array offsets in ship.js (writeUniforms).
export const SHIP_WGSL = /* wgsl */ `
struct Ship {
  model:    mat4x4f,   // ship local -> camera-relative world
  shadowVP: mat4x4f,   // camera-relative world -> sun shadow clip space
  p0:       vec4f,     // time, sail flutter, lantern flicker, 1 / shadow resolution
  p1:       vec4f,     // ship world x, z, forward x, z
  p2:       vec4f,     // speed (m/s), hull half length, hull half beam, hull centre z (local)
  lant:     array<vec4f, ${MAX_LANTERNS}>,  // camera-relative lantern position, w = intensity
  p3:       vec4f,     // reflection plane height relative to the camera, sail set (0..1), hide head, neck height (local y)
  wheel:    vec4f,     // wheel axle centre (local xyz), wheel angle (rad)
  crew:     vec4f,     // helmsman foot pivot (local xyz), counter-roll (rad)
  wind:     vec4f,     // helmsman counter-pitch (rad), sail fill (0 = luffing .. 1 = full), gust, -
};
`;

// Hull half-width at the waterline, sampled along the hull (ship local z).
export const HULL_STATIONS = 64;
export const HULL_WGSL = /* wgsl */ `
// hull: ${HULL_STATIONS} half-widths, then the local z of the first and last station
fn hullHalfWidth(lz: f32) -> f32 {
  let z0 = hull[${HULL_STATIONS}];
  let z1 = hull[${HULL_STATIONS + 1}];
  let t = (lz - z0) / (z1 - z0) * ${HULL_STATIONS - 1}.0;
  if (t < 0.0 || t > ${HULL_STATIONS - 1}.0) { return 0.0; }
  let i = u32(floor(t));
  let j = min(i + 1u, ${HULL_STATIONS - 1}u);
  return mix(hull[i], hull[j], fract(t));
}
`;

const SHIP_BINDINGS = /* wgsl */ `
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var transLUT: texture_2d<f32>;
@group(0) @binding(2) var skyLUT: texture_2d<f32>;
@group(0) @binding(3) var sampC: sampler;
@group(0) @binding(4) var<uniform> S: Ship;
@group(0) @binding(5) var envCube: texture_cube<f32>;
@group(0) @binding(6) var shadowMap: texture_depth_2d;
@group(0) @binding(7) var shadowSamp: sampler_comparison;
@group(0) @binding(8) var<storage, read> sunVis: array<f32, 4>;
`;

// Shared sun shadow lookup (needs S, shadowMap, shadowSamp). 12-tap Poisson PCF.
export const SHADOW_FUNCS = /* wgsl */ `
const POISSON = array<vec2f, 12>(
  vec2f(-0.326, -0.406), vec2f(-0.840, -0.074), vec2f(-0.696, 0.457), vec2f(-0.203, 0.621),
  vec2f(0.962, -0.195), vec2f(0.473, -0.480), vec2f(0.519, 0.767), vec2f(0.185, -0.893),
  vec2f(0.507, 0.064), vec2f(0.896, 0.412), vec2f(-0.322, -0.933), vec2f(-0.792, -0.598));
fn sunShadow(rel: vec3f, radius: f32) -> f32 {
  let c = S.shadowVP * vec4f(rel, 1.0);
  let uv = vec2f(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || c.z > 1.0) { return 1.0; }
  var s = 0.0;
  let r = radius * S.p0.w;
  for (var i = 0; i < 12; i++) {
    s += textureSampleCompareLevel(shadowMap, shadowSamp, uv + POISSON[i] * r, c.z);
  }
  return s / 12.0;
}
`;

const VERTEX_IN = /* wgsl */ `
struct VI {
  @location(0) pos: vec3f,
  @location(1) nrm: vec4f,
  @location(2) uv: vec2f,
  @location(3) col: vec4f,
  @location(4) mat: vec4f,
};
const MAT_SAIL: u32 = 2u;
const MAT_FLAG: u32 = 8u;
fn matId(m: vec4f) -> u32 { return u32(m.x * 255.0 + 0.5); }
const W_ROTATE: u32 = 255u;   // material.w: part of the turning wheel
const W_CREW: u32 = 200u;     // material.w: the helmsman
// direction the canvas bellies out to (downwind of the modelled sails)
const BELLY_DIR: vec3f = vec3f(-0.33, 0.0, -0.94);

fn rotX(p: vec3f, a: f32) -> vec3f { let c = cos(a); let s = sin(a); return vec3f(p.x, c * p.y - s * p.z, s * p.y + c * p.z); }
fn rotZ(p: vec3f, a: f32) -> vec3f { let c = cos(a); let s = sin(a); return vec3f(c * p.x - s * p.y, s * p.x + c * p.y, p.z); }

struct Posed { pos: vec3f, nrm: vec3f };

// Local-space animation: wind in the sails, the flag, the turning wheel and the helmsman.
fn pose(v: VI) -> Posed {
  let id = matId(v.mat);
  let tag = u32(v.mat.w * 255.0 + 0.5);
  let k = v.mat.z;
  let t = S.p0.x;
  var o: Posed;
  o.pos = v.pos;
  o.nrm = v.nrm.xyz;
  if (id == MAT_SAIL) {
    // both sheets of a sail must move the same way: push along the downwind-facing normal
    let n = v.nrm.xyz * select(1.0, -1.0, dot(v.nrm.xyz, BELLY_DIR) < 0.0);
    let fill = S.wind.y;
    let gust = S.wind.z;
    // full sails: the modelled belly, swelling with the gusts and rippling a little
    let ripple = sin(t * 3.1 + v.uv.x * 0.8 - v.uv.y * 0.5) * 0.04 + sin(t * 5.3 - v.uv.y * 1.7) * 0.02;
    var d = (fill * (gust - 1.0) * 0.9 - (1.0 - fill) * 0.85) * k + ripple * k * fill;
    // luffing: wind from ahead makes the slack canvas shake
    let luff = (1.0 - fill) * (sin(t * 8.5 - v.uv.y * 1.3 + v.uv.x * 0.9) * 0.55 + sin(t * 13.0 + v.uv.x * 2.1) * 0.25);
    d += luff * k * 0.45;
    // shortened sail: the canvas is gathered up towards its head (the yard)
    let furl = (1.0 - S.p3.y) * 0.85;
    o.pos += n * (d * S.p0.y) + vec3f(0.0, v.uv.y * furl, 0.0);
    return o;
  }
  if (id == MAT_FLAG) {
    let ph = t * (5.0 + 3.0 * S.wind.z) - v.uv.x * 2.2;
    o.pos += vec3f(0.0, sin(ph * 0.7 + v.uv.y) * 0.08, 0.0) * k + v.nrm.xyz * (sin(ph) * 0.22 * k);
    return o;
  }
  if (tag == W_ROTATE) {
    o.pos = S.wheel.xyz + rotZ(v.pos - S.wheel.xyz, S.wheel.w);
    o.nrm = rotZ(v.nrm.xyz, S.wheel.w);
    return o;
  }
  if (tag == W_CREW) {
    // arms: turn with the wheel about its axle (material.z = how much a vertex follows),
    // up to about 50 degrees, as a helmsman moves his hands before he re-grips
    let ga = clamp(S.wheel.w, -0.87, 0.87) * k;
    let ap = S.wheel.xyz + rotZ(v.pos - S.wheel.xyz, ga);
    let an = rotZ(v.nrm.xyz, ga);
    // body: stands upright against the roll and pitch of the deck, and breathes;
    // the hands stay on the wheel
    let q = ap - S.crew.xyz;
    let breathe = 1.0 + 0.012 * sin(t * 1.7) * smoothstep(1.0, 1.4, q.y) * (1.0 - smoothstep(1.4, 1.6, q.y));
    let a = (S.crew.w + 0.03 * sin(t * 0.9)) * (1.0 - k);
    let b = S.wind.x * (1.0 - k);
    o.pos = S.crew.xyz + rotX(rotZ(q * vec3f(breathe, 1.0, breathe), a), b);
    o.nrm = rotX(rotZ(an, a), b);
    return o;
  }
  return o;
}

// Round holes and tears in the canvas (same pattern on both sheets: it depends on the
// local position only).
fn canvasHole(lp: vec3f) -> bool {
  let q = vec2f(lp.x * 0.8 + lp.z * 1.1, lp.y * 0.95);
  let n = gnoise(q * 0.55) * 0.7 + gnoise(q * 1.9 + 7.3) * 0.3 + gnoise(q * 6.0 + 1.7) * 0.12;
  return n > 0.36;
}
`;

export const SHIP_SHADOW = FRAME_WGSL + SHIP_WGSL + NOISE_WGSL + /* wgsl */ `
@group(0) @binding(0) var<uniform> S: Ship;
` + VERTEX_IN + /* wgsl */ `
struct SO { @builtin(position) pos: vec4f, @location(0) lp: vec3f, @location(1) @interpolate(flat) mat: vec4f };
@vertex fn vs(v: VI) -> SO {
  var o: SO;
  o.pos = S.shadowVP * (S.model * vec4f(pose(v).pos, 1.0));
  o.lp = v.pos;
  o.mat = v.mat;
  return o;
}
// sunlight passes through the holes in the sails
@fragment fn fs(in: SO) {
  if (matId(in.mat) == MAT_SAIL && canvasHole(in.lp)) { discard; }
}
`;

export const SHIP = FRAME_WGSL + ATMO_CORE_WGSL + SHIP_WGSL + SHIP_BINDINGS + SKY_RENDER_FUNCS + SHADOW_FUNCS + VERTEX_IN + /* wgsl */ `
const ENV_MAXMIP: f32 = ${ENV_MIPS - 1}.0;

struct VO {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) nrm: vec3f,
  @location(2) uv: vec2f,
  @location(3) col: vec4f,
  @location(4) @interpolate(flat) mat: vec4f,
  @location(5) ly: f32,          // local height above the design waterline
  @location(6) lp: vec3f,        // rest position in ship space (hole pattern)
};

@vertex fn vs(v: VI) -> VO {
  var o = transform(v);
  // first-person view from the helmsman's eyes: leave out his head, hat and beard
  if (S.p3.z > 0.5 && u32(v.mat.w * 255.0 + 0.5) == W_CREW && v.pos.y > S.p3.w) { o.pos = vec4f(0.0, 0.0, 0.0, 0.0); }
  return o;
}

fn transform(v: VI) -> VO {
  let ps = pose(v);
  var o: VO;
  let w = S.model * vec4f(ps.pos, 1.0);
  o.rel = w.xyz;
  o.pos = F.viewProj * w;
  o.nrm = (S.model * vec4f(ps.nrm, 0.0)).xyz;
  o.lp = v.pos;
  o.uv = v.uv;
  o.col = v.col;
  o.mat = v.mat;
  o.ly = v.pos.y;
  return o;
}

fn h21(p: vec2f) -> f32 {
  return f32(pcgHash(u32(i32(p.x) + 4096) * 1664525u + pcgHash(u32(i32(p.y) + 4096)))) / 4294967296.0;
}
fn srgbToLin(c: vec3f) -> vec3f { return pow(c, vec3f(2.2)); }
// anti-aliased thin line at the edges of a unit cell (x and w in cell units, aa = pixel
// footprint in cell units; derivatives are taken up front, outside the material branches)
fn cellLine(x: f32, w: f32, aa: f32) -> f32 {
  let f = fract(x);
  let d = min(f, 1.0 - f);
  return (1.0 - smoothstep(w - aa, w + aa, d)) * (1.0 - smoothstep(0.15, 0.5, aa));
}
fn ggx(NdH: f32, a: f32) -> f32 {
  let a2 = a * a;
  let d = NdH * NdH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}
fn smithG(NdV: f32, NdL: f32, a: f32) -> f32 {
  let k = a * 0.5;
  return (NdV / (NdV * (1.0 - k) + k)) * (NdL / (NdL * (1.0 - k) + k));
}

// Mirror image of the ship in the sea plane, for the water reflection: placed mirrored,
// but shaded at its real position as seen from the mirrored eye.
@vertex fn vsRefl(v: VI) -> VO {
  var o = transform(v);
  let m = S.p3.x;
  o.pos = F.viewProj * vec4f(o.rel.x, 2.0 * m - o.rel.y, o.rel.z, 1.0);
  return o;
}

@fragment fn fs(in: VO, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let dist = length(in.rel);
  return vec4f(shade(in, front, -in.rel / dist, dist), 1.0);
}

@fragment fn fsRefl(in: VO, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let eye = vec3f(0.0, 2.0 * S.p3.x, 0.0);
  let d = eye - in.rel;
  let dist = length(d);
  let c = shade(in, !front, d / dist, dist);
  // only what stands above the water is mirrored; the higher a point, the more the
  // rough sea scatters its reflection, so tall rigging fades out
  let h = in.rel.y - S.p3.x;
  if (h < 0.0) { discard; }
  return vec4f(c, exp(-max(h - 2.5, 0.0) / 7.0));
}

fn shade(in: VO, front: bool, V: vec3f, dist: f32) -> vec3f {
  let id = matId(in.mat);
  var N = normalize(in.nrm);
  if (!front) { N = -N; }
  let L = F.sunDir.xyz;
  let uv = in.uv;
  let fu = fwidth(uv.x);
  let fv = fwidth(uv.y);

  var alb = srgbToLin(in.col.rgb);
  var rough = max(in.mat.y, 0.05);
  var metal = 0.0;
  var ao = in.col.a;
  var emis = vec3f(0.0);
  var trans = 0.0;
  var wet = 0.0;
  var hole = false;

  if (id == 0u || id == 1u || id == 6u || id == 10u) {
    // planking: rows across v, butt joints along u, per-plank tint, grain, caulked seams
    let pw = select(select(0.24, 0.2, id == 1u), 0.4, id == 10u);
    let pl = select(5.5, 1.2, id == 10u);
    let row = floor(uv.y / pw);
    let off = h21(vec2f(row, 7.0)) * pl;
    let seg = floor((uv.x + off) / pl);
    let tint = 0.78 + 0.4 * h21(vec2f(row, seg));
    let grain = gnoise(vec2f((uv.x + off) * 1.3, uv.y * 38.0 + row * 3.1)) * 0.5
              + gnoise(vec2f((uv.x + off) * 7.0, uv.y * 90.0)) * 0.25;
    let detail = 1.0 - smoothstep(0.2, 1.0, fv / pw);
    let seam = max(cellLine(uv.y / pw, 0.035, fv / pw), cellLine((uv.x + off) / pl, 0.004, fu / pl));
    var k = mix(1.0, tint * (1.0 + grain * 0.35), detail);
    if (id == 6u) { k = mix(1.0, 0.92 + 0.16 * h21(vec2f(row, seg)) + grain * 0.1, detail); }
    alb *= k * (1.0 - 0.65 * seam * detail);
    rough = clamp(rough + grain * 0.15, 0.05, 1.0);
    if (id == 10u) { metal = 0.35; }
  } else if (id == 2u || id == 8u) {
    // canvas: sewn cloths, weave, weather stains; light passes through
    let cloths = cellLine(uv.x / 0.62, 0.02, fu / 0.62) * 0.5;
    let weave = gnoise(uv * vec2f(160.0, 160.0)) * (1.0 - smoothstep(0.004, 0.02, fu));
    let stain = gnoise(uv * 0.35) * 0.5 + gnoise(uv * 1.3 + 5.0) * 0.25;
    alb *= (1.0 - cloths * 0.25) * (0.92 + weave * 0.08) * (0.9 + stain * 0.35);
    trans = select(0.3, 0.12, id == 8u);
    hole = id == 2u && canvasHole(in.lp);
  } else if (id == 3u) {
    let tw = sin((uv.x * 40.0 + uv.y * 25.0) * 6.2832);
    alb *= 0.85 + 0.15 * tw * (1.0 - smoothstep(0.05, 0.2, fu * 40.0));
  } else if (id == 4u) {
    rough = max(rough, 0.35);
    metal = 0.6;
  } else if (id == 5u) {
    metal = 1.0;
  } else if (id == 7u) {
    let fl = 0.85 + 0.15 * S.p0.z;
    emis = vec3f(1.0, 0.52, 0.18) * (in.mat.z * 26.0 * fl);
    alb *= 0.2;
  } else if (id == 9u) {
    let grain = gnoise(vec2f(uv.x * 1.6, uv.y * 30.0)) * (1.0 - smoothstep(0.02, 0.1, fv));
    alb *= 0.9 + grain * 0.25;
  }

  // weather: rain and spray streaks run down the timber
  if (id == 0u || id == 6u || id == 9u || id == 1u) {
    let streak = gnoise(vec2f((in.lp.x + in.lp.z) * 2.2, in.lp.y * 0.22)) + 0.4 * gnoise(vec2f((in.lp.x - in.lp.z) * 7.0, in.lp.y * 0.9));
    alb *= 1.0 - 0.32 * smoothstep(0.05, 0.55, streak);
  }
  // the hull is dark and glossy where the sea washes over it
  wet = 1.0 - smoothstep(0.2, 1.4, in.ly);
  alb *= mix(1.0, 0.55, wet);
  rough = mix(rough, 0.18, wet);

  // ---- lighting ----
  let sunE = sunIlluminance() * sunVis[0];
  let NdL = dot(N, L);
  let sh = sunShadow(in.rel + L * 0.06 + N * 0.03, 1.6);
  let a = rough * rough;
  let H = normalize(V + L);
  let NdV = max(dot(N, V), 1e-3);
  let NdH = max(dot(N, H), 0.0);
  let f0 = mix(vec3f(0.04), alb, metal);
  let Fs = f0 + (vec3f(1.0) - f0) * pow(1.0 - max(dot(V, H), 0.0), 5.0);
  let nl = max(NdL, 0.0);
  let specSun = Fs * ggx(NdH, a) * smithG(NdV, nl, a) / (4.0 * NdV + 1e-3) * nl;
  let diff = alb * (1.0 - metal) / PI;
  var col = (diff * nl + specSun) * sunE * sh;
  // light through the canvas when the sun is behind it
  col += alb * sunE * (trans * max(-NdL, 0.0) * sh / PI) * vec3f(1.0, 0.82, 0.62);

  // sky light (diffuse irradiance from the blurred environment) + bounce from the sea
  let skyIrr = textureSampleLevel(envCube, sampC, normalize(N + vec3f(0.0, 0.25, 0.0)), ENV_MAXMIP).rgb;
  let seaIrr = textureSampleLevel(envCube, sampC, vec3f(0.0, 1.0, 0.0), ENV_MAXMIP).rgb * 0.12;
  let irr = mix(seaIrr, skyIrr * 1.6, smoothstep(-0.6, 0.4, N.y));
  col += diff * PI * irr * ao * (1.0 + trans * 0.4);
  // environment reflection
  var R = reflect(-V, N);
  let below = R.y < 0.0;
  R.y = abs(R.y);
  let envLod = clamp(rough * ENV_MAXMIP, 0.0, ENV_MAXMIP);
  var env = textureSampleLevel(envCube, sampC, normalize(R), envLod).rgb;
  if (below) { env *= 0.18; }
  let Fe = f0 + (max(vec3f(1.0 - rough), f0) - f0) * pow(1.0 - NdV, 5.0);
  col += env * Fe * ao * ao;

  // stern lanterns: warm point lights on nearby planks and sails
  for (var i = 0; i < ${MAX_LANTERNS}; i++) {
    let lp = S.lant[i];
    if (lp.w <= 0.0) { continue; }
    let d = lp.xyz - in.rel;
    let d2 = dot(d, d);
    let ld = d * inverseSqrt(d2);
    let wrap = max(dot(N, ld) * 0.8 + 0.2, 0.0);
    col += diff * PI * vec3f(1.0, 0.5, 0.17) * (lp.w * S.p0.z * wrap / (d2 + 0.4));
  }
  col += emis;

  // aerial perspective, same as the sea
  let T = exp(-max(F.fogExt.rgb, vec3f(F.fogExt.w)) * dist);
  let hz = normalize(vec3f(-V.x, max(-V.y, 0.002), -V.z));
  col = col * T + skyAtmo(hz) * (1.0 - T);
  if (hole) { discard; }
  return col;
}
`;

// Water height at up to 16 points (world xz in P), inverting the horizontal displacement.
export const SHIP_PROBE = FRAME_WGSL + /* wgsl */ `
struct Pts { p: array<vec4f, ${SHIP_PROBES / 2}> };
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<uniform> P: Pts;
@group(0) @binding(2) var dispTex: texture_2d_array<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var<storage, read_write> outH: array<f32, ${SHIP_PROBES}>;
fn dispAt(xz: vec2f) -> vec3f {
  var d = vec3f(0.0);
  for (var c = 0u; c < ${CASCADES}u; c++) {
    d += textureSampleLevel(dispTex, samp, xz * F.cascInvL[c >> 2u][c & 3u], c, 0.0).xyz;
  }
  return d;
}
@compute @workgroup_size(${SHIP_PROBES})
fn main(@builtin(local_invocation_index) k: u32) {
  let q = P.p[k >> 1u];
  let goal = select(q.xy, q.zw, (k & 1u) == 1u);
  var x0 = goal;
  for (var i = 0; i < 4; i++) {
    let d = dispAt(x0);
    x0 = goal - d.xz;
  }
  outH[k] = dispAt(x0).y;
}
`;
