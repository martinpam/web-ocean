// Pirate ship: mesh loading, buoyancy from the simulated sea, sailing controls, and the
// GPU passes (sun shadow map, lit mesh, water height probe).
// Mesh format (assets/ship.bin, little-endian): 64-byte header 'SHP1', vertexCount,
// indexCount, -, bbox min xyz, bbox max xyz; then 28-byte vertices (pos f32x3, normal
// snorm8x4, uv f16x2 in metres, albedo sRGB + AO unorm8x4, material id/roughness/extra
// unorm8x4) and u32 indices. Space: metres, Y up, bow towards -Z, waterline at y = 0.
import { SHIP, SHIP_SHADOW, SHIP_PROBE, SHIP_PROBES, SHADOW_RES, MAX_LANTERNS, HULL_STATIONS } from './wgsl/ship.js';

const BUF = GPUBufferUsage;
const TEX = GPUTextureUsage;
const DEG = Math.PI / 180;
const STRIDE = 28;
const MAT_LANTERN = 7;
const HULL_MATS = new Set([0, 6, 10]);
const MAX_SPEED = 7.5;          // m/s (~15 kn) running with full sail
const SHADOW_FAR = 3000;        // m behind the ship (long sunset shadows on the water)

async function loadMesh(url) {
  // the self-contained bundle embeds the mesh as base64 (see build.mjs)
  if (typeof SHIP_B64 !== 'undefined') {
    const s = atob(SHIP_B64); // eslint-disable-line no-undef
    const u = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    return u.buffer;
  }
  const r = await fetch(url);
  if (!r.ok) throw new Error('ship mesh missing: ' + url);
  return r.arrayBuffer();
}

function parseMesh(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'SHP1') throw new Error('bad ship mesh header');
  const vCount = dv.getUint32(4, true), iCount = dv.getUint32(8, true);
  const bbox = new Float32Array(buf, 16, 6).slice();
  const verts = new Uint8Array(buf, 64, vCount * STRIDE);
  const idxOff = 64 + vCount * STRIDE;
  const indices = new Uint32Array(buf.slice(idxOff, idxOff + iCount * 4));
  return { vCount, iCount, bbox, verts, dv };
}

// Half-width of the hull at the waterline per station, plus lantern positions.
function analyseMesh(m) {
  const { vCount, bbox, dv } = m;
  const z0 = bbox[2], z1 = bbox[5];
  const hw = new Float32Array(HULL_STATIONS + 2);
  const has = new Uint8Array(HULL_STATIONS);
  const lantern = [];
  for (let i = 0; i < vCount; i++) {
    const o = 64 + i * STRIDE;
    const x = dv.getFloat32(o, true), y = dv.getFloat32(o + 4, true), z = dv.getFloat32(o + 8, true);
    const mat = dv.getUint8(o + 24);
    if (HULL_MATS.has(mat) && Math.abs(y) < 0.6) {
      const s = Math.round((z - z0) / (z1 - z0) * (HULL_STATIONS - 1));
      hw[s] = Math.max(hw[s], Math.abs(x));
      has[s] = 1;
    }
    // lanterns glow at full strength; the dimmer stern windows get no point light
    if (mat === MAT_LANTERN && dv.getUint8(o + 26) > 250) lantern.push(x, y, z);
  }
  // fill empty stations between occupied ones
  for (let s = 0; s < HULL_STATIONS; s++) {
    if (has[s]) continue;
    let a = s - 1, b = s + 1;
    while (a >= 0 && !has[a]) a--;
    while (b < HULL_STATIONS && !has[b]) b++;
    if (a >= 0 && b < HULL_STATIONS) hw[s] = hw[a] + (hw[b] - hw[a]) * (s - a) / (b - a);
  }
  hw[HULL_STATIONS] = z0; hw[HULL_STATIONS + 1] = z1;
  let wlMin = Infinity, wlMax = -Infinity, beam = 0;
  for (let s = 0; s < HULL_STATIONS; s++) {
    if (hw[s] <= 0.05) continue;
    const z = z0 + (z1 - z0) * s / (HULL_STATIONS - 1);
    wlMin = Math.min(wlMin, z); wlMax = Math.max(wlMax, z); beam = Math.max(beam, hw[s]);
  }
  // greedy 1 m clusters of lantern glass -> point lights
  const lights = [];
  for (let i = 0; i < lantern.length; i += 3) {
    let c = lights.find((l) => Math.hypot(l.x / l.n - lantern[i], l.y / l.n - lantern[i + 1], l.z / l.n - lantern[i + 2]) < 1.0);
    if (!c) { c = { x: 0, y: 0, z: 0, n: 0 }; lights.push(c); }
    c.x += lantern[i]; c.y += lantern[i + 1]; c.z += lantern[i + 2]; c.n++;
  }
  lights.sort((a, b) => b.n - a.n);
  const lanterns = lights.slice(0, MAX_LANTERNS).map((l) => [l.x / l.n, l.y / l.n, l.z / l.n]);
  if (!isFinite(wlMin)) { wlMin = z0 * 0.8; wlMax = z1 * 0.8; beam = (bbox[3] - bbox[0]) * 0.4; }
  return { hw, wlMin, wlMax, beam, lanterns };
}

export async function createShip(device, R) {
  const qs = new URLSearchParams(location.search);
  const mesh = parseMesh(await loadMesh(qs.get('shipmesh') || 'assets/ship.bin'));
  const hull = analyseMesh(mesh);
  const halfLen = (hull.wlMax - hull.wlMin) * 0.5;
  const midZ = (hull.wlMax + hull.wlMin) * 0.5;
  const bb = mesh.bbox;
  const radius = 0.5 * Math.hypot(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]) + 0.5;
  const centre = [(bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2];

  const vbuf = device.createBuffer({ size: mesh.verts.byteLength, usage: BUF.VERTEX | BUF.COPY_DST, label: 'ship verts' });
  device.queue.writeBuffer(vbuf, 0, mesh.verts);
  const iBytes = mesh.iCount * 4;
  const ibuf = device.createBuffer({ size: iBytes, usage: BUF.INDEX | BUF.COPY_DST, label: 'ship idx' });
  device.queue.writeBuffer(ibuf, 0, new Uint8Array(mesh.dv.buffer, 64 + mesh.vCount * STRIDE, iBytes));
  const hullBuf = device.createBuffer({ size: hull.hw.byteLength, usage: BUF.STORAGE | BUF.COPY_DST, label: 'hull widths' });
  device.queue.writeBuffer(hullBuf, 0, hull.hw);

  const ubo = device.createBuffer({ size: 256, usage: BUF.UNIFORM | BUF.COPY_DST, label: 'ship ubo' });
  const U = new Float32Array(64);
  const shadowTex = device.createTexture({
    label: 'ship shadow', size: [SHADOW_RES, SHADOW_RES], format: 'depth32float',
    usage: TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING,
  });
  const shadowView = shadowTex.createView();
  const shadowSamp = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });

  const vertexBuffers = [{
    arrayStride: STRIDE,
    attributes: [
      { shaderLocation: 0, offset: 0, format: 'float32x3' },
      { shaderLocation: 1, offset: 12, format: 'snorm8x4' },
      { shaderLocation: 2, offset: 16, format: 'float16x2' },
      { shaderLocation: 3, offset: 20, format: 'unorm8x4' },
      { shaderLocation: 4, offset: 24, format: 'unorm8x4' },
    ],
  }];
  const [shipMod, shadowMod] = await Promise.all([R.shader(SHIP, 'ship'), R.shader(SHIP_SHADOW, 'ship shadow')]);
  const [pipe, reflPipe, shadowPipe, probePipe] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'ship', layout: 'auto',
      vertex: { module: shipMod, entryPoint: 'vs', buffers: vertexBuffers },
      fragment: { module: shipMod, entryPoint: 'fs', targets: [{ format: R.HDR }] },
      // sails and flags are two opposite-facing sheets, so back faces are culled
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
      multisample: { count: R.MSAA },
    }),
    device.createRenderPipelineAsync({
      label: 'ship reflection', layout: 'auto',
      vertex: { module: shipMod, entryPoint: 'vsRefl', buffers: vertexBuffers },
      fragment: { module: shipMod, entryPoint: 'fsRefl', targets: [{ format: R.HDR }] },
      primitive: { topology: 'triangle-list', cullMode: 'front' },  // mirroring flips the winding
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'greater' },
    }),
    device.createRenderPipelineAsync({
      label: 'ship shadow', layout: 'auto',
      vertex: { module: shadowMod, entryPoint: 'vs', buffers: vertexBuffers },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
    }),
    R.computePipe(SHIP_PROBE, 'ship probe'),
  ]);
  const bgOf = (p, entries) => device.createBindGroup({
    layout: p.getBindGroupLayout(0), entries: entries.map((r, i) => ({ binding: i, resource: r })),
  });
  const bg = bgOf(pipe, [{ buffer: R.frameUBO }, R.transView, R.skyView, R.sampClamp, { buffer: ubo },
    R.envCubeView, shadowView, shadowSamp, { buffer: R.sunVisBuf }]);
  const bgRefl = bgOf(reflPipe, [{ buffer: R.frameUBO }, R.transView, R.skyView, R.sampClamp, { buffer: ubo },
    R.envCubeView, shadowView, shadowSamp, { buffer: R.sunVisBuf }]);
  const bgShadow = bgOf(shadowPipe, [{ buffer: ubo }]);
  // mirrored ship at half resolution; alpha marks where it covers the sea
  let reflTex = null, reflDepth = null;
  const reflPass = {
    colorAttachments: [{ view: null, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }],
    depthStencilAttachment: { view: null, depthClearValue: 0, depthLoadOp: 'clear', depthStoreOp: 'discard' },
  };
  function resize(w, h) {
    for (const t of [reflTex, reflDepth]) t && t.destroy();
    const rw = Math.max(1, w >> 1), rh = Math.max(1, h >> 1);
    reflTex = device.createTexture({ label: 'ship refl', size: [rw, rh], format: R.HDR, usage: TEX.RENDER_ATTACHMENT | TEX.TEXTURE_BINDING });
    reflDepth = device.createTexture({ label: 'ship refl depth', size: [rw, rh], format: 'depth32float', usage: TEX.RENDER_ATTACHMENT });
    reflPass.colorAttachments[0].view = reflTex.createView();
    reflPass.depthStencilAttachment.view = reflDepth.createView();
    return reflPass.colorAttachments[0].view;
  }
  const shadowPass = {
    colorAttachments: [],
    depthStencilAttachment: { view: shadowView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
  };

  // ---- buoyancy probes: a grid over the waterplane (local x, z) ----
  const probeLocal = [];
  for (const fz of [-0.85, -0.45, 0, 0.45, 0.85]) {
    const w = Math.sqrt(Math.max(0, 1 - fz * fz)) * 0.8 + 0.2;
    for (const fx of [-0.8, 0.8]) probeLocal.push(fx * hull.beam * w, midZ + fz * halfLen);
  }
  while (probeLocal.length < SHIP_PROBES * 2) probeLocal.push(0, midZ);
  const probeUBO = device.createBuffer({ size: SHIP_PROBES * 8, usage: BUF.UNIFORM | BUF.COPY_DST, label: 'ship probe pts' });
  const probeF = new Float32Array(SHIP_PROBES * 2);
  const probeOut = device.createBuffer({ size: SHIP_PROBES * 4, usage: BUF.STORAGE | BUF.COPY_SRC, label: 'ship probe out' });
  const probeRead = device.createBuffer({ size: SHIP_PROBES * 4, usage: BUF.MAP_READ | BUF.COPY_DST, label: 'ship probe read' });
  const bgProbe = bgOf(probePipe, [{ buffer: R.frameUBO }, { buffer: probeUBO }, R.dispView, R.samplerSim, { buffer: probeOut }]);
  const heights = new Float32Array(SHIP_PROBES);
  let probePending = false, probeCopied = false, haveHeights = false;
  const onProbe = () => {
    heights.set(new Float32Array(probeRead.getMappedRange()));
    probeRead.unmap();
    probePending = false;
    haveHeights = true;
  };
  const onProbeFail = () => { probePending = false; };

  // ---- state ----
  const st = {
    // default heading: wind on the starboard quarter, as the sails are modelled
    x: 0, z: 0, yaw: 253 * DEG, speed: 0, sail: 0.6, rudder: 0, yawRate: 0,
    heave: 0, heaveV: 0, pitch: 0, pitchV: 0, roll: 0, rollV: 0, heel: 0,
    halfLen, beam: hull.beam, midZ, radius, centre,
  };
  if (qs.get('ship')) {
    const v = qs.get('ship').split(',').map(Number);
    st.x = v[0] || 0; st.z = v[1] || 0; st.yaw = (v[2] || 0) * DEG;
  }
  if (qs.has('sail')) st.sail = Number(qs.get('sail'));
  st.speed = st.sail * MAX_SPEED * 0.6;
  // basis of the ship, recomputed every frame
  const B = { r: [1, 0, 0], u: [0, 1, 0], b: [0, 0, 1], f: [0, 0, -1] };

  function update(dt, keys, windDirDeg) {
    // sails (W/S), rudder (A/D)
    if (keys.KeyW || keys.ArrowUp) st.sail = Math.min(1, st.sail + dt * 0.35);
    if (keys.KeyS || keys.ArrowDown) st.sail = Math.max(0, st.sail - dt * 0.35);
    const steer = (keys.KeyD || keys.ArrowRight ? 1 : 0) - (keys.KeyA || keys.ArrowLeft ? 1 : 0);
    st.rudder += (steer - st.rudder) * Math.min(1, dt * 2.5);
    const fx = Math.sin(st.yaw), fz = -Math.cos(st.yaw);
    const wx = Math.cos(windDirDeg * DEG), wz = Math.sin(windDirDeg * DEG);
    const along = fx * wx + fz * wz;             // 1 = running before the wind
    const cross = -fz * wx + fx * wz;            // wind towards starboard (+) / port (-)
    const t = Math.min(1, Math.max(0, (along + 0.75) / 0.8));
    const windK = 0.18 + 0.82 * t * t * (3 - 2 * t);
    const target = st.sail * MAX_SPEED * windK;
    st.speed += (target - st.speed) * Math.min(1, dt * (target > st.speed ? 0.09 : 0.16));
    const turn = st.rudder * (0.025 + 0.075 * Math.min(1, st.speed / 4));
    st.yawRate += (turn - st.yawRate) * Math.min(1, dt * 1.2);
    st.yaw += st.yawRate * dt;
    st.x += fx * st.speed * dt;
    st.z += fz * st.speed * dt;
    // heel: wind pressure on the sails plus the turn
    const heelT = -cross * st.sail * windK * 0.09 + st.yawRate * st.speed * 0.12;
    st.heel += (heelT - st.heel) * Math.min(1, dt * 0.4);

    // fit a plane h = a + b*x + c*z to the probe heights (ship-local axes)
    let a = 0, b = 0, c = 0;
    if (haveHeights) {
      let sxx = 0, szz = 0, sxh = 0, szh = 0, n = 0, sh = 0;
      for (let i = 0; i < 10; i++) { sh += heights[i]; n++; }
      a = sh / n;
      for (let i = 0; i < 10; i++) {
        const lx = probeLocal[i * 2], lz = probeLocal[i * 2 + 1] - midZ;
        const h = heights[i] - a;
        sxx += lx * lx; szz += lz * lz; sxh += lx * h; szh += lz * h;
      }
      b = sxh / sxx; c = szh / szz;
    }
    // damped oscillators towards the sea surface: heave, pitch (about x), roll (about z)
    const spring = (val, vel, tgt, period, zeta) => {
      const w = 2 * Math.PI / period;
      const acc = w * w * (tgt - val) - 2 * zeta * w * vel;
      return vel + acc * dt;
    };
    st.heaveV = spring(st.heave, st.heaveV, a, 3.2, 0.55); st.heave += st.heaveV * dt;
    st.pitchV = spring(st.pitch, st.pitchV, c, 4.0, 0.45); st.pitch += st.pitchV * dt;
    st.rollV = spring(st.roll, st.rollV, b, 5.5, 0.22); st.roll += st.rollV * dt;

    // orientation: heading basis tilted onto the smoothed wave plane plus heel
    const r0 = [Math.cos(st.yaw), 0, Math.sin(st.yaw)];
    const b0 = [-fx, 0, -fz];
    const gb = st.roll + Math.tan(st.heel), gc = st.pitch;
    let u = [-(gb * r0[0] + gc * b0[0]), 1, -(gb * r0[2] + gc * b0[2])];
    let l = Math.hypot(u[0], u[1], u[2]); u = u.map((v) => v / l);
    const d = b0[0] * u[0] + b0[2] * u[2];
    let bk = [b0[0] - d * u[0], -d * u[1], b0[2] - d * u[2]];
    l = Math.hypot(bk[0], bk[1], bk[2]); bk = bk.map((v) => v / l);
    B.u = u; B.b = bk;
    B.r = [u[1] * bk[2] - u[2] * bk[1], u[2] * bk[0] - u[0] * bk[2], u[0] * bk[1] - u[1] * bk[0]];
    B.f = [-bk[0], -bk[1], -bk[2]];

    for (let i = 0; i < SHIP_PROBES; i++) {
      const lx = probeLocal[i * 2], lz = probeLocal[i * 2 + 1];
      probeF[i * 2] = st.x + r0[0] * lx + b0[0] * lz;
      probeF[i * 2 + 1] = st.z + r0[2] * lx + b0[2] * lz;
    }
    device.queue.writeBuffer(probeUBO, 0, probeF);
  }

  // world position of a ship-local point
  function toWorld(lx, ly, lz, out) {
    out[0] = st.x + B.r[0] * lx + B.u[0] * ly + B.b[0] * lz;
    out[1] = st.heave + B.r[1] * lx + B.u[1] * ly + B.b[1] * lz;
    out[2] = st.z + B.r[2] * lx + B.u[2] * ly + B.b[2] * lz;
    return out;
  }

  const tmp = [0, 0, 0];
  function writeUniforms(cam, sunDir, time) {
    // model: columns r, u, b, translation (camera-relative)
    U.fill(0);
    U.set([B.r[0], B.r[1], B.r[2], 0, B.u[0], B.u[1], B.u[2], 0, B.b[0], B.b[1], B.b[2], 0,
      st.x - cam.x, st.heave - cam.y, st.z - cam.z, 1], 0);
    // shadow: orthographic along the sun direction around the ship's bounding sphere
    toWorld(centre[0], centre[1], centre[2], tmp);
    const cx = tmp[0] - cam.x, cy = tmp[1] - cam.y, cz = tmp[2] - cam.z;
    const L = sunDir;
    let ex = [L[2], 0, -L[0]];
    let l = Math.hypot(ex[0], ex[2]) || 1; ex = [ex[0] / l, 0, ex[2] / l];
    const ey = [L[1] * ex[2] - L[2] * ex[1], L[2] * ex[0] - L[0] * ex[2], L[0] * ex[1] - L[1] * ex[0]];
    const rr = radius, depth = radius + SHADOW_FAR;
    const cL = cx * L[0] + cy * L[1] + cz * L[2];
    const s = 16;
    U[s + 0] = ex[0] / rr; U[s + 4] = ex[1] / rr; U[s + 8] = ex[2] / rr; U[s + 12] = -(cx * ex[0] + cy * ex[1] + cz * ex[2]) / rr;
    U[s + 1] = ey[0] / rr; U[s + 5] = ey[1] / rr; U[s + 9] = ey[2] / rr; U[s + 13] = -(cx * ey[0] + cy * ey[1] + cz * ey[2]) / rr;
    U[s + 2] = -L[0] / depth; U[s + 6] = -L[1] / depth; U[s + 10] = -L[2] / depth; U[s + 14] = (rr + cL) / depth;
    U[s + 15] = 1;
    const flick = Math.sin(time * 9.1) * 0.5 + Math.sin(time * 23.7 + 1.3) * 0.3 + Math.sin(time * 3.3) * 0.2;
    U[32] = time; U[33] = 0.4 + 0.6 * st.sail; U[34] = 1 + 0.12 * flick; U[35] = 1 / SHADOW_RES;
    U[36] = st.x; U[37] = st.z; U[38] = Math.sin(st.yaw); U[39] = -Math.cos(st.yaw);
    U[40] = st.speed; U[41] = halfLen; U[42] = hull.beam; U[43] = midZ;
    hull.lanterns.forEach((p, i) => {
      toWorld(p[0], p[1], p[2], tmp);
      U[44 + i * 4] = tmp[0] - cam.x; U[45 + i * 4] = tmp[1] - cam.y; U[46 + i * 4] = tmp[2] - cam.z; U[47 + i * 4] = 2.2;
    });
    U[60] = st.heave - cam.y; U[61] = st.sail;
    device.queue.writeBuffer(ubo, 0, U);
  }

  function encodeCompute(cp) {
    cp.setPipeline(probePipe);
    cp.setBindGroup(0, bgProbe);
    cp.dispatchWorkgroups(1);
  }
  function encodeShadow(enc) {
    const p = enc.beginRenderPass(shadowPass);
    p.setPipeline(shadowPipe);
    p.setBindGroup(0, bgShadow);
    p.setVertexBuffer(0, vbuf);
    p.setIndexBuffer(ibuf, 'uint32');
    p.drawIndexed(mesh.iCount);
    p.end();
    probeCopied = false;
    if (!probePending) { enc.copyBufferToBuffer(probeOut, 0, probeRead, 0, SHIP_PROBES * 4); probeCopied = true; }
  }
  function encodeReflection(enc) {
    const p = enc.beginRenderPass(reflPass);
    p.setPipeline(reflPipe);
    p.setBindGroup(0, bgRefl);
    p.setVertexBuffer(0, vbuf);
    p.setIndexBuffer(ibuf, 'uint32');
    p.drawIndexed(mesh.iCount);
    p.end();
  }
  function draw(rp) {
    rp.setPipeline(pipe);
    rp.setBindGroup(0, bg);
    rp.setVertexBuffer(0, vbuf);
    rp.setIndexBuffer(ibuf, 'uint32');
    rp.drawIndexed(mesh.iCount);
  }
  function afterSubmit() {
    if (probeCopied) { probePending = true; probeRead.mapAsync(GPUMapMode.READ).then(onProbe, onProbeFail); }
  }

  return {
    st, B, ubo, hullBuf, shadowView, shadowSamp, resize, update, toWorld, writeUniforms,
    encodeCompute, encodeShadow, encodeReflection, draw, afterSubmit,
  };
}
