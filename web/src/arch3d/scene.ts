import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { ArchDetail, ArchLive, ArchRun } from "../lib/types";

// The Architecture page's model in 3D, set in the same marks as every other scene in the app (a dot where it's faint,
// then a plus, an ×, a block) but as points in space: every neuron of every MLP and every attention head is one.
//
// The residual stream runs along x, from the embedding to the unembedding. Each layer stands across it as a slice in
// the y–z plane: its attention above the stream (a tile per head, grouped by the key/value head they share; filled
// where the head attends over every earlier token, an outline with its diagonal where it keeps a running state
// instead), its MLP below (a sheet with a point per neuron, brighter where the neuron writes more), and a ring
// around the stream for each of its two norms. The stream's thickness is the layer's typical residual size, or the
// text's when one has been run.
//
// Opening a layer parts the layers either side of it (one warp of x, applied in the shader so everything moves
// together), dims the rest, and flies the camera in; opening a block of it faces it head on, so single neurons and
// heads can be picked. Running text lights each head and neuron by what it did at the chosen token, sweeping from
// the first layer to the last.

export const S = 1.8; // layer spacing along the stream
export const W = 2.2; // the width of a layer's sheets (z)
const GAP = 1.1; // how far an opened layer's neighbours move aside, each way
const OFF = { norm1: -0.6, attn: -0.32, norm2: -0.04, mlp: 0.28 } as const; // where each block sits within a layer
const MLP_TOP = -0.42, ATTN_BOT = 0.42;
const DUR = 1100;

export type Part = keyof typeof OFF;
export type End = "embed" | "unembed";
export type Focus =
  | { level: "model" }
  | { level: "layer"; layer: number }
  | { level: "part"; layer: number; part: Part }
  | { level: "end"; end: End };
export type Pick =
  | { kind: "layer"; layer: number }
  | { kind: "part"; layer: number; part: Part }
  | { kind: "end"; end: End }
  | { kind: "neuron"; layer: number; neuron: number }
  | { kind: "head"; layer: number; head: number };

export type Hooks = { hover?: (p: Pick | null, x: number, y: number) => void; click?: (p: Pick) => void };

const VERT = /* glsl */ `
uniform float uPx, uSize, uFx, uGap, uS, uMin;
attribute float b; attribute float a; attribute float h;
varying float vB; varying float vA; varying float vH; varying float vPx;
float warp(float x) {
  if (uGap <= 0.0) return x;
  float lo = uFx - uS * 0.5, hi = uFx + uS * 0.5;
  if (x < lo) return x - uGap;
  if (x > hi) return x + uGap;
  return lo - uGap + (x - lo) * (uS + 2.0 * uGap) / uS;
}
void main() {
  vec3 p = position; p.x = warp(p.x);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  float px = uSize * uPx / max(0.001, -mv.z) * (1.0 + h * 1.4);
  vPx = px;
  gl_PointSize = max(px, uMin);
  gl_Position = projectionMatrix * mv;
  vB = b; vA = a; vH = h;
}`;

// the marks of glyphs.ts drawMarks, drawn inside each point once it's big enough on screen to show one
const FRAG = /* glsl */ `
uniform vec3 uBase, uPos, uNeg, uHi;
uniform float uDim, uHover, uRun, uReveal, uMin, uGlyph;
varying float vB; varying float vA; varying float vH; varying float vPx;
float mark(vec2 q, float br) {
  vec2 d = abs(q);
  if (br < 0.3) return step(max(d.x, d.y), 0.1 + 0.3 * br);
  if (br < 0.55) return max(step(d.x, 0.08) * step(d.y, 0.4), step(d.y, 0.08) * step(d.x, 0.4));
  if (br < 0.8) return step(abs(d.x - d.y), 0.1) * step(max(d.x, d.y), 0.33);
  return step(max(d.x, d.y), 0.42);
}
void main() {
  float base = vB * mix(1.0, 0.38, uRun);
  float act = abs(vA) * uRun * uReveal;
  vec3 col = uBase; float br = base;
  if (act > base) { br = act; col = vA > 0.0 ? uPos : uNeg; }
  br = min(1.0, br + uHover * 0.22);
  if (vH > 0.0) { br = 1.0; col = uHi; }
  float lum = br;
  if (vPx >= uGlyph) {
    if (mark(gl_PointCoord - 0.5, br) < 0.5) discard;
    lum = br < 0.3 ? 0.5 + br : 0.95;
  }
  float cover = clamp(vPx / uMin, 0.0, 1.0); cover *= cover; // a point smaller than a pixel lights only its share of it
  gl_FragColor = vec4(col * lum * uDim * cover, 1.0);
}`;

type Pts = {
  obj: THREE.Points; geo: THREE.BufferGeometry; b: Float32Array; a: Float32Array; h: Float32Array;
  u: { uSize: { value: number }; uDim: { value: number }; uHover: { value: number }; uReveal: { value: number } };
};

type Heads = { n: number; tc: number; tr: number; gs: number; cell: number; ga: number; top: number; ranges: [number, number][] };
type Layer = {
  x: number; attn: Pts; mlp: Pts; norm1: Pts; norm2: Pts;
  cols: number; rows: number; sp: number; bottom: number; heads: Heads;
  num: HTMLElement; lens: HTMLElement;
};

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const hash = (a: number, b: number) => { const s = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453; return s - Math.floor(s); };
const css = (el: Element, v: string, d: string) => getComputedStyle(el).getPropertyValue(v).trim() || d;

export function createScene(host: HTMLElement, m: ArchDetail, hooks: Hooks = {}, preview = false) {
  const L = m.layers.length, N = m.neurons;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
  renderer.setClearColor(0x000000, 0);
  host.appendChild(renderer.domElement);
  renderer.domElement.className = "a3-gl";

  const labels = document.createElement("div");
  labels.className = "a3-labels";
  if (!preview) host.appendChild(labels);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 600);
  const controls = preview ? null : new OrbitControls(camera, renderer.domElement);
  if (controls) {
    controls.enableDamping = true;
    controls.dampingFactor = 0.09;
    controls.rotateSpeed = 0.6;
    controls.zoomSpeed = 0.9;
    controls.screenSpacePanning = true;
    controls.minDistance = 0.4;
    controls.maxDistance = 220;
  }

  const col = (v: string, d: string) => new THREE.Color(css(host, v, d));
  const G = {
    uPx: { value: 800 }, uFx: { value: 0 }, uGap: { value: 0 }, uS: { value: S }, uMin: { value: 1 }, uGlyph: { value: 6 },
    uRun: { value: 0 },
    uBase: { value: col("--ascii", "#c3cff5") }, uPos: { value: col("--g-pos", "#f0b45a") },
    uNeg: { value: col("--g-neg", "#7aa2f7") }, uHi: { value: new THREE.Color("#ffffff") },
  };

  function pts(pos: number[], b: number[], size: number): Pts {
    const n = pos.length / 3, geo = new THREE.BufferGeometry();
    const bb = new Float32Array(b), aa = new Float32Array(n), hh = new Float32Array(n);
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute("b", new THREE.BufferAttribute(bb, 1));
    geo.setAttribute("a", new THREE.BufferAttribute(aa, 1));
    geo.setAttribute("h", new THREE.BufferAttribute(hh, 1));
    const u = { uSize: { value: size }, uDim: { value: 1 }, uHover: { value: 0 }, uReveal: { value: 1 } };
    const mat = new THREE.ShaderMaterial({
      uniforms: { ...G, ...u }, vertexShader: VERT, fragmentShader: FRAG,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const obj = new THREE.Points(geo, mat);
    obj.frustumCulled = false;
    scene.add(obj);
    return { obj, geo, b: bb, a: aa, h: hh, u };
  }

  const line = (pos: number[], b: number[], x: number, y0: number, y1: number, step = 0.05, bright = 0.32) => {
    for (let y = y0; Math.sign(y1 - y0) * (y1 - y) > 0; y += Math.sign(y1 - y0) * step) (pos.push(x, y, 0), b.push(bright));
  };
  const ring = (x: number, r: number, k: number, bright: number) => {
    const pos: number[] = [], b: number[] = [];
    for (let i = 0; i < k; i++) {
      const t = (i / k) * Math.PI * 2;
      pos.push(x, Math.cos(t) * r, Math.sin(t) * r);
      b.push(bright);
    }
    return pts(pos, b, 0.035);
  };

  // ---------------------------------------------------------------- the layers
  const headMax = Math.max(1e-6, ...m.layers.flatMap((l) => l.heads));
  const residMax = Math.max(1e-6, ...m.layers.map((l) => l.resid));

  const layers: Layer[] = m.layers.map((ld, l) => {
    const x = l * S;

    // the MLP: a sheet with a point per neuron, then the line it reads from and writes back to the stream along
    const cols = Math.ceil(Math.sqrt(N)), rows = Math.ceil(N / cols), sp = W / cols;
    const mp: number[] = [], mb: number[] = [];
    for (let i = 0; i < N; i++) {
      const c = i % cols, r = Math.floor(i / cols);
      mp.push(x + OFF.mlp, MLP_TOP - (r + 0.5) * sp, W / 2 - (c + 0.5) * sp);
      mb.push(0.18 + 0.12 * hash(l, i));
    }
    line(mp, mb, x + OFF.mlp, -0.08, MLP_TOP);
    const mlp = pts(mp, mb, sp * 0.9);

    // attention: a tile per head, the heads that share a key/value head side by side with a gap after each group
    const full = ld.attention !== "linear";
    const n = ld.heads.length || 8;
    const kv = (full ? m.kv_heads : m.linear_k_heads) || n;
    let gs = Math.max(1, Math.round(n / kv));
    const tc = n <= 8 ? n : gs >= 8 ? 8 : Math.max(gs, Math.floor(8 / gs) * gs);
    if (gs > tc || tc % gs) gs = tc;
    const tr = Math.ceil(n / tc);
    const cell = W / (tc + (tc / gs - 1) * 0.3), ga = cell * 0.3, top = ATTN_BOT + tr * cell;
    const g = Math.min(16, Math.ceil(Math.sqrt(ld.head_dim || 64)));
    const ap: number[] = [], ab: number[] = [], ranges: [number, number][] = [];
    for (let hd = 0; hd < n; hd++) {
      const c = hd % tc, r = Math.floor(hd / tc);
      const z0 = W / 2 - c * cell - Math.floor(c / gs) * ga - cell * 0.12, y0 = top - r * cell - cell * 0.12, st = (cell * 0.76) / (g - 1);
      const base = ld.heads.length ? 0.2 + 0.42 * (ld.heads[hd] / headMax) : 0.3;
      const s0 = ap.length / 3;
      for (let i = 0; i < g; i++)
        for (let j = 0; j < g; j++) {
          const edge = i === 0 || j === 0 || i === g - 1 || j === g - 1;
          if (!full && !edge && i !== j) continue; // a running state: the square it keeps, and its diagonal
          ap.push(x + OFF.attn, y0 - i * st, z0 - j * st);
          ab.push(base * (full ? (edge ? 1 : 0.75) : 1));
        }
      ranges.push([s0, ap.length / 3]);
    }
    line(ap, ab, x + OFF.attn, 0.08, ATTN_BOT);
    const attn = pts(ap, ab, Math.min(0.05, ((cell * 0.76) / (g - 1)) * 0.95));

    const num = document.createElement("span");
    num.className = "a3-num";
    num.textContent = String(l);
    const lens = document.createElement("span");
    lens.className = "a3-lens";
    labels.append(num, lens);

    return {
      x, attn, mlp, cols, rows, sp, bottom: MLP_TOP - rows * sp,
      norm1: ring(x + OFF.norm1, 0.2, 26, 0.42), norm2: ring(x + OFF.norm2, 0.2, 26, 0.42),
      heads: { n, tc, tr, gs, cell, ga, top, ranges }, num, lens,
    };
  });

  // ---------------------------------------------------------------- the ends and the stream
  const X0 = -S, X1 = (L - 1) * S + S;
  const XN = (L - 1) * S + S * 0.5; // the final norm
  function end(x: number, dir: number) {
    const pos: number[] = [], b: number[] = [];
    const cs = 26, rs = 54, h = 3.2, sp = W / cs;
    for (let r = 0; r < rs; r++)
      for (let c = 0; c < cs; c++) (pos.push(x, h / 2 - (r + 0.5) * (h / rs), W / 2 - (c + 0.5) * sp), b.push(0.14 + 0.3 * hash(r * 3 + dir, c)));
    // rays from the vocabulary down to the stream
    for (let k = 0; k < 9; k++) {
      const y0 = h / 2 - (k / 8) * h;
      for (let u = 0; u <= 1; u += 0.03) pos.push(x + dir * u * S * 0.7, y0 * (1 - u) ** 1.6, 0), b.push(0.16 + 0.3 * u);
    }
    return pts(pos, b, sp * 0.8);
  }
  const embed = end(X0, 1), unembed = end(X1, -1), fnorm = ring(XN, 0.2, 26, 0.42);

  const streamPos: number[] = [], streamB: number[] = [], STEP = 0.035, K = 6;
  const streamX: number[] = [];
  for (let x = X0 + S * 0.7; x <= X1 - S * 0.7; x += STEP)
    for (let k = 0; k < K; k++) (streamPos.push(x, 0, 0), streamB.push(0.55), streamX.push(x));
  const stream = pts(streamPos, streamB, 0.03);

  // the stream's radius at each layer: its typical size, or the run's at the chosen token
  let radius = m.layers.map((l) => l.resid / residMax);
  function shapeStream() {
    const p = stream.geo.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < streamX.length; i++) {
      const x = streamX[i], f = Math.max(0, Math.min(L - 1, x / S)), lo = Math.floor(f), hi = Math.min(L - 1, lo + 1);
      const r = 0.025 + 0.11 * (radius[lo] + (radius[hi] - radius[lo]) * (f - lo));
      const t = ((i % K) / K) * Math.PI * 2 + x * 2.2;
      p.setXYZ(i, x, Math.cos(t) * r, Math.sin(t) * r);
    }
    p.needsUpdate = true;
  }
  shapeStream();

  const endLabels = (["Embedding", "Unembedding"] as const).map((t) => {
    const e = document.createElement("span");
    e.className = "a3-end";
    e.textContent = t;
    labels.append(e);
    return e;
  });
  const partLabels = (Object.keys(OFF) as Part[]).map((p) => {
    const e = document.createElement("span");
    e.className = "a3-part";
    labels.append(e);
    return { p, e };
  });

  // ---------------------------------------------------------------- warp, tweens, camera
  const warpX = (x: number, fx = G.uFx.value, gap = G.uGap.value) => {
    if (gap <= 0) return x;
    const lo = fx - S / 2, hi = fx + S / 2;
    return x < lo ? x - gap : x > hi ? x + gap : lo - gap + ((x - lo) * (S + 2 * gap)) / S;
  };

  type Tw = { from: number; to: number; t0: number; dur: number; set: (v: number) => void };
  const tws = new Map<string, Tw>();
  const tween = (key: string, from: number, to: number, dur: number, set: (v: number) => void, delay = 0) => {
    if (dur <= 0) return (tws.delete(key), set(to));
    tws.set(key, { from, to, t0: performance.now() + delay, dur, set });
  };
  const toU = (key: string, u: { value: number }, to: number, dur = DUR * 0.8, delay = 0) => tween(key, u.value, to, dur, (v) => (u.value = v), delay);

  const center = (X0 + X1) / 2, len = X1 - X0;
  type Pose = { pos: THREE.Vector3; at: THREE.Vector3 };
  function pose(f: Focus): Pose {
    const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
    if (f.level === "model") {
      // just far enough back that the whole model fits: move the camera, see how far its corners land, repeat
      const at = V(center, -0.5, 0), dir = V(-0.55, 0.38, 0.74).normalize(), keep = camera.position.clone(), q = camera.quaternion.clone();
      const top = Math.max(...layers.map((l) => l.heads.top)), bot = Math.min(...layers.map((l) => l.bottom));
      const corners = [X0, X1].flatMap((x) => [bot, top].flatMap((y) => [-W / 2, W / 2].map((z) => V(x, y, z))));
      let d = len;
      for (let i = 0; i < 5; i++) {
        camera.position.copy(at).addScaledVector(dir, d);
        camera.lookAt(at);
        camera.updateMatrixWorld();
        const reach = Math.max(...corners.map((c) => { const p = c.clone().project(camera); return Math.max(Math.abs(p.x), Math.abs(p.y)); }));
        d *= Math.max(0.5, Math.min(2, reach / 0.94));
      }
      camera.position.copy(keep);
      camera.quaternion.copy(q);
      return { pos: at.clone().addScaledVector(dir, d), at };
    }
    if (f.level === "end") {
      const x = f.end === "embed" ? X0 : X1, side = f.end === "embed" ? -1 : 1;
      return { pos: V(x + side * 3.4, 0.9, 4.6), at: V(x - side * 0.4, 0, 0) };
    }
    const ly = layers[f.layer], fx = ly.x;
    // how far back a square this size fits the view, with a margin
    const fit = (size: number) => (size * 0.62) / Math.tan((Math.min(camera.fov, camera.fov * camera.aspect) * Math.PI) / 360);
    const wx = (dx: number) => warpX(fx + dx, fx, GAP);
    if (f.level === "layer") return { pos: V(wx(0) - 5.4, 2.3, 5.6), at: V(wx(0), -0.5, 0) };
    if (f.part === "mlp") {
      const y = (MLP_TOP + ly.bottom) / 2, d = fit(Math.max(W, MLP_TOP - ly.bottom));
      return { pos: V(wx(OFF.mlp) - d, y + 0.18, 0.55), at: V(wx(OFF.mlp), y, 0) };
    }
    if (f.part === "attn") {
      const y = (ATTN_BOT + ly.heads.top) / 2, d = fit(Math.max(W, ly.heads.top - ATTN_BOT));
      return { pos: V(wx(OFF.attn) - d, y + 0.15, 0.5), at: V(wx(OFF.attn), y, 0) };
    }
    return { pos: V(wx(OFF[f.part]) - 1.1, 0.4, 0.75), at: V(wx(OFF[f.part]), 0, 0) };
  }

  let fly: { from: Pose; to: Pose; t0: number; dur: number } | null = null;
  function flyTo(p: Pose, dur = DUR) {
    const from = { pos: camera.position.clone(), at: controls ? controls.target.clone() : look.clone() };
    fly = { from, to: p, t0: performance.now(), dur };
    if (controls) controls.enabled = false;
  }
  const look = new THREE.Vector3();

  // ---------------------------------------------------------------- focus
  let focus: Focus = { level: "model" };
  const allParts = (ly: Layer) => [ly.attn, ly.mlp, ly.norm1, ly.norm2];

  function dims(f: Focus) {
    const dimOf = (l: number, p: Part | null): number => {
      if (f.level === "model") return 1;
      if (f.level === "end") return 0.12;
      if (f.layer !== l) return f.level === "layer" ? 0.16 : 0.07;
      if (f.level === "layer" || p === f.part) return 1;
      return 0.3;
    };
    layers.forEach((ly, l) => (Object.keys(OFF) as Part[]).forEach((p) => toU(`d${ly[p].obj.id}`, ly[p].u.uDim, dimOf(l, p))));
    const endDim = (e: End) => (f.level === "model" ? 1 : f.level === "end" ? (f.end === e ? 1 : 0.12) : f.level === "layer" ? 0.3 : 0.12);
    toU("de", embed.u.uDim, endDim("embed"));
    toU("du", unembed.u.uDim, endDim("unembed"));
    toU("dn", fnorm.u.uDim, endDim("unembed"));
    toU("ds", stream.u.uDim, f.level === "model" ? 1 : f.level === "layer" ? 0.85 : 0.08);
  }

  function setFocus(f: Focus, dur = DUR) {
    focus = f;
    const open = f.level === "layer" || f.level === "part";
    if (open) {
      if (G.uGap.value > 0.01) toU("fx", G.uFx, layers[f.layer].x, dur);
      else G.uFx.value = layers[f.layer].x;
    }
    toU("gap", G.uGap, open ? GAP : 0, dur);
    dims(f);
    setHover(null);
    partLabels.forEach(({ p, e }) => (e.textContent = open ? partName(f.layer, p) : ""));
    flyTo(pose(f), dur);
  }

  const partName = (l: number, p: Part) => {
    const ld = m.layers[l];
    if (p === "attn") return `${ld.attention === "linear" ? "Linear attention" : "Attention"} · ${ld.heads.length} heads`;
    if (p === "mlp") return `MLP · ${N.toLocaleString("en-US")} neurons`;
    return p === "norm1" ? "Norm · before attention" : "Norm · before the MLP";
  };

  // ---------------------------------------------------------------- hover and selection marks
  let hoverObjs: Pts[] = [];
  let hoverMark: { p: Pts; i0: number; i1: number } | null = null;
  let selMarks: { p: Pts; i0: number; i1: number }[] = [];

  const mark = (p: Pts, i0: number, i1: number, v: number) => {
    for (let i = i0; i < i1; i++) p.h[i] = v;
    (p.geo.attributes.h as THREE.BufferAttribute).needsUpdate = true;
  };
  const markOf = (k: Pick): { p: Pts; i0: number; i1: number } | null =>
    k.kind === "neuron" ? { p: layers[k.layer].mlp, i0: k.neuron, i1: k.neuron + 1 }
      : k.kind === "head" ? { p: layers[k.layer].attn, i0: layers[k.layer].heads.ranges[k.head][0], i1: layers[k.layer].heads.ranges[k.head][1] }
        : null;

  function setHover(k: Pick | null) {
    const objs = !k ? [] : k.kind === "layer" ? allParts(layers[k.layer]) : k.kind === "part" ? [layers[k.layer][k.part]]
      : k.kind === "end" ? (k.end === "embed" ? [embed] : [unembed, fnorm]) : [];
    hoverObjs.filter((o) => !objs.includes(o)).forEach((o) => tween(`h${o.obj.id}`, o.u.uHover.value, 0, 180, (v) => (o.u.uHover.value = v)));
    objs.filter((o) => !hoverObjs.includes(o)).forEach((o) => tween(`h${o.obj.id}`, o.u.uHover.value, 1, 140, (v) => (o.u.uHover.value = v)));
    hoverObjs = objs;

    if (hoverMark) mark(hoverMark.p, hoverMark.i0, hoverMark.i1, 0);
    hoverMark = k ? markOf(k) : null;
    if (hoverMark) mark(hoverMark.p, hoverMark.i0, hoverMark.i1, 0.6);
    selMarks.forEach((s) => mark(s.p, s.i0, s.i1, 1));
  }

  function select(ks: Pick[]) {
    selMarks.forEach((s) => mark(s.p, s.i0, s.i1, 0));
    selMarks = ks.map(markOf).filter((x): x is NonNullable<typeof x> => !!x);
    selMarks.forEach((s) => mark(s.p, s.i0, s.i1, 1));
  }

  // ---------------------------------------------------------------- picking
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), box = new THREE.Box3(), hit = new THREE.Vector3();
  const boxAt = (x0: number, x1: number, y0: number, y1: number, z = W / 2) => box.set(new THREE.Vector3(warpX(x0), y0, -z), new THREE.Vector3(warpX(x1), y1, z));
  const near = (b: THREE.Box3) => (ray.ray.intersectBox(b, hit) ? hit.distanceTo(ray.ray.origin) : Infinity);
  const onPlane = (x: number) => {
    const o = ray.ray.origin, d = ray.ray.direction;
    if (Math.abs(d.x) < 1e-6) return null;
    const t = (warpX(x) - o.x) / d.x;
    return t > 0 ? o.clone().addScaledVector(d, t) : null;
  };

  function pick(cx: number, cy: number): Pick | null {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);

    if (focus.level === "part") {
      const ly = layers[focus.layer];
      if (focus.part === "mlp") {
        const p = onPlane(ly.x + OFF.mlp);
        if (p) {
          const c = Math.floor((W / 2 - p.z) / ly.sp), rr = Math.floor((MLP_TOP - p.y) / ly.sp), i = rr * ly.cols + c;
          if (c >= 0 && c < ly.cols && rr >= 0 && i < N) return { kind: "neuron", layer: focus.layer, neuron: i };
        }
      }
      if (focus.part === "attn") {
        const p = onPlane(ly.x + OFF.attn), H = ly.heads;
        if (p && p.y >= ATTN_BOT && p.y <= H.top) {
          const rr = Math.floor((H.top - p.y) / H.cell);
          for (let c = 0; c < H.tc; c++) {
            const z0 = W / 2 - c * H.cell - Math.floor(c / H.gs) * H.ga;
            const i = rr * H.tc + c;
            if (p.z <= z0 && p.z >= z0 - H.cell && i < H.n) return { kind: "head", layer: focus.layer, head: i };
          }
        }
      }
    }

    let best: Pick | null = null, bd = Infinity;
    const take = (d: number, k: Pick) => d < bd && ((bd = d), (best = k));
    const open = focus.level === "layer" || focus.level === "part" ? focus.layer : -1;
    layers.forEach((ly, l) => {
      if (l === open) {
        take(near(boxAt(ly.x + OFF.attn - 0.06, ly.x + OFF.attn + 0.06, ATTN_BOT - 0.05, ly.heads.top + 0.05)), { kind: "part", layer: l, part: "attn" });
        take(near(boxAt(ly.x + OFF.mlp - 0.06, ly.x + OFF.mlp + 0.06, ly.bottom - 0.05, MLP_TOP + 0.05)), { kind: "part", layer: l, part: "mlp" });
        for (const p of ["norm1", "norm2"] as const) take(near(boxAt(ly.x + OFF[p] - 0.08, ly.x + OFF[p] + 0.08, -0.24, 0.24, 0.24)), { kind: "part", layer: l, part: p });
      } else take(near(boxAt(ly.x - S * 0.42, ly.x + S * 0.42, ly.bottom, ly.heads.top)), { kind: "layer", layer: l });
    });
    take(near(boxAt(X0 - 0.1, X0 + 0.1, -1.6, 1.6)), { kind: "end", end: "embed" });
    take(near(boxAt(X1 - 0.1, X1 + 0.1, -1.6, 1.6)), { kind: "end", end: "unembed" });
    return best;
  }

  // ---------------------------------------------------------------- pointer
  let down: { x: number; y: number } | null = null, hovered: Pick | null = null;
  const same = (a: Pick | null, b: Pick | null) => JSON.stringify(a) === JSON.stringify(b);
  const el = renderer.domElement;
  const onMove = (e: PointerEvent) => {
    if (e.buttons) return;
    const k = pick(e.clientX, e.clientY);
    if (!same(k, hovered)) ((hovered = k), setHover(k));
    el.style.cursor = k ? "pointer" : "grab";
    hooks.hover?.(k, e.clientX, e.clientY);
  };
  const onDown = (e: PointerEvent) => (down = { x: e.clientX, y: e.clientY });
  const onUp = (e: PointerEvent) => {
    if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return (down = null);
    down = null;
    const k = pick(e.clientX, e.clientY);
    if (k) hooks.click?.(k);
  };
  const onLeave = () => ((hovered = null), setHover(null), hooks.hover?.(null, 0, 0));
  if (!preview) {
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointerleave", onLeave);
  }

  // ---------------------------------------------------------------- data in
  function setNorms(l: number, norms: number[]) {
    const ly = layers[l], sorted = [...norms].sort((a, b) => a - b), p99 = sorted[Math.floor(sorted.length * 0.99)] || 1;
    for (let i = 0; i < N; i++) ly.mlp.b[i] = 0.12 + 0.5 * Math.min(1, norms[i] / p99) ** 1.5;
    (ly.mlp.geo.attributes.b as THREE.BufferAttribute).needsUpdate = true;
  }

  let run: ArchRun | null = null, swept = false;
  function setRun(r: ArchRun | null) {
    liveEnd();
    run = r;
    swept = false;
    toU("run", G.uRun, r ? 1 : 0, 500);
    if (!r) {
      radius = m.layers.map((l) => l.resid / residMax);
      shapeStream();
      layers.forEach((ly) => (ly.lens.textContent = ""));
    }
  }

  /** Lights the run at token t: every head by how much it wrote, every neuron by its impact (q: int8 [L, N], each
   *  layer scaled to its own peak, given in peak). */
  function setToken(t: number, q: Int8Array | null, peak: number[] | null) {
    if (!run) return;
    const rm = Math.max(1e-6, ...run.resid.map((r) => r[t]));
    radius = run.resid.map((r) => r[t] / rm);
    shapeStream();

    const gp = peak ? Math.max(1e-6, ...peak) : 1;
    const hm = Math.max(1e-6, ...run.heads.flatMap((h) => h[t] || []));
    layers.forEach((ly, l) => {
      const hs = run!.heads[l][t] || [];
      ly.heads.ranges.forEach(([i0, i1], h) => ly.attn.a.fill(Math.sqrt((hs[h] || 0) / hm), i0, i1));
      (ly.attn.geo.attributes.a as THREE.BufferAttribute).needsUpdate = true;
      if (q && peak) {
        const s = Math.sqrt(peak[l] / gp);
        for (let i = 0; i < N; i++) ly.mlp.a[i] = Math.sign(q[l * N + i]) * Math.sqrt(Math.abs(q[l * N + i]) / 127) * s;
        (ly.mlp.geo.attributes.a as THREE.BufferAttribute).needsUpdate = true;
      }
      const top = run!.lens[l][t]?.[0];
      ly.lens.textContent = top ? JSON.stringify(top.tok).slice(1, -1) : "";
      ly.lens.style.setProperty("--p", String(0.35 + 0.65 * (top?.p ?? 0)));
      // the first time, sweep from the embedding to the unembedding; after that, crossfade
      for (const p of allParts(ly)) {
        if (!swept) (p.u.uReveal.value = 0), toU(`r${p.obj.id}`, p.u.uReveal, 1, 420, 80 + l * (1100 / L));
        else (p.u.uReveal.value = 0.35), toU(`r${p.obj.id}`, p.u.uReveal, 1, 260, l * 6);
      }
    });
    swept = true;
  }

  // ---------------------------------------------------------------- live, while the model writes
  // Each token it writes arrives as a frame of what every layer did to choose it. A pulse carries the frame down the
  // stream: as it reaches a layer, that layer's heads and neurons head to their new values (eased every animation
  // frame, so the view flows rather than jumps), it flares, and its lens word updates. When the pulse reaches the
  // unembedding the token itself appears there.
  const liveN = layers.map(() => new Map<number, number>()); // neuron -> where its impact is heading
  const liveH = layers.map((ly) => new Float32Array(ly.heads.n));
  let live = false, radiusT = radius.slice();
  const pulses: number[] = [], TRAVEL = 1.1; // seconds from the embedding to the unembedding
  const reach = (l: number) => (l + 1) / (L + 1) * TRAVEL * 1000; // ms until the pulse reaches layer l
  let pend: { f: ArchLive; t0: number; next: number; gp: number; hm: number; rm: number } | null = null;
  const outTok = document.createElement("span");
  outTok.className = "a3-out";
  labels.append(outTok);

  function liveStart() {
    live = true;
    run = null;
    pend = null;
    toU("run", G.uRun, 1, 400);
    layers.forEach((ly, l) => {
      ly.mlp.a.fill(0);
      ly.attn.a.fill(0);
      (ly.mlp.geo.attributes.a as THREE.BufferAttribute).needsUpdate = true;
      (ly.attn.geo.attributes.a as THREE.BufferAttribute).needsUpdate = true;
      allParts(ly).forEach((p) => (p.u.uReveal.value = 1));
      liveN[l].clear();
      liveH[l].fill(0);
      ly.lens.textContent = "";
    });
    outTok.textContent = "";
  }

  /** Sends one token down the model; resolves once it has come out at the unembedding. */
  function liveFrame(f: ArchLive, text: string) {
    if (!live) liveStart();
    while (pend && pend.next < L) arrive(pend.next++);
    const rm = Math.max(1e-6, ...f.resid), gp = Math.max(1e-6, ...f.peak), hm = Math.max(1e-6, ...f.heads.flat());
    const t0 = performance.now();
    pend = { f, t0, next: 0, gp, hm, rm };
    layers.forEach((ly, l) => {
      for (const p of allParts(ly)) tween(`f${p.obj.id}`, 0.6, 0, 520, (v) => (p.u.uHover.value = Math.max(v, hoverObjs.includes(p) ? 1 : 0)), reach(l));
    });
    pulses.push(t0);
    const word = text.trim() ? text : JSON.stringify(text).slice(1, -1);
    return new Promise<void>((done) => setTimeout(() => {
      if (live) {
        outTok.textContent = word;
        outTok.classList.remove("pop");
        void outTok.offsetWidth;
        outTok.classList.add("pop");
      }
      done();
    }, TRAVEL * 1000));
  }

  function arrive(l: number) {
    if (!pend) return;
    const { f, gp, hm, rm } = pend, ly = layers[l];
    radiusT[l] = f.resid[l] / rm;
    (f.heads[l] || []).forEach((v, h) => (liveH[l][h] = Math.sqrt(v / hm)));
    const want = liveN[l], s = Math.sqrt(f.peak[l] / gp);
    for (const k of want.keys()) want.set(k, 0);
    for (const [n, v] of f.top[l]) want.set(n, Math.sign(v) * Math.sqrt(Math.abs(v)) * s);
    if (f.lens[l] != null) ly.lens.textContent = JSON.stringify(f.lens[l]).slice(1, -1);
  }

  function liveEnd() {
    live = false;
    pend = null;
    outTok.textContent = "";
  }

  function liveStep(dt: number) {
    if (!live) return;
    while (pend && pend.next < L && performance.now() - pend.t0 >= reach(pend.next)) arrive(pend.next++);
    const k = 1 - Math.exp(-dt * 10);
    for (let l = 0; l < L; l++) radius[l] += (radiusT[l] - radius[l]) * k;
    shapeStream();
    layers.forEach((ly, l) => {
      ly.heads.ranges.forEach(([i0, i1], h) => {
        const cur = ly.attn.a[i0];
        ly.attn.a.fill(cur + (liveH[l][h] - cur) * k, i0, i1);
      });
      (ly.attn.geo.attributes.a as THREE.BufferAttribute).needsUpdate = true;
      const want = liveN[l];
      if (!want.size) return;
      for (const [n, t] of want) {
        const v = ly.mlp.a[n] + (t - ly.mlp.a[n]) * k;
        ly.mlp.a[n] = v;
        if (t === 0 && Math.abs(v) < 0.01) ((ly.mlp.a[n] = 0), want.delete(n));
      }
      (ly.mlp.geo.attributes.a as THREE.BufferAttribute).needsUpdate = true;
    });
  }

  // ---------------------------------------------------------------- frame
  const v = new THREE.Vector3();
  let raf = 0, w = 0, hgt = 0, alive = true, spin = 0, spinning = false, last = performance.now();
  function size() {
    const r = host.getBoundingClientRect();
    if (!r.width || !r.height || (r.width === w && r.height === hgt)) return;
    w = r.width; hgt = r.height;
    renderer.setSize(w, hgt, false);
    renderer.domElement.style.width = `${w}px`;
    renderer.domElement.style.height = `${hgt}px`;
    camera.aspect = w / hgt;
    camera.updateProjectionMatrix();
    const dpr = renderer.getPixelRatio();
    G.uPx.value = (hgt * dpr) / (2 * Math.tan((camera.fov * Math.PI) / 360));
    G.uMin.value = 1.1 * dpr;
    G.uGlyph.value = 6 * dpr;
  }
  const ro = new ResizeObserver(size);
  ro.observe(host);

  function place(e: HTMLElement, x: number, y: number, z: number, show: boolean, dx = 0, dy = 0) {
    v.set(x, y, z).project(camera);
    const on = show && v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05;
    e.style.opacity = on ? "" : "0";
    if (on) e.style.transform = `translate(${((v.x + 1) / 2) * w + dx}px, ${((1 - v.y) / 2) * hgt + dy}px)`;
    return on ? ((v.x + 1) / 2) * w : null;
  }

  function labelsFrame() {
    const open = focus.level === "layer" || focus.level === "part" ? focus.layer : -1;
    let lastX = -1e9, lastLens = -1e9;
    layers.forEach((ly, l) => {
      const x = warpX(ly.x), dim = open < 0 || l === open || Math.abs(l - open) <= 2;
      const sx = place(ly.num, x + OFF.mlp, ly.bottom - 0.12, 0, dim && l !== open);
      if (sx !== null) {
        const ok = Math.abs(sx - lastX) > 22 || l === open;
        ly.num.style.opacity = ok ? "" : "0";
        if (ok) lastX = sx;
        ly.num.classList.toggle("on", l === open);
      }
      const ls = place(ly.lens, x + OFF.attn, ly.heads.top + 0.14, 0, (!!run || live) && !!ly.lens.textContent && dim && focus.level !== "part");
      if (ls !== null) {
        const wd = ly.lens.offsetWidth;
        const ok = ls - wd / 2 > lastLens + 4 || l === open;
        ly.lens.style.opacity = ok ? "" : "0";
        if (ok) lastLens = ls + wd / 2;
      }
    });
    place(endLabels[0], warpX(X0), -1.78, 0, focus.level !== "part");
    place(endLabels[1], warpX(X1), -1.78, 0, focus.level !== "part");
    place(outTok, warpX(X1), 1.95, 0, live && !!outTok.textContent);
    partLabels.forEach(({ p, e }) => {
      if (open < 0) return place(e, 0, 0, 0, false);
      const ly = layers[open], x = warpX(ly.x + OFF[p]);
      const y = p === "attn" ? ly.heads.top + 0.16 : p === "mlp" ? ly.bottom - 0.32 : -0.34;
      place(e, x, y, 0, focus.level === "layer" || (focus.level === "part" && focus.part === p && p.startsWith("norm")));
    });
  }

  function frame(now: number) {
    raf = requestAnimationFrame(frame);
    if (!host.offsetWidth) return;
    size();
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;

    for (const [k, t] of tws) {
      const u = Math.max(0, Math.min(1, (now - t.t0) / t.dur));
      if (now < t.t0) continue;
      t.set(t.from + (t.to - t.from) * ease(u));
      if (u >= 1) tws.delete(k);
    }

    if (fly) {
      const u = ease(Math.min(1, (now - fly.t0) / fly.dur));
      camera.position.lerpVectors(fly.from.pos, fly.to.pos, u);
      look.lerpVectors(fly.from.at, fly.to.at, u);
      if (controls) controls.target.copy(look);
      camera.lookAt(look);
      if (u >= 1) ((fly = null), controls && (controls.enabled = true));
    } else if (preview) {
      // Home's preview: turns slowly round the model while hovered
      if (spinning) spin += dt * 0.25;
      const p = pose({ level: "model" }), off = p.pos.clone().sub(p.at);
      off.applyAxisAngle(new THREE.Vector3(0, 1, 0), Math.sin(spin) * 0.5);
      camera.position.copy(p.at).add(off);
      camera.lookAt(p.at);
    }
    controls?.update();

    liveStep(dt);

    // a slow pulse along the stream, so the model reads as running; while it writes, one pulse per token
    const pulse = (now / 1000) * 0.18 % 1;
    while (pulses.length && now - pulses[0] > TRAVEL * 1300) pulses.shift();
    const fronts = pulses.map((p) => (now - p) / 1000 / TRAVEL);
    const sb = stream.b;
    for (let i = 0; i < sb.length; i++) {
      const f = (streamX[i] - X0) / len;
      let b = live ? 0.3 : 0.3 + 0.38 * Math.exp(-(((f - pulse) * 14) ** 2));
      for (const p of fronts) b = Math.max(b, 0.35 + 0.65 * Math.exp(-(((f - p) * 22) ** 2)));
      sb[i] = b;
    }
    (stream.geo.attributes.b as THREE.BufferAttribute).needsUpdate = true;

    if (!preview) labelsFrame();
    renderer.render(scene, camera);
  }

  // start: the Home preview's pose, then fly in
  size();
  const start = pose({ level: "model" });
  camera.position.copy(preview ? start.pos : start.pos.clone().sub(start.at).multiplyScalar(1.6).add(start.at).add(new THREE.Vector3(0, 6, 0)));
  look.copy(start.at);
  camera.lookAt(look);
  if (controls) controls.target.copy(look);
  if (!preview) {
    flyTo(start, 1500);
    layers.forEach((ly, l) => allParts(ly).forEach((p) => ((p.u.uDim.value = 0), toU(`d${p.obj.id}`, p.u.uDim, 1, 700, 100 + l * (900 / L)))));
  }
  raf = requestAnimationFrame(frame);

  return {
    setFocus,
    focusNow: () => focus,
    setNorms,
    setRun,
    setToken,
    liveStart,
    liveFrame,
    liveEnd,
    select,
    setSpin: (on: boolean) => (spinning = on),
    /** Re-centres the camera on what's in focus, after the user has orbited away. */
    reset: () => flyTo(pose(focus), 700),
    dispose() {
      if (!alive) return;
      alive = false;
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls?.dispose();
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointerleave", onLeave);
      scene.traverse((o) => {
        if (o instanceof THREE.Points) (o.geometry.dispose(), (o.material as THREE.Material).dispose());
      });
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
      labels.remove();
    },
  };
}

export type Scene = ReturnType<typeof createScene>;
