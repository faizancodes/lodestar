// The WGSL for one Qwen3 forward pass. Weights stay as the bf16 the checkpoint ships, two to a u32, and are widened
// to f32 as they're read; activations are f32 and the KV cache is f16, two to a u32. Shapes are override constants,
// so one shader serves every matrix; how many rows to run comes from a small uniform (Step) per pass.

const COMMON = /* wgsl */ `
struct Step { T: u32, pos0: u32, row0: u32, pad: u32 }
fn bf(w: u32, hi: bool) -> f32 { return bitcast<f32>(select(w << 16u, w & 0xffff0000u, hi)); }
`;

// a token's row of the embedding, which is split into chunks of CHUNK rows to stay under the binding size limit
export const EMBED = COMMON + /* wgsl */ `
override H: u32; override CHUNK: u32;
@group(0) @binding(0) var<uniform> st: Step;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> e0: array<u32>;
@group(0) @binding(3) var<storage, read> e1: array<u32>;
@group(0) @binding(4) var<storage, read> e2: array<u32>;
@group(0) @binding(5) var<storage, read> e3: array<u32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = g.y; let i = g.x;
  if (t >= st.T || i >= H) { return; }
  let id = ids[t]; let c = id / CHUNK; let k = ((id % CHUNK) * H + i) >> 1u;
  var w: u32;
  switch c { case 0u: { w = e0[k]; } case 1u: { w = e1[k]; } case 2u: { w = e2[k]; } default: { w = e3[k]; } }
  out[t * H + i] = bf(w, (i & 1u) == 1u);
}`;

// RMSNorm of rows row0 .. row0+T-1 of x, written compactly from row 0 of out
export const RMSNORM = COMMON + /* wgsl */ `
override D: u32; override EPS: f32 = 1e-6;
@group(0) @binding(0) var<uniform> st: Step;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let t = wg.x;
  if (t >= st.T) { return; }
  let base = (st.row0 + t) * D;
  var s = 0.0;
  for (var i = l.x; i < D; i += 256u) { let v = x[base + i]; s += v * v; }
  part[l.x] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
  let r = inverseSqrt(part[0] / f32(D) + EPS);
  for (var i = l.x; i < D; i += 256u) { out[t * D + i] = x[base + i] * r * bf(w[(i) >> 1u], ((i) & 1u) == 1u); }
}`;

// y[t, ROW0 + o] (+)= Σ_i x[t, i] W[o, i], for T tokens: a workgroup per output row and per tile of 8 tokens, so each
// weight is read once per tile
export const MATMUL = COMMON + /* wgsl */ `
override IN: u32; override YS: u32; override ROW0: u32 = 0u; override ACC: bool = false;
const TT = 8u;
@group(0) @binding(0) var<uniform> st: Step;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<array<f32, 8>, 64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let o = wg.x; let t0 = wg.y * TT;
  if (t0 >= st.T) { return; }
  let nt = min(TT, st.T - t0);
  var acc: array<f32, 8>;
  let half = IN >> 1u; let row = o * half;
  for (var k = l.x; k < half; k += 64u) {
    let w = W[row + k]; let w0 = bf(w, false); let w1 = bf(w, true);
    for (var j = 0u; j < nt; j++) {
      let xb = (t0 + j) * IN + 2u * k;
      acc[j] += w0 * x[xb] + w1 * x[xb + 1u];
    }
  }
  for (var j = 0u; j < 8u; j++) { part[l.x][j] = acc[j]; }
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (l.x < s) { for (var j = 0u; j < 8u; j++) { part[l.x][j] += part[l.x + s][j]; } }
    workgroupBarrier();
  }
  if (l.x < nt) {
    let idx = (t0 + l.x) * YS + ROW0 + o;
    if (ACC) { y[idx] += part[0][l.x]; } else { y[idx] = part[0][l.x]; }
  }
}`;

// Qwen3's per-head RMSNorm on q or k, then rotary position (rotate-half), in place: [T, NH, 128]
export const QKROPE = COMMON + /* wgsl */ `
override NH: u32; override THETA: f32 = 1000000.0;
@group(0) @binding(0) var<uniform> st: Step;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
var<workgroup> part: array<f32, 64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let h = wg.x; let t = wg.y;
  if (t >= st.T) { return; }
  let base = (t * NH + h) * 128u; let i = l.x;
  let a = b[base + i]; let c = b[base + i + 64u];
  part[i] = a * a + c * c;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (i < k) { part[i] += part[i + k]; } workgroupBarrier(); }
  let r = inverseSqrt(part[0] / 128.0 + 1e-6);
  let na = a * r * bf(w[(i) >> 1u], ((i) & 1u) == 1u); let nc = c * r * bf(w[(i + 64u) >> 1u], ((i + 64u) & 1u) == 1u);
  let ang = f32(st.pos0 + t) * pow(THETA, -f32(2u * i) / 128.0);
  let co = cos(ang); let si = sin(ang);
  b[base + i] = na * co - nc * si;
  b[base + i + 64u] = nc * co + na * si;
}`;

// this step's keys and values into the layer's cache, at their positions
export const KVSTORE = COMMON + /* wgsl */ `
override KVH: u32;
@group(0) @binding(0) var<uniform> st: Step;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> kc: array<u32>;
@group(0) @binding(4) var<storage, read_write> vc: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let h = wg.x; let t = wg.y;
  if (t >= st.T) { return; }
  let src = (t * KVH + h) * 128u + 2u * l.x;
  let dst = ((st.pos0 + t) * KVH + h) * 64u + l.x;
  kc[dst] = pack2x16float(vec2f(k[src], k[src + 1u]));
  vc[dst] = pack2x16float(vec2f(v[src], v[src + 1u]));
}`;

// causal attention for each query head and token, over the cache up to the token's own position (grouped: GROUP query
// heads share a key/value head)
export const ATTN = COMMON + /* wgsl */ `
override NH: u32; override KVH: u32; override CTX: u32;
@group(0) @binding(0) var<uniform> st: Step;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<u32>;
@group(0) @binding(3) var<storage, read> vc: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
var<workgroup> sc: array<f32, CTX>;
var<workgroup> red: array<f32, 64>;
var<workgroup> qs: array<f32, 128>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let h = wg.x; let t = wg.y;
  if (t >= st.T) { return; }
  let kvh = h / (NH / KVH); let n = st.pos0 + t + 1u; let i = l.x;
  let qb = (t * NH + h) * 128u;
  qs[i] = q[qb + i]; qs[i + 64u] = q[qb + i + 64u];
  workgroupBarrier();
  var mx = -1e30;
  for (var j = i; j < n; j += 64u) {
    let kb = (j * KVH + kvh) * 64u;
    var d = 0.0;
    for (var p = 0u; p < 64u; p++) { let kk = unpack2x16float(kc[kb + p]); d += qs[2u * p] * kk.x + qs[2u * p + 1u] * kk.y; }
    d *= 0.08838834764831845;
    sc[j] = d; mx = max(mx, d);
  }
  red[i] = mx;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (i < k) { red[i] = max(red[i], red[i + k]); } workgroupBarrier(); }
  let m = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = i; j < n; j += 64u) { let e = exp(sc[j] - m); sc[j] = e; sum += e; }
  red[i] = sum;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (i < k) { red[i] += red[i + k]; } workgroupBarrier(); }
  let inv = 1.0 / red[0];
  var acc = vec2f(0.0);
  for (var j = 0u; j < n; j++) { acc += sc[j] * unpack2x16float(vc[(j * KVH + kvh) * 64u + i]); }
  o[qb + 2u * i] = acc.x * inv;
  o[qb + 2u * i + 1u] = acc.y * inv;
}`;

// silu(gate) * up, plus each neuron's push (zero when nothing's pushed): the input to down_proj, where the app's
// neurons live
export const SWIGLU = COMMON + /* wgsl */ `
override N: u32;
@group(0) @binding(0) var<uniform> st: Step;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> u: array<f32>;
@group(0) @binding(3) var<storage, read> add: array<f32>;
@group(0) @binding(4) var<storage, read_write> a: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gi: vec3u) {
  let t = gi.y; let i = gi.x;
  if (t >= st.T || i >= N) { return; }
  let k = t * N + i; let x = g[k];
  a[k] = x / (1.0 + exp(-x)) * u[k] + add[i];
}`;

// take coef times the refusal direction's component out of each row of the residual stream
export const ERASE = COMMON + /* wgsl */ `
override D: u32;
struct Abl { coef: f32, p0: f32, p1: f32, p2: f32 }
@group(0) @binding(0) var<uniform> st: Step;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<uniform> ab: Abl;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) l: vec3u) {
  let t = wg.x;
  if (t >= st.T) { return; }
  let base = t * D;
  var s = 0.0;
  for (var i = l.x; i < D; i += 256u) { s += x[base + i] * v[i]; }
  part[l.x] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (l.x < k) { part[l.x] += part[l.x + k]; } workgroupBarrier(); }
  let p = part[0] * ab.coef;
  for (var i = l.x; i < D; i += 256u) { x[base + i] -= p * v[i]; }
}`;

// every down_proj column's norm ([D, N] weights): how far a neuron's unit activation moves the residual stream
export const COLNORM = COMMON + /* wgsl */ `
override D: u32; override N: u32;
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let n = g.x;
  if (n >= N) { return; }
  var s = 0.0;
  for (var i = 0u; i < D; i++) { let w = bf(W[(i * N + n) >> 1u], ((i * N + n) & 1u) == 1u); s += w * w; }
  out[n] = sqrt(s);
}`;

// one neuron's down_proj column, into row `row` of out: what it writes into the residual stream
export const COLUMN = COMMON + /* wgsl */ `
override D: u32; override N: u32;
struct Col { n: u32, row: u32, sign: f32, pad: u32 }
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<uniform> c: Col;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= D) { return; }
  out[c.row * D + i] = c.sign * bf(W[(i * N + c.n) >> 1u], ((i * N + c.n) & 1u) == 1u);
}`;
