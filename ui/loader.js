// Loading scene in the "squares" style: the poster's sun and engraved hills, mirrored in a rippling lake.
// Each grid cell is a square whose area follows the scene's brightness there.
// The sun rises with load progress and pulses once when the model is ready; now and then a few stars
// link up and a spark runs through them, like a signal passing through a network's layers.
const W = 96, H = 32, HORIZON = 20;          // scene units (x: columns, y: rows twice as tall)
const SX = 2, SY = 4;                         // grid cells per scene unit, so cells come out square
const COLS = W * SX, ROWS = H * SY, CELL = 3.6;
const SUN = { x: W / 2, y: 11, r: 6.2 };
const SUN_LOW = HORIZON + 3.5, SUN_HIGH = 11; // sun centre before loading starts and once it finishes

const hash = (x, y) => { const s = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return s - Math.floor(s); };
const ridges = [
  (x) => HORIZON - 2.6 - 1.8 * Math.sin(x * 0.05 + 3.1) - 0.7 * Math.sin(x * 0.17 + 1), // front
  (x) => HORIZON - 6 - 2.2 * Math.sin(x * 0.065 + 1.2) - 0.9 * Math.sin(x * 0.15),     // back
];

let pulseAt = null; // time the ready pulse started

const DRIFT = [0.9, 0.35]; // scene units per second: the near hill slides faster than the far one, for depth

function hill(x, y, t) {
  for (let k = 0; k < ridges.length; k++) {
    const v = y - ridges[k](x + t * DRIFT[k]);
    if (v < 0) continue;
    if (v < 0.3) return k ? 0.6 : 0.8;
    // engraved contour lines that follow the ridge; the far hill is lighter
    return k ? ((v % 2) < 0.3 ? 0.4 : 0) : ((v % 1.25) < 0.3 ? 0.65 : 0.08);
  }
  return null;
}

function sky(x, y, t, gx, gy) {
  const d = Math.hypot((x - SUN.x) * 0.5, y - SUN.y);
  if (d < SUN.r) return 0.8 + 0.2 * Math.cos((d / SUN.r) * 1.4) - 0.04 * Math.sin(t * 2);
  const out = d - SUN.r;
  // rings fade out near the top and sides instead of being cut off by the frame
  const edge = Math.min(1, y / 5) * Math.min(1, Math.min(x, W - x) / 12);
  const ring = Math.pow(Math.max(0, Math.cos(((out - t * 2.4) * Math.PI) / 3.2)), 10) * Math.exp(-out / 13) * edge;
  const glow = 0.4 * Math.exp(-out / 2.5);
  const haze = 0.16 * Math.max(0, (y - HORIZON + 7) / 7);
  let b = Math.max(ring * 0.85, glow) + haze;
  if (pulseAt !== null) {
    // one bright ring racing outward when the model is ready
    const k = (t - pulseAt) / 1.1;
    if (k < 1) b = Math.max(b, Math.exp(-((out - k * 40) ** 2) / 4) * (1 - k) * edge);
  }
  if (b > 0.06) return b;
  const h = hash(gx, gy);
  if (h < 0.0016 && y < HORIZON - 8) return 0.45 + 0.4 * Math.sin(t * 2.5 + h * 9e4);
  return 0;
}

function above(x, y, t, gx, gy) {
  x = Math.max(0, Math.min(W - 1, x));
  return hill(x, y, t) ?? sky(x, y, t, gx, gy);
}

function cell(gx, gy, t) {
  const x = gx / SX, y = gy / SY;
  if (y < HORIZON) return above(x, y, t, gx, gy);
  if (y < HORIZON + 0.25) return gx % 2 ? 0 : 0.55; // dotted horizon
  const depth = y - HORIZON;
  if (Math.sin(x * 0.12 + depth * 3.4 - t * 3.2) * Math.sin(x * 0.05 - t) > 0.75) return 0; // breaks in the water
  const wobble = Math.sin(depth * 1.3 - t * 3) * (0.4 + depth * 0.18) + Math.sin(x * 0.28 + t * 1.6) * 0.4;
  const b = above(x + wobble, HORIZON - depth, t, gx, gy) * 0.55;
  // sunlight glinting on the water: a narrow column under the sun whose squares flicker on and off,
  // brighter the higher the sun has risen
  const hw = 5 + depth * 0.7, dx = Math.abs(x + wobble * 0.6 - SUN.x);
  if (dx < hw) {
    const risen = Math.max(0.25, (SUN_LOW - SUN.y) / (SUN_LOW - SUN_HIGH));
    const gx2 = gx >> 1; // glints are short two-square dashes
    const glint = gy % 2 === 0 && hash(gx2, gy + Math.floor(t * 5 + hash(gx2, gy) * 5) * 131) > 0.78;
    const fall = 1 - (dx / hw) ** 2;
    if (glint) return Math.max(b, (1 - depth * 0.025) * (0.55 + 0.45 * fall) * risen);
    return b * (1 - 0.45 * fall * risen); // darker water between glints makes them read
  }
  return b;
}

// ------------------------------------------------------------------ neuron stars
// fixed "neuron" stars on either side of the sun, as grid cells
const NEURONS = [];
for (let i = 0; NEURONS.length < 18 && i < 400; i++) {
  const x = 4 + hash(i, 1) * (W - 8), y = 1.5 + hash(i, 2) * 9;
  if (Math.abs(x - SUN.x) < 21) continue;
  if (NEURONS.some((n) => Math.hypot((n.x - x) * 0.5, n.y - y) < 2.4)) continue;
  NEURONS.push({ x, y, gx: Math.round(x * SX), gy: Math.round(y * SY) });
}

const SIGNAL = { len: 2.8, gap: 0.9 }; // seconds a signal lasts, and the pause before the next
let signal = null;

function newSignal(t, seed) {
  // a chain of 3 to 5 stars on one side, left to right like layers
  const left = hash(seed, 3) < 0.5;
  const pool = NEURONS.filter((n) => (n.x < SUN.x) === left).sort((a, b) => a.x - b.x);
  const start = Math.floor(hash(seed, 4) * Math.max(1, pool.length - 3));
  const path = [pool[start]];
  const want = 3 + Math.floor(hash(seed, 5) * 3);
  while (path.length < want) {
    const cur = path[path.length - 1];
    const next = pool.filter((n) => n.x > cur.x && !path.includes(n))
      .sort((a, b) => Math.hypot(a.x - cur.x, (a.y - cur.y) * 2) - Math.hypot(b.x - cur.x, (b.y - cur.y) * 2))[0];
    if (!next) break;
    path.push(next);
  }
  if (path.length < 3) return null;
  // dotted links between consecutive stars, every other cell
  const links = [], seg = [0];
  for (let i = 1; i < path.length; i++) {
    const a = path[i - 1], b = path[i];
    const steps = Math.max(Math.abs(b.gx - a.gx), Math.abs(b.gy - a.gy));
    for (let s = 2; s < steps - 1; s += 2) links.push([Math.round(a.gx + ((b.gx - a.gx) * s) / steps), Math.round(a.gy + ((b.gy - a.gy) * s) / steps)]);
    seg.push(seg[i - 1] + steps);
  }
  return { start: t, path, links, seg };
}

function overlay(t) {
  const o = new Map();
  const put = (gx, gy, b) => {
    if (gx < 0 || gy < 0 || gx >= COLS || gy >= ROWS) return;
    const i = gy * COLS + gx;
    o.set(i, Math.max(o.get(i) || 0, b));
  };
  for (const n of NEURONS) put(n.gx, n.gy, 0.45 + 0.15 * Math.sin(t * 1.7 + n.x));

  if (!signal || t > signal.start + SIGNAL.len + SIGNAL.gap) signal = newSignal(t, Math.floor(t * 7));
  if (!signal) return o;
  const k = (t - signal.start) / SIGNAL.len;
  if (k < 0 || k > 1) return o;
  const fade = Math.min(1, k / 0.12, (1 - k) / 0.2);
  for (const [gx, gy] of signal.links) put(gx, gy, 0.5 * fade);
  // the spark runs along the chain between 15% and 80% of the signal's life
  const total = signal.seg[signal.seg.length - 1];
  const at = Math.max(0, Math.min(1, (k - 0.15) / 0.65)) * total;
  signal.path.forEach((n, i) => {
    const lit = at >= signal.seg[i] ? Math.max(0.35, 1 - (at - signal.seg[i]) / 40) : 0;
    put(n.gx, n.gy, Math.max(0.7 * fade, lit * fade));
    if (lit > 0.5) for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [2, 0], [-2, 0], [0, 2], [0, -2]])
      put(n.gx + dx, n.gy + dy, lit * (Math.abs(dx) + Math.abs(dy) > 1 ? 0.35 : 0.75) * fade);
  });
  if (k > 0.15 && k < 0.8) {
    const i = signal.seg.findIndex((s, j) => j > 0 && at <= s);
    const a = signal.path[i - 1], b = signal.path[i], f = (at - signal.seg[i - 1]) / (signal.seg[i] - signal.seg[i - 1] || 1);
    for (let tr = 0; tr < 6; tr++) {
      const ff = Math.max(0, f - tr * 0.05), px = Math.round(a.gx + (b.gx - a.gx) * ff), py = Math.round(a.gy + (b.gy - a.gy) * ff);
      put(px, py, 1 - tr * 0.15);
      if (tr === 0) for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) put(px + dx, py + dy, 0.7);
    }
  }
  return o;
}

// ------------------------------------------------------------------ drawing
function draw(ctx, t) {
  ctx.clearRect(0, 0, COLS * CELL, ROWS * CELL);
  const o = overlay(t), skyRows = HORIZON * SY;
  for (let gy = 0; gy < ROWS; gy++) {
    for (let gx = 0; gx < COLS; gx++) {
      let b = Math.min(1, cell(gx, gy, t));
      if (gy < skyRows) {
        const v = o.get(gy * COLS + gx);
        if (v !== undefined && hill(gx / SX, gy / SY, t) === null) b = Math.max(b, v);
      }
      if (b < 0.05) continue;
      const s = CELL * 0.9 * Math.sqrt(b);
      ctx.fillRect(gx * CELL + (CELL - s) / 2, gy * CELL + (CELL - s) / 2, s, s);
    }
  }
}

let raf = 0, target = 0, shown = 0;

/** Load progress, 0..1; the sun rises toward it. */
export function setProgress(p) {
  target = Math.max(target, Math.max(0, Math.min(1, p || 0)));
}

/** Finish the rise and play the ready pulse; resolves when it has played. */
export function finishScene() {
  if (!raf) return Promise.resolve();
  target = 1;
  return new Promise((done) => {
    pulseRequest = done;
  });
}
let pulseRequest = null;

export function startScene(canvas) {
  if (raf) return;
  target = shown = 0;
  pulseAt = null;
  signal = null;
  const dpr = devicePixelRatio || 1;
  canvas.width = COLS * CELL * dpr;
  canvas.height = ROWS * CELL * dpr;
  canvas.style.width = COLS * CELL + "px";
  canvas.style.height = ROWS * CELL + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = getComputedStyle(canvas).color;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const t0 = performance.now();
  let last = 0;
  const tick = (now) => {
    raf = requestAnimationFrame(tick);
    if (now - last < 66) return; // ~15 fps
    const dt = Math.min(0.2, (now - last) / 1000);
    last = now;
    const t = (now - t0) / 1000;
    shown += (target - shown) * Math.min(1, dt * (pulseRequest ? 5 : 1.8));
    const e = 1 - (1 - shown) ** 2;
    SUN.y = SUN_LOW + (SUN_HIGH - SUN_LOW) * e;
    if (pulseRequest && pulseAt === null && shown > 0.97) pulseAt = t;
    if (pulseAt !== null && t - pulseAt > 1.1 && pulseRequest) {
      const done = pulseRequest;
      pulseRequest = null;
      done();
    }
    draw(ctx, t);
    if (still && !pulseRequest) stopScene();
  };
  raf = requestAnimationFrame(tick);
}

export function stopScene() {
  cancelAnimationFrame(raf);
  raf = 0;
  if (pulseRequest) {
    const done = pulseRequest;
    pulseRequest = null;
    done();
  }
}
