import { drawMarks } from "./glyphs";
import type { Put } from "./glyphs";
import { compassStar } from "./mark";

// The start screen's scene in ASCII marks: the model arriving, drawn as Lodestar's night. The lodestar (the app's
// mark) stands over a mountain range mirrored in a still lake, and around it is one ring for each of the model's
// layers, innermost first. A ring waits as a faint dotted orbit; as its layer's weights arrive it draws itself
// round from the top, a bright head running ahead, and the stars on it wake and draw out into trails. When every
// layer is in, a wave runs out through the rings, and the workspace follows.
const COLS = 192, ROWS = 128, CELL = 3.6;      // grid cells (square)
const HORIZON = 80;                             // the lake's edge, as a row
const STAR = { x: COLS / 2, y: 42 };            // the lodestar, at the centre of the rings
const R0 = 17, R1 = 150;                        // the innermost and outermost ring's radius; the outer ones run off the sky
const SPIN = 0.05;                              // radians a second the lit sky turns
const SCALE = 1.45;                             // cells per unit of the mark's 31-unit design
const WAVE = 1.3;                               // seconds the ready wave takes to cross the rings

const hash = (x: number, y: number) => { const s = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return s - Math.floor(s); };

let pulseAt: number | null = null; // time the current wave (wake or ready) started
let wakeRequest = false;

// the stars, spread over the sky; each rides the ring nearest it once the layers are known
const STARS = Array.from({ length: 200 }, (_, i) => ({
  f: hash(i, 1) ** 0.8, // how far out, 0..1 across the rings
  a: hash(i, 2) * Math.PI * 2,
  head: 0.3 + hash(i, 3) ** 2.2 * 0.7, // most faint, a few bright
  tw: hash(i, 4) * 6.28,
}));

const ridgeOf = (gx: number) => ({
  back: Math.round(HORIZON - 10 + 3 * Math.sin(gx * 0.045 + 1.3) + 2 * Math.sin(gx * 0.13)),
  front: Math.round(HORIZON - 3 - 5 * Math.exp(-((gx - 40) ** 2) / 260) - 7 * Math.exp(-((gx - 156) ** 2) / 340) + 0.8 * Math.sin(gx * 0.31)),
});

const RIDGES = Array.from({ length: COLS }, (_, gx) => ridgeOf(gx));

// ------------------------------------------------------------------ drawing
/** lit: how far round each ring has drawn itself (0..1 per layer); wave: how far the wave is across, 0..1, or null. */
function draw(ctx: CanvasRenderingContext2D, t: number, lit: number[], wave: number | null) {
  const grid = new Float32Array(COLS * ROWS);
  const L = lit.length;
  const ring = (i: number) => R0 + (R1 - R0) * ((i + 0.5) / L) ** 1.15; // a little closer in toward the lodestar
  const done = lit.reduce((a, b) => a + b, 0) / L;

  const sky: Put = (gx, gy, b) => { // only above the mountains
    gx = Math.round(gx);
    gy = Math.round(gy);
    if (gx < 0 || gy < 0 || gx >= COLS || gy >= HORIZON) return;
    const rg = RIDGES[gx];
    if (gy >= Math.min(rg.back, rg.front)) return;
    const i = gy * COLS + gx;
    if (b > grid[i]) grid[i] = b;
  };

  const at = (r: number, a: number): [number, number] => [STAR.x + Math.cos(a) * r, STAR.y + Math.sin(a) * r];
  const TOP = -Math.PI / 2;

  for (let i = 0; i < L; i++) {
    const r = ring(i), l = lit[i];
    const flash = wave === null ? 0 : Math.exp(-(((wave * (L + 6) - 3 - i) / 1.6) ** 2)); // the wave passing this ring

    // waiting: a faint dotted orbit
    const dots = Math.max(8, Math.round((2 * Math.PI * r) / 8));
    for (let k = 0; k < dots; k++) {
      const a = (k / dots) * 2 * Math.PI + i * 0.37;
      sky(...at(r, a), 0.1 + 0.03 * Math.sin(t * 1.5 + k + i));
    }

    // arriving: the ring draws itself round from the top, its head bright, the arc behind it settling to a fine line
    if (l > 0 || flash > 0.02) {
      const sweep = l * 2 * Math.PI, step = 0.6 / r;
      for (let u = 0; u <= sweep; u += step) {
        const behind = (sweep - u) * r; // cells behind the head
        const head = l < 1 ? Math.exp(-behind / 16) : 0;
        if (head > 0.1 || Math.round(u * r) % 3 === 0) sky(...at(r, TOP + u), 0.22 + 0.78 * head);
      }

      if (flash > 0.02) for (let u = 0; u < 2 * Math.PI; u += step) sky(...at(r, TOP + u), 0.24 + 0.7 * flash);
    }
  }

  // the stars: points at rest; once their ring is in they turn with the sky and draw out into trails
  for (const s of STARS) {
    const i = Math.min(L - 1, Math.floor(s.f * L)), l = lit[i], r = ring(i);
    const bright = s.head * (0.85 + 0.15 * Math.sin(t * 2.2 + s.tw));
    const head = s.a + t * SPIN * l, len = 0.02 + l * (0.25 + s.head * 0.45);

    for (let u = 0; u <= len; u += 0.6 / r) {
      const k = 1 - u / len;
      sky(...at(r, head - u), (0.25 + 0.75 * l) * bright * k ** 1.4);
    }
  }

  // the lodestar itself, brightening as the model arrives
  const shine = 0.55 + 0.45 * done;
  for (let dy = -20; dy <= 20; dy++)
    for (let dx = -20; dx <= 20; dx++) {
      const b = compassStar(dx / SCALE, dy / SCALE) * shine;
      if (b > 0.06) sky(STAR.x + dx, STAR.y + dy, b);
    }

  // two ranges: a dim far ridge and a near one lit along its edge, the near one filled with a few dark marks
  for (let gx = 0; gx < COLS; gx++) {
    const { back, front } = RIDGES[gx];
    for (let gy = Math.min(back, front); gy < HORIZON; gy++) {
      let b = 0;
      if (gy === front) b = hash(gx, 7) < 0.15 ? 0.6 : 0.42;
      else if (gy === back && back < front) b = 0.22;
      else if (gy > front) b = (gx + gy) % 5 === 0 ? 0.07 : 0;
      grid[gy * COLS + gx] = b;
    }
  }

  for (let gx = 0; gx < COLS; gx += 3) grid[HORIZON * COLS + gx] = 0.22; // the lake's edge, dotted

  // the lake: everything above, mirrored and rippled, dimmer with depth; the lodestar's light runs down it in glints
  for (let gy = HORIZON + 1; gy < ROWS; gy++) {
    const depth = gy - HORIZON;
    for (let gx = 0; gx < COLS; gx++) {
      if (Math.sin(gx * 0.06 + depth * 0.85 - t * 3.2) * Math.sin(gx * 0.025 - t) > 0.75) continue; // breaks in the water

      const wob = Math.round(Math.sin(depth * 0.33 - t * 3) * (1 + depth * 0.06) + Math.sin(gx * 0.14 + t * 1.6) * 0.8);
      const sx = Math.max(0, Math.min(COLS - 1, gx + wob)), sy = HORIZON - depth;
      let b = sy >= 0 ? grid[sy * COLS + sx] * 0.5 * (1 - depth / 70) : 0;

      const dx = Math.abs(gx + wob * 0.6 - STAR.x), hw = 3 + depth * 0.35;
      if (dx < hw && gy % 2 === 0) {
        const gx2 = gx >> 1;
        if (hash(gx2, gy + Math.floor(t * 5 + hash(gx2, gy) * 5) * 131) > 0.8) b = Math.max(b, (0.4 + 0.5 * done) * (1 - dx / hw) * (1 - depth / 60));
      }

      grid[gy * COLS + gx] = b;
    }
  }

  drawMarks(ctx, grid, COLS, ROWS, CELL);
}

let raf = 0, layers = 28, target = 0, front = 0, measured: number | null = null;
let pulseRequest: (() => void) | null = null;

/** Load progress, 0..1: until the layers report in, the rings fill by it. */
export function setProgress(p: number) {
  const f = Math.max(0, Math.min(1, ((p || 0) - 0.04) / 0.86));
  if (measured === null) target = Math.max(target, f * layers);
}

/** How many layers the model has and how far each has arrived (0..1), as the server counts them. */
export function setLayers(count: number | null, lit: number[] | null) {
  if (count && count !== layers) ((front *= count / layers), (target *= count / layers), (layers = count));
  if (lit) ((measured = lit.reduce((a, b) => a + b, 0)), (target = Math.max(target, measured)));
}

/** A wave out through the rings from the lodestar, as a load begins. */
export function wake() {
  wakeRequest = true;
}

/** Fill the last rings and send the ready wave out through them; resolves when it has crossed. */
export function finishScene() {
  if (!raf) return Promise.resolve();

  target = layers;
  pulseAt = null; // a wake wave still in flight gives way to the ready one

  return new Promise<void>((done) => {
    pulseRequest = done;
  });
}

export function startScene(canvas: HTMLCanvasElement) {
  if (raf) return;

  layers = 28;
  target = front = 0;
  measured = pulseAt = null;

  const dpr = devicePixelRatio || 1;
  canvas.width = COLS * CELL * dpr;
  canvas.height = ROWS * CELL * dpr;
  canvas.style.width = COLS * CELL + "px";
  canvas.style.height = ROWS * CELL + "px";
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = getComputedStyle(canvas).color;

  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const t0 = performance.now();
  let last = 0;

  const tick = (now: number) => {
    raf = requestAnimationFrame(tick);
    if (now - last < 66) return; // ~15 fps

    const dt = Math.min(0.2, (now - last) / 1000);
    last = now;
    const t = (now - t0) / 1000;
    if (wakeRequest) ((pulseAt = t), (wakeRequest = false));

    // the front eases toward what's arrived, quicker at the end so the last rings close before the wave
    front += (target - front) * Math.min(1, dt * (pulseRequest ? 6 : 2.2));
    if (still) front = target;
    const lit = Array.from({ length: layers }, (_, i) => Math.max(0, Math.min(1, front - i)));

    if (pulseRequest && pulseAt === null && front > layers - 0.05) pulseAt = t;
    const wave = pulseAt === null ? null : (t - pulseAt) / WAVE;
    if (wave !== null && wave > 1) {
      pulseAt = null;
      if (pulseRequest) {
        const done = pulseRequest;
        pulseRequest = null;
        done();
      }
    }

    draw(ctx, t, lit, wave !== null && wave <= 1 ? wave : null);
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
