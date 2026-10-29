import { drawMarks } from "./glyphs.js";
import { compassStar } from "./mark.js";

// The start screen's scene in ASCII marks: Lodestar's night as a long exposure. The lodestar (the app's mark)
// stands over a mountain range mirrored in a still lake; around it the other stars are points at rest and draw
// out into longer trails as the load runs, the way a long exposure builds, while the lodestar climbs and brightens.
// A ring of light leaves it when a load begins and again when the model is ready.
const COLS = 192, ROWS = 128, CELL = 3.6;      // grid cells (square)
const HORIZON = 80;                             // the lake's edge, as a row
const STAR = { x: COLS / 2, y: 48 };            // the lodestar; its row rises with the load
const STAR_LOW = 48, STAR_HIGH = 30;
const SPIN = 0.018;                              // radians a second the sky turns around the lodestar
const SCALE = 1.2;                              // cells per unit of the mark's 31-unit design

const hash = (x, y) => { const s = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return s - Math.floor(s); };

let pulseAt = null; // time the current ring (wake or ready) started
let wakeRequest = false;

const STARS = Array.from({ length: 230 }, (_, i) => ({
  r: 5 + hash(i, 1) ** 1.15 * 150,
  a: hash(i, 2) * Math.PI * 2,
  head: 0.24 + hash(i, 3) ** 2.4 * 0.76, // most faint, a few bright
  tw: hash(i, 4) * 6.28,
}));
const ridgeOf = (gx) => ({
  back: Math.round(HORIZON - 10 + 3 * Math.sin(gx * 0.045 + 1.3) + 2 * Math.sin(gx * 0.13)),
  front: Math.round(HORIZON - 3 - 5 * Math.exp(-((gx - 40) ** 2) / 260) - 7 * Math.exp(-((gx - 156) ** 2) / 340) + 0.8 * Math.sin(gx * 0.31)),
});
const RIDGES = Array.from({ length: COLS }, (_, gx) => ridgeOf(gx));

// ------------------------------------------------------------------ drawing
function draw(ctx, t, exposure) {
  const grid = new Float32Array(COLS * ROWS);
  const sky = (gx, gy, b) => { // only above the mountains
    if (gx < 0 || gy < 0 || gx >= COLS || gy >= HORIZON) return;
    const rg = RIDGES[gx];
    if (gy >= Math.min(rg.back, rg.front)) return;
    const i = gy * COLS + gx;
    if (b > grid[i]) grid[i] = b;
  };
  // trails: an arc behind each star as long as the exposure so far, brightest at its head
  const len = 0.16 + exposure * 0.9;
  for (const s of STARS) {
    const head = s.a + t * SPIN, step = 0.7 / s.r;
    const bright = s.head * (0.85 + 0.15 * Math.sin(t * 2.2 + s.tw));
    for (let u = 0; u <= len; u += step) {
      const k = 1 - u / len;
      sky(Math.round(STAR.x + Math.cos(head - u) * s.r), Math.round(STAR.y + Math.sin(head - u) * s.r), bright * k ** 1.5);
    }
  }
  // the lodestar itself, brightening as it climbs
  const lit = 0.55 + 0.45 * exposure;
  for (let dy = -20; dy <= 20; dy++)
    for (let dx = -20; dx <= 20; dx++) {
      const b = compassStar(dx / SCALE, dy / SCALE) * lit;
      if (b > 0.06) sky(Math.round(STAR.x) + dx, Math.round(STAR.y) + dy, b);
    }
  // a ring of light racing out from it (wake and ready)
  if (pulseAt !== null) {
    const k = (t - pulseAt) / 1.1;
    if (k < 1)
      for (let gy = 0; gy < HORIZON; gy++)
        for (let gx = 0; gx < COLS; gx++) {
          const d = Math.hypot(gx - STAR.x, gy - STAR.y) - k * 120;
          if (Math.abs(d) < 4) sky(gx, gy, Math.exp(-(d * d) / 3) * (1 - k));
        }
  }
  // two ranges: a dim far ridge and a near one lit along its edge, the near one filled with a few dark marks
  for (let gx = 0; gx < COLS; gx++) {
    const { back, front } = RIDGES[gx];
    for (let gy = Math.min(back, front); gy < HORIZON; gy++) {
      let b = 0;
      if (gy === front) b = hash(gx, 7) < 0.2 ? 0.95 : 0.72;
      else if (gy === back && back < front) b = 0.36;
      else if (gy > front) b = (gx + gy) % 4 === 0 ? 0.09 : 0;
      grid[gy * COLS + gx] = b;
    }
  }
  for (let gx = 0; gx < COLS; gx += 2) grid[HORIZON * COLS + gx] = 0.5; // the lake's edge, dotted
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
        if (hash(gx2, gy + Math.floor(t * 5 + hash(gx2, gy) * 5) * 131) > 0.8) b = Math.max(b, (0.4 + 0.5 * exposure) * (1 - dx / hw) * (1 - depth / 60));
      }
      grid[gy * COLS + gx] = b;
    }
  }
  drawMarks(ctx, grid, COLS, ROWS, CELL);
}

let raf = 0, target = 0, shown = 0;

/** Load progress, 0..1; the exposure builds and the lodestar climbs toward it. */
export function setProgress(p) {
  target = Math.max(target, Math.max(0, Math.min(1, p || 0)));
}

/** One ring of light from the lodestar, as a load begins. */
export function wake() {
  wakeRequest = true;
}

/** Finish the rise and play the ready pulse; resolves when it has played. */
export function finishScene() {
  if (!raf) return Promise.resolve();
  target = 1;
  pulseAt = null; // a wake ring still in flight gives way to the ready one
  return new Promise((done) => {
    pulseRequest = done;
  });
}
let pulseRequest = null;

export function startScene(canvas) {
  if (raf) return;
  target = shown = 0;
  pulseAt = null;
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
    if (wakeRequest) ((pulseAt = t), (wakeRequest = false));
    shown += (target - shown) * Math.min(1, dt * (pulseRequest ? 5 : 1.8));
    const e = 1 - (1 - shown) ** 2;
    STAR.y = STAR_LOW + (STAR_HIGH - STAR_LOW) * e;
    if (pulseRequest && pulseAt === null && shown > 0.97) pulseAt = t;
    if (pulseAt !== null && t - pulseAt > 1.1 && pulseRequest) {
      const done = pulseRequest;
      pulseRequest = null;
      done();
    }
    draw(ctx, t, e);
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
