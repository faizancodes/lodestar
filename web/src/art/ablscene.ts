import { drawMarks } from "./glyphs";
import type { Put } from "./glyphs";

// Abliterate's loading scenes, in the same ASCII-mark style as the scan: one small scene per analysis stage.
//   0 read     prompts slide through the model's layers and leave their last word's state in a tally
//   1 compare  two clouds (bad requests, everyday ones) and the arrow between their means: the refusal direction
//   2 trial    candidate layers are tried in turn; each sends four requests that come back answered or refused
//   3 check    replies are written out line by line, then judged
const COLS = 150, ROWS = 40, CELL = 4.2;

const hash = (...v: number[]) => { let h = 0; for (const x of v) h = Math.sin(h * 1.7 + x * 12.9898 + 78.233) * 43758.5453; return h - Math.floor(h); };

function line(put: Put, x0: number, y0: number, x1: number, y1: number, b: number, every = 1, upto = 1) {
  const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
  for (let s = 0; s <= steps * upto; s++)
    if (s % every === 0) put(Math.round(x0 + ((x1 - x0) * s) / steps), Math.round(y0 + ((y1 - y0) * s) / steps), b);
}

const block = (put: Put, x: number, y: number, r: number, b: number) => { for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) put(x + dx, y + dy, b); };
const cross = (put: Put, x: number, y: number, b: number) => { for (let d = -2; d <= 2; d++) (put(x + d, y + d, b), put(x + d, y - d, b)); };
const tick = (put: Put, x: number, y: number, b: number) => { for (let d = 0; d <= 2; d++) put(x - 3 + d, y + d - 1, b); for (let d = 0; d <= 4; d++) put(x - 1 + d, y + 1 - d, b); };

function brackets(put: Put, x0: number, y0: number, x1: number, y1: number, b: number) {
  for (let d = 0; d < 3; d++)
    for (const [x, y] of [[x0 + d, y0], [x0, y0 + d], [x1 - d, y0], [x1, y0 + d], [x0 + d, y1], [x0, y1 - d], [x1 - d, y1], [x1, y1 - d]]) put(x, y, b);
}

// ---------------------------------------------------------------- 0 read
const LAYER_X = Array.from({ length: 9 }, (_, i) => 62 + i * 6);
const LANES = [8, 16, 24, 32];

function read(put: Put, t: number) {
  for (const x of LAYER_X) for (let y = 3; y < ROWS - 3; y += 2) put(x, y, 0.16);

  const end = LAYER_X[LAYER_X.length - 1] + 4;
  LANES.forEach((y, lane) => {
    const period = 3.2, off = lane * 0.83;
    const n = Math.floor((t + off) / period), k = ((t + off) / period) % 1;
    const words = Array.from({ length: 5 }, (_, w) => 2 + Math.floor(hash(n, lane, w) * 5));
    const len = words.reduce((a, w) => a + w + 2, 0);
    const head = Math.round(-len + k * (end + len + 2)); // the sentence's right edge

    let x = head;
    for (let w = words.length - 1; w >= 0; w--) { // right to left: the last word leads
      for (let c = 0; c < words[w]; c++) {
        const cx = x - c;
        if (cx < end) put(cx, y, w === words.length - 1 ? 1 : 0.55);
      }
      x -= words[w] + 2;
    }

    // layers flare where the sentence passes through them
    for (const lx of LAYER_X) if (lx <= head && lx > head - len) for (let d = -2; d <= 2; d++) put(lx, y + d, 0.75 - Math.abs(d) * 0.18);

    // the last word's state lands in this lane's tally, which grows over the run
    const fill = Math.min(26, 6 + Math.floor((t + off) / period) * 3 + (head >= end ? 3 : 0));
    for (let c = 0; c < fill; c++) for (let d = -1; d <= 1; d++) put(end + 6 + c, y + d, c >= fill - 3 && head >= end && k < 0.92 ? 1 : 0.42);
  });
}

// ---------------------------------------------------------------- 1 compare
const CLOUD = (seed: number, cx: number, cy: number) => Array.from({ length: 26 }, (_, i) => {
  const a = hash(seed, i) * Math.PI * 2, r = Math.sqrt(hash(seed, i, 1)) * 11;
  return { x: cx + Math.cos(a) * r * 1.6, y: cy + Math.sin(a) * r * 0.75, p: hash(seed, i, 2) * 6.28 };
});
const BAD = CLOUD(3, 104, 13), OK = CLOUD(7, 44, 27);
const mean = (pts: { x: number; y: number }[]) => [pts.reduce((a, p) => a + p.x, 0) / pts.length, pts.reduce((a, p) => a + p.y, 0) / pts.length];
const [ax, ay] = mean(OK), [bx, by] = mean(BAD);

function compare(put: Put, t: number) {
  const period = 4.2, k = (t / period) % 1;

  for (const p of OK) block(put, Math.round(p.x + Math.sin(t * 0.9 + p.p) * 1.2), Math.round(p.y + Math.cos(t * 0.7 + p.p) * 0.8), 0, 0.5);
  for (const p of BAD) {
    const x = Math.round(p.x + Math.sin(t * 0.9 + p.p) * 1.2), y = Math.round(p.y + Math.cos(t * 0.7 + p.p) * 0.8);
    put(x, y, 1); put(x + 1, y, 0.6);
  }

  brackets(put, Math.round(ax) - 3, Math.round(ay) - 3, Math.round(ax) + 3, Math.round(ay) + 3, 0.7);
  brackets(put, Math.round(bx) - 3, Math.round(by) - 3, Math.round(bx) + 3, Math.round(by) + 3, 0.7);

  // the difference of means, drawn from the everyday mean to the bad one
  const grow = Math.min(1, Math.max(0, (k - 0.15) / 0.5));
  line(put, ax, ay, bx, by, 0.95, 1, grow);
  if (grow >= 1) {
    const ang = Math.atan2(by - ay, bx - ax);
    for (const s of [-0.5, 0.5]) line(put, bx, by, bx - Math.cos(ang + s) * 7, by - Math.sin(ang + s) * 7, 1); // arrowhead

    const pulse = (k - 0.65) / 0.35; // a bright dot runs along it once it's drawn
    const px = ax + (bx - ax) * pulse, py = ay + (by - ay) * pulse;
    block(put, Math.round(px), Math.round(py), 1, 1);
  }
}

// ---------------------------------------------------------------- 2 trial
const TL = Array.from({ length: 16 }, (_, i) => 12 + i * 8);
const CANDS = [7, 9, 11, 12, 13, 14], IS_CAND = new Set(CANDS);

function trial(put: Put, t: number) {
  const per = 1.15, cycle = per * CANDS.length + 1.6;
  const n = Math.floor(t / cycle), u = t % cycle;
  const refusedOf = CANDS.map((_, c) => Array.from({ length: 4 }, (_, q) => hash(n, c, q) < 0.15 + 0.6 * hash(n, c)));
  const best = refusedOf.map((r) => r.filter(Boolean).length).reduce((bi, v, i, a) => (v < a[bi] ? i : bi), 0);

  TL.forEach((x, i) => {
    const cand = IS_CAND.has(i);
    for (let y = 3; y <= 14; y += cand ? 1 : 2) put(x, y, cand ? 0.5 : 0.18);
  });

  CANDS.forEach((li, c) => {
    const x = TL[li], since = u - c * per;
    if (since < 0) return;

    const active = since < per;
    if (active) brackets(put, x - 3, 1, x + 3, 16, 0.85);
    for (let y = 3; y <= 14; y++) put(x, y, active ? 0.95 : 0.6);

    refusedOf[c].forEach((refused, q) => {
      const land = 0.15 + q * 0.17; // each request drops into the tray in turn
      const f = Math.min(1, (since - land) / 0.25);
      if (f <= 0) return;

      const slot = 20 + q * 5;
      if (f < 1) put(x, Math.round(16 + f * (slot - 16)), 1);
      else if (refused) cross(put, x, slot, 0.55);
      else block(put, x, slot, 1, 0.9);
    });
  });

  if (u > per * CANDS.length) { // the layer that refused least is kept
    const x = TL[CANDS[best]];
    const blink = Math.floor((u - per * CANDS.length) / 0.2) % 2 ? 1 : 0.6;
    brackets(put, x - 4, 0, x + 4, ROWS - 2, blink);
  }
}

// ---------------------------------------------------------------- 3 check
function check(put: Put, t: number) {
  const cycle = 5.2, n = Math.floor(t / cycle), u = t % cycle;

  [0, 1].forEach((col) => {
    const x0 = 10 + col * 72, lines = 5, start = col * 0.5;
    let chars = Math.max(0, (u - start) * 34);

    for (let l = 0; l < lines; l++) {
      const y = 6 + l * 5, w = l === lines - 1 ? 18 + Math.floor(hash(n, col, l) * 14) : 40 + Math.floor(hash(n, col, l) * 12);
      const shown = Math.min(w, chars);

      let x = 0;
      while (x < shown) {
        const word = 2 + Math.floor(hash(n, col, l, x) * 5);
        for (let c = 0; c < word && x + c < shown; c++) put(x0 + x + c, y, 0.6);
        x += word + 1;
      }

      if (chars > 0 && chars < w && Math.floor(t * 4) % 2) block(put, x0 + Math.round(shown) + 1, y, 0, 1); // caret
      chars -= w;
      if (chars < 0) break;
    }

    if (chars > 6) { // finished: a judge's mark beside it
      const mx = x0 + 60, my = 14, refused = hash(n, col) < 0.5;
      brackets(put, mx - 5, my - 5, mx + 5, my + 5, 0.55);
      (refused ? cross : tick)(put, mx, my, 1);
    }
  });
}

const SCENES = [read, compare, trial, check];

/** Starts drawing scene 0; call .set(i) to switch scenes. Runs until the canvas leaves the page. */
export function startAblScene(canvas: HTMLCanvasElement) {
  const dpr = devicePixelRatio || 1;
  canvas.width = COLS * CELL * dpr;
  canvas.height = ROWS * CELL * dpr;
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let scene = 0, t0 = performance.now(), last = 0;

  const draw = (t: number) => {
    const grid = new Float32Array(COLS * ROWS);
    const put: Put = (x, y, b) => {
      if (x < 0 || y < 0 || x >= COLS || y >= ROWS) return;
      const i = y * COLS + x;
      if (b > grid[i]) grid[i] = b;
    };

    SCENES[scene](put, t);
    ctx.fillStyle = getComputedStyle(canvas).color;
    drawMarks(ctx, grid, COLS, ROWS, CELL);
  };

  const tickFn = (now: number): void => {
    if (!canvas.isConnected || canvas.closest("[hidden]")) return void (running = false);

    requestAnimationFrame(tickFn);
    if (now - last < 66) return; // ~15 fps
    last = now;
    draw((now - t0) / 1000 + 0.6);
  };

  let running = false;
  const run = () => {
    if (still) return draw(3);
    if (!running) ((running = true), requestAnimationFrame(tickFn));
  };

  run();
  return {
    set(i: number) {
      if (i === scene || !SCENES[i]) return run();

      scene = i;
      t0 = performance.now();
      canvas.classList.remove("swap");
      canvas.offsetWidth; // restart the swap animation
      canvas.classList.add("swap");
      run();
    },
  };
}
