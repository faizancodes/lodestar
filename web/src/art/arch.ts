import { drawMarks } from "./glyphs";
import type { Put } from "./glyphs";
import type { Arch } from "../lib/types";

// The home screen's diagram of the loaded model, left to right in the same ASCII marks as the cards above it:
//   the vocabulary, a tall column, fanning into the residual stream; the stream running level through every layer;
//   each layer's attention above it (a causal triangle where it attends over every earlier token, only the diagonal
//   where it keeps a running state instead) and its MLP below (a bulge as wide as the neurons outnumber the
//   stream's width, dotted with neurons); then the stream fanning back out to the vocabulary.
// When there isn't room for every layer, the first and last few are drawn with a dotted gap for the rest.
// A still until hovered; then signals run along the stream, lighting each layer as they pass and firing a few of
// its neurons.
const CELL = 3.2;
const hash = (...v: number[]) => { let h = 0; for (const x of v) h = Math.sin(h * 1.7 + x * 12.9898 + 78.233) * 43758.5453; return h - Math.floor(h); };

export type ArchShape = { layers: number; hidden: number; neurons: number; arch: Arch };

function scene(put: Put, cols: number, rows: number, t: number, m: ArchShape) {
  const L = m.layers, sy = Math.round(rows * 0.5);
  const ew = Math.max(12, Math.round(cols * 0.075)); // the embedding and unembedding fans
  const x0 = ew + 3, x1 = cols - ew - 3, MIN = 6; // the narrowest a layer can be drawn
  const fit = Math.max(4, Math.floor((x1 - x0) / MIN)), gap = L > fit ? 2 : 0;
  const shown = L > fit ? [...Array(fit - gap).keys()].map((i) => (i < (fit - gap) / 2 ? i : L - (fit - gap) + i)) : [...Array(L).keys()];
  const lw = (x1 - x0) / (shown.length + gap);
  const top = Math.round(rows * 0.08), bot = Math.round(rows * 0.92);

  // signals along the stream while playing: where each one is, from 0 (embedding) to 1 (unembedding)
  const pulses = t ? [0, 0.34, 0.68].map((k) => ((t * 0.16 + k) % 1)) : [];
  const near = (x: number) => pulses.reduce((b, p) => Math.max(b, Math.exp(-(((x - (2 + p * (cols - 4))) / 5) ** 2))), 0);

  // the vocabulary on each side: a tall column of dots, and rays converging on the stream
  for (const side of [0, 1]) {
    const vx = side ? cols - 2 : 1, ex = side ? x1 + 1 : x0 - 1;
    for (let y = top; y <= bot; y++) put(vx, y, 0.3 + 0.25 * hash(y, side));
    for (let r = 0; r < 11; r++) {
      const fy = top + ((bot - top) * r) / 10;
      for (let u = 0; u <= 1; u += 0.6 / ew) {
        const x = vx + (ex - vx) * u, y = fy + (sy - fy) * u * u;
        put(x, y, 0.2 + 0.4 * u ** 2 + 0.5 * near(x));
      }
    }
  }

  // the residual stream
  for (let x = x0 - 1; x <= x1 + 1; x++) {
    const flow = t ? 0.06 * Math.sin(x * 0.5 - t * 6) : 0;
    put(x, sy, 0.62 + flow + 0.4 * near(x));
  }

  const ratio = Math.max(1, m.neurons / m.hidden);
  if (gap) { // the layers left out: a dotted run on the stream, between the first few and the last few
    const gx = x0 + Math.floor(shown.length / 2) * lw;
    for (let x = Math.round(gx + 1); x < gx + gap * lw - 1; x += 2) (put(x, sy - 6, 0.3), put(x, sy + 6, 0.3));
  }

  shown.forEach((l, i) => {
    const lx = x0 + (i + (gap && i >= shown.length / 2 ? gap : 0)) * lw, cx = lx + lw / 2, lit = near(cx), full = m.arch.attention[l] !== "linear";

    // attention, above the stream: rows are positions reading, columns the positions read
    const aw = Math.max(3, lw - 2), ay0 = top + 2, ay1 = sy - 4, ah = ay1 - ay0;
    for (let y = ay0; y <= ay1; y++)
      for (let x = Math.ceil(lx + 1); x <= lx + aw; x++) {
        const u = (x - lx - 1) / aw, v = (y - ay0) / ah;
        if (u > v + 0.02) continue; // a token never reads ahead of itself
        const diag = v - u < 0.1, left = u < 0.08;
        const b = full ? (diag ? 0.62 : left ? 0.4 : 0.14 + 0.12 * v) : diag ? 0.55 : left ? 0.12 : 0;
        if (b) put(x, y, b + lit * 0.32);
      }

    // the MLP, below: from the stream's width out to its neurons and back
    const my0 = sy + 4, my1 = bot - 1, mh = my1 - my0, maxW = Math.max(1.5, lw / 2 - 0.6), tip = maxW / ratio;
    const beat = Math.floor(t * 9);
    for (let y = my0; y <= my1; y++) {
      const d = Math.abs((y - my0) / mh - 0.5) * 2, hw = tip + (maxW - tip) * (1 - d);
      for (let x = Math.ceil(cx - hw); x <= cx + hw; x++) {
        const edge = Math.abs(x - cx) > hw - 1;
        const fire = lit > 0.3 && hash(l, x, y, beat) > 0.93 ? 0.9 : 0;
        put(x, y, Math.max(edge ? 0.3 + lit * 0.25 : hash(x, y) > 0.55 ? 0.2 + lit * 0.2 : 0.06, fire));
      }
    }

    // where each block reads from and writes back to the stream
    for (let y = ay1 + 1; y < sy; y++) put(cx, y, 0.28 + lit * 0.4);
    for (let y = sy + 1; y < my0; y++) put(cx, y, 0.28 + lit * 0.4);
  });
}

function render(canvas: HTMLCanvasElement, m: ArchShape, t: number) {
  const w = canvas.clientWidth, h = canvas.clientHeight, dpr = devicePixelRatio || 1;
  if (!w || !h) return;

  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) ((canvas.width = Math.round(w * dpr)), (canvas.height = Math.round(h * dpr)));
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = getComputedStyle(canvas).color;

  const cols = Math.floor(w / CELL), rows = Math.floor(h / CELL), grid = new Float32Array(cols * rows);
  const put: Put = (x, y, b) => {
    x = Math.round(x);
    y = Math.round(y);
    if (x < 0 || y < 0 || x >= cols || y >= rows) return;
    const i = y * cols + x;
    if (b > grid[i]) grid[i] = b;
  };

  scene(put, cols, rows, t, m);
  drawMarks(ctx, grid, cols, rows, CELL);
}

/** Draws the model as a still, plays it while `hover` is hovered, and redraws when the canvas changes size.
 *  Returns a function that stops all of that. */
export function startArch(canvas: HTMLCanvasElement, hover: HTMLElement, m: ArchShape) {
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let raf = 0, t0 = 0, last = 0;

  const tick = (now: number) => {
    raf = requestAnimationFrame(tick);
    if (now - last < 50) return;
    last = now;
    render(canvas, m, (now - t0) / 1000 + 0.01);
  };

  const start = () => !still && !raf && ((t0 = performance.now()), (raf = requestAnimationFrame(tick)));
  const stop = () => (cancelAnimationFrame(raf), (raf = 0), render(canvas, m, 0));
  const ro = new ResizeObserver(() => !raf && render(canvas, m, 0));

  render(canvas, m, 0);
  ro.observe(canvas);
  hover.addEventListener("mouseenter", start);
  hover.addEventListener("mouseleave", stop);

  return () => {
    cancelAnimationFrame(raf);
    ro.disconnect();
    hover.removeEventListener("mouseenter", start);
    hover.removeEventListener("mouseleave", stop);
  };
}
