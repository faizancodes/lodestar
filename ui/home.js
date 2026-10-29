import { drawMarks } from "./glyphs.js";

// The small scenes on the home screen's cards, one per page, in the same ASCII marks as everything else:
//   abliterate  a field of strokes all leaning one way, until a cut: past it they lie flat, the direction gone
//   steer       a bundle of paths running level, one bent upward by a push and ending in an arrowhead
//   chat        a conversation as lines of marks, theirs dim on the left, the model's bright on the right
const CELL = 4;

const ART = {
  abliterate(put, cols, rows) {
    const cut = Math.round(cols * 0.55);
    for (let y = 2; y < rows - 2; y += 4)
      for (let x = 3; x < cols - 3; x += 5) {
        if (Math.abs(x - cut) < 3) continue;
        const before = x < cut, b = before ? 0.5 : 0.75;
        for (let k = -1; k <= 1; k++) put(x + k, before ? y - k : y, b); // leaning strokes before the cut, level after
      }
    for (let y = 0; y < rows; y++) if (y % 2 === 0) put(cut, y, 0.9); // the cut
  },
  steer(put, cols, rows) {
    const mid = Math.round(rows / 2);
    for (const dy of [-8, -4, 4, 8]) for (let x = 2; x < cols - 2; x += 2) put(x, mid + dy, 0.28);
    const lift = (x) => (x < cols * 0.35 ? 0 : -(((x - cols * 0.35) / (cols * 0.6)) ** 2) * (rows * 0.42));
    let last = [0, mid];
    for (let x = 2; x < cols - 4; x++) {
      const y = Math.round(mid + lift(x));
      put(x, y, x < cols * 0.35 ? 0.5 : 0.9);
      last = [x, y];
    }
    const [hx, hy] = last; // arrowhead along the path's final heading
    for (let k = 1; k <= 3; k++) (put(hx - k, hy + k - 1, 0.8), put(hx - k - 1, hy - 1 + k * 0.2, 0.6));
    put(Math.round(cols * 0.35), mid - 2, 0.85); // where the push lands
    put(Math.round(cols * 0.35), mid - 3, 0.6);
  },
  chat(put, cols, rows) {
    const lines = [[3, 0.32, 0.4], [3, 0.32, 0.55], [8, 0.7, 0.9], [8, 0.7, 0.7], [8, 0.7, 0.8], [3, 0.32, 0.35]];
    lines.forEach(([x0, b, frac], i) => {
      const y = 3 + i * 4 + (i > 1 ? 1 : 0) + (i > 4 ? 1 : 0), x1 = Math.round(x0 + (cols - 12) * frac * (x0 > 3 ? 0.9 : 0.6));
      let x = x0;
      while (x < x1) { // words: runs of marks with gaps
        const w = 2 + ((x * 7 + i * 13) % 5);
        for (let k = 0; k < w && x + k < x1; k++) put(x + k, y, b);
        x += w + 1;
      }
      if (i === 4) put(x1 + 1, y, 1); // the caret, where the model is still writing
    });
  },
};

/** Draws a card's scene once into its canvas, at whatever size CSS gave it. */
export function drawHomeArt(canvas, kind) {
  const w = canvas.clientWidth, h = canvas.clientHeight, dpr = devicePixelRatio || 1;
  if (!w || !h || !ART[kind]) return;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = getComputedStyle(canvas).color;
  const cols = Math.floor(w / CELL), rows = Math.floor(h / CELL), grid = new Float32Array(cols * rows);
  const put = (x, y, b) => {
    x = Math.round(x);
    y = Math.round(y);
    if (x < 0 || y < 0 || x >= cols || y >= rows) return;
    const i = y * cols + x;
    if (b > grid[i]) grid[i] = b;
  };
  ART[kind](put, cols, rows);
  drawMarks(ctx, grid, cols, rows, CELL);
}
