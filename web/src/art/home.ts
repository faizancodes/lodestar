import { drawMarks } from "./glyphs";
import type { Put } from "./glyphs";

// The small scenes on the home screen's cards, one per page, in the same ASCII marks as everything else. Each is a
// still until its card is hovered, then plays (~20 fps) until the pointer leaves.
//   abliterate  the refusal direction as a bright arrow with a halo; a clean cut partway along, and past it the
//               arrow crumbles into dust drifting the way it pointed
//   steer       a field of level flow lines; a glowing push bends the ones near it upward like a lens, and the
//               flow streams along them
//   chat        two bubbles drawn in dots: theirs dim, the model's bright, its reply typing out behind a caret
const CELL = 3.2;
const hash = (...v: number[]) => { let h = 0; for (const x of v) h = Math.sin(h * 1.7 + x * 12.9898 + 78.233) * 43758.5453; return h - Math.floor(h); };

export type HomeArt = "abliterate" | "steer" | "chat";

const ART: Record<HomeArt, (put: Put, cols: number, rows: number, t: number) => void> = {
  abliterate(put, cols, rows, t) {
    const x0 = cols * 0.08, y0 = rows * 0.82, x1 = cols * 0.9, y1 = rows * 0.22;
    const L = Math.hypot(x1 - x0, y1 - y0), dx = (x1 - x0) / L, dy = (y1 - y0) / L;
    const cut = L * 0.52;
    for (let y = 0; y < rows; y++)
      for (let x = 0; x < cols; x++) {
        const along = (x - x0) * dx + (y - y0) * dy, across = Math.abs((x - x0) * dy - (y - y0) * dx);
        if (along < 0 || along > L + 4) continue;
        if (along < cut - 1.5) { // the solid part: a bright core in a soft halo, brightening toward the cut
          const lift = 0.55 + 0.45 * (along / cut);
          if (across < 1.6) put(x, y, lift);
          else if (across < 7) put(x, y, 0.45 * Math.exp(-(across - 1.6) / 1.8) * lift);
        } else if (along > cut + 1.5) { // past the cut: dust, thinning and drifting along the old direction
          const k = (along - cut) / (L - cut + 4);
          const drift = Math.floor(along - t * 6);
          const head = along > L - 9 && across < (L + 1 - along) * 0.8; // the arrowhead it was, in dust
          if ((across < 3.5 || head) && hash(x, y, drift) > 0.58 + k * 0.3) put(x, y, (0.72 - k * 0.45) * (head ? 0.9 : 1 - across / 4));
        }
      }
    for (let s = -5; s <= 5; s++) { // the cut: a short bright slice across the arrow
      const cx = x0 + dx * cut + dy * s, cy = y0 + dy * cut - dx * s;
      put(cx, cy, 0.95 - Math.abs(s) * 0.09);
    }
  },
  steer(put, cols, rows, t) {
    const px = cols * 0.42, py = rows * 0.62; // where the push lands
    for (let i = 0; i < 9; i++) {
      const base = rows * (0.1 + i * 0.1), near = Math.exp(-((base - py) ** 2) / 60);
      for (let x = 0; x < cols; x++) {
        const bend = near * rows * 0.34 * (1 / (1 + Math.exp(-(x - px) / 5)));
        const y = base - bend, flow = 0.5 + 0.5 * Math.sin(x * 0.45 - t * 5 + i);
        put(x, y, (0.14 + near * 0.6) * (0.55 + 0.45 * flow));
      }
    }
    for (let y = 0; y < rows; y++) // the push: a glowing node with a soft ring
      for (let x = Math.floor(px - 8); x < px + 8; x++) {
        const r = Math.hypot(x - px, y - py);
        if (r < 1.3) put(x, y, 1);
        else if (r < 5) put(x, y, 0.45 * Math.exp(-(r - 1.3) / 1.2) + (Math.abs(r - 3.6 - Math.sin(t * 3) * 0.4) < 0.5 ? 0.3 : 0));
      }
  },
  chat(put, cols, rows, t) {
    const bubble = (x0: number, y0: number, x1: number, y1: number, b: number) => { // a rounded outline in dots
      for (let x = x0 + 2; x <= x1 - 2; x += 2) (put(x, y0, b), put(x, y1, b));
      for (let y = y0 + 2; y <= y1 - 2; y += 2) (put(x0, y, b), put(x1, y, b));
      put(x0 + 1, y0 + 1, b); put(x1 - 1, y0 + 1, b); put(x0 + 1, y1 - 1, b); put(x1 - 1, y1 - 1, b);
    };
    const words = (x0: number, y: number, x1: number, b: number, seed: number, upto = Infinity) => { // a line of text as runs of marks
      let x = x0, n = 0;
      while (x < x1 && n < upto) {
        const w = 2 + Math.floor(hash(seed, x) * 5);
        for (let k = 0; k < w && x + k < x1 && n < upto; k++, n++) put(x + k, y, b);
        x += w + 1;
      }
      return x;
    };
    const a = { x0: 4, y0: 3, x1: Math.round(cols * 0.56), y1: 12 };
    bubble(a.x0, a.y0, a.x1, a.y1, 0.28);
    words(a.x0 + 3, 6, a.x1 - 3, 0.3, 1);
    words(a.x0 + 3, 9, a.x1 - 10, 0.3, 2);
    const m = { x0: Math.round(cols * 0.34), y0: 16, x1: cols - 4, y1: rows - 3 };
    bubble(m.x0, m.y0, m.x1, m.y1, 0.5);
    const lines: [number, number][] = [[19, 1], [22, 1], [25, 0.62]], width = m.x1 - m.x0 - 6;
    const typed = t ? (t * 26) % (width * 2.6 + 20) : Infinity; // characters out so far; the full reply when still
    let left = typed, caret = null as [number, number] | null;
    lines.forEach(([y, frac], i) => {
      if (y > m.y1 - 2 || left <= 0) return;
      const end = words(m.x0 + 3, y, m.x0 + 3 + Math.round(width * frac), 0.78, 10 + i, left);
      left -= width * frac;
      caret = [Math.min(end, m.x0 + 3 + Math.round(width * frac)), y];
    });
    if (caret && Math.floor((t || 0) * 2.5) % 2 === 0) put(caret[0] + 1, caret[1], 1);
  },
};

function render(canvas: HTMLCanvasElement, kind: HomeArt, t: number) {
  const w = canvas.clientWidth, h = canvas.clientHeight, dpr = devicePixelRatio || 1;
  if (!w || !h || !ART[kind]) return;
  if (canvas.width !== Math.round(w * dpr)) ((canvas.width = Math.round(w * dpr)), (canvas.height = Math.round(h * dpr)));
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
  ART[kind](put, cols, rows, t);
  drawMarks(ctx, grid, cols, rows, CELL);
}

/** Draws a card's scene as a still, and plays it while `card` is hovered or focused. */
export function drawHomeArt(canvas: HTMLCanvasElement, kind: HomeArt, card?: HTMLElement | null) {
  render(canvas, kind, 0);
  if (!card || card.dataset.artBound) return;
  card.dataset.artBound = "1";
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let raf = 0, t0 = 0, last = 0;
  const tick = (now: number) => {
    raf = requestAnimationFrame(tick);
    if (now - last < 50) return;
    last = now;
    render(canvas, kind, (now - t0) / 1000 + 0.01);
  };
  const start = () => !still && !raf && ((t0 = performance.now()), (raf = requestAnimationFrame(tick)));
  const stop = () => (cancelAnimationFrame(raf), (raf = 0), render(canvas, kind, 0));
  card.addEventListener("mouseenter", start);
  card.addEventListener("focus", start);
  card.addEventListener("mouseleave", stop);
  card.addEventListener("blur", stop);
}
