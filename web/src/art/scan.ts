import { drawMarks } from "./glyphs";
import type { Put } from "./glyphs";

// "Reading every neuron" scene in the same ASCII-mark style as the loading scene: a small network of layers,
// a scan line sweeping through it, neurons lighting up by how strongly they fire, and signals passing
// along the strongest connections to the next layer just as the scan reaches it.
const COLS = 110, ROWS = 86, CELL = 4.2;
const LAYER_SIZES = [5, 7, 8, 8, 8, 7, 4];
const PERIOD = 3.4; // seconds per sweep

const hash = (...v: number[]) => { let h = 0; for (const x of v) h = Math.sin(h * 1.7 + x * 12.9898 + 78.233) * 43758.5453; return h - Math.floor(h); };

// layers as columns of neurons
const LAYERS = LAYER_SIZES.map((n, l) => {
  const x = Math.round(8 + (l * (COLS - 16)) / (LAYER_SIZES.length - 1));
  const gap = (ROWS - 16) / (n - 1);
  return Array.from({ length: n }, (_, i) => ({ x, y: Math.round(8 + i * gap) }));
});

// connections between adjacent layers; only the nearer ones are drawn at rest so the layers stay readable
type Edge = { l: number; i: number; j: number; cells: [number, number][]; near: boolean };
const EDGES: Edge[] = [];
const EDGE_AT: Edge[][][] = LAYERS.map((layer) => layer.map(() => [])); // [layer][from][to]
for (let l = 0; l < LAYERS.length - 1; l++)
  LAYERS[l].forEach((a, i) => LAYERS[l + 1].forEach((b, j) => {
    const steps = Math.max(Math.abs(b.x - a.x), Math.abs(b.y - a.y));
    const cells: [number, number][] = [];
    for (let s = 4; s < steps - 3; s++) cells.push([Math.round(a.x + ((b.x - a.x) * s) / steps), Math.round(a.y + ((b.y - a.y) * s) / steps)]);
    const near = Math.abs(i / (LAYERS[l].length - 1) - j / (LAYERS[l + 1].length - 1)) < 0.2;
    const e = { l, i, j, cells, near };
    EDGES.push(e);
    EDGE_AT[l][i][j] = e;
  }));

const headX = (k: number) => -6 + k * (COLS + 12); // scan line position for sweep phase k in 0..1
const litAt = (x: number) => (x + 6) / (COLS + 12);   // phase at which the scan reaches column x

function draw(ctx: CanvasRenderingContext2D, t: number) {
  const grid = new Float32Array(COLS * ROWS);
  const put: Put = (x, y, b) => {
    if (x < 0 || y < 0 || x >= COLS || y >= ROWS) return;
    const i = y * COLS + x;
    if (b > grid[i]) grid[i] = b;
  };
  const sweep = Math.floor(t / PERIOD), k = (t / PERIOD) % 1;
  const act = (l: number, i: number) => Math.pow(hash(sweep, l, i), 1.6); // how strongly each neuron fires this sweep

  for (const e of EDGES) if (e.near) e.cells.forEach(([x, y], s) => s % 4 === 0 && put(x, y, 0.16));

  // signals: each strongly firing neuron sends a pulse to its two strongest targets, arriving with the scan
  for (let l = 0; l < LAYERS.length - 1; l++) {
    const k0 = litAt(LAYERS[l][0].x), k1 = litAt(LAYERS[l + 1][0].x);
    if (k < k0 || k > k1 + 0.08) continue;
    const f = Math.min(1, (k - k0) / (k1 - k0));
    LAYERS[l].forEach((_, i) => {
      if (act(l, i) < 0.45) return;
      let first = 0, second = -1, h1 = -1, h2 = -1;
      for (let j = 0; j < LAYERS[l + 1].length; j++) {
        const h = hash(sweep, l, i, j);
        if (h > h1) [second, h2, first, h1] = [first, h1, j, h];
        else if (h > h2) [second, h2] = [j, h];
      }
      for (const j of [first, second]) {
        const e = EDGE_AT[l][i][j];
        if (!e.cells.length) continue;
        e.cells.forEach(([x, y], s) => s % 2 === 0 && put(x, y, 0.32 * (1 - Math.max(0, k - k1) / 0.08)));
        if (f < 1) {
          const p = Math.floor(f * (e.cells.length - 1));
          for (let tr = 0; tr < 4; tr++) {
            const c = e.cells[Math.max(0, p - tr * 2)];
            put(c[0], c[1], 1 - tr * 0.2);
          }
        }
      }
    });
  }

  // neurons: dim until the scan reaches them, then flare by activation and glow down slowly
  LAYERS.forEach((layer, l) => layer.forEach((n, i) => {
    const since = (k - litAt(n.x)) * PERIOD;
    const a = act(l, i);
    const b = since < 0 ? 0.35 : 0.35 + 0.65 * a * Math.exp(-since / 2.2) + (since < 0.25 ? 0.3 : 0);
    const on = a > 0.45 && since >= 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) put(n.x + dx, n.y + dy, Math.min(1, b) * (dx || dy ? (on ? 0.85 : 0.5) : 1));
    if (a > 0.7 && since >= 0 && since < 1.4) // a halo on the strongest, right after they fire
      for (const [dx, dy] of [[2, 0], [-2, 0], [0, 2], [0, -2], [2, 1], [2, -1], [-2, 1], [-2, -1], [1, 2], [-1, 2], [1, -2], [-1, -2]])
        put(n.x + dx, n.y + dy, 0.45 * (1 - since / 1.4));
  }));

  // the scan line, with corner brackets around the layer it is reading
  const hx = Math.round(headX(k));
  for (let y = 2; y < ROWS - 2; y += 2) put(hx, y, 0.28);
  const cur = LAYERS.reduce((best, layer) => (Math.abs(layer[0].x - hx) < Math.abs(best[0].x - hx) ? layer : best));
  if (Math.abs(cur[0].x - hx) < 6) {
    const x0 = cur[0].x - 5, x1 = cur[0].x + 5, y0 = cur[0].y - 5, y1 = cur[cur.length - 1].y + 5;
    for (let d = 0; d < 3; d++)
      for (const [x, y] of [[x0 + d, y0], [x0, y0 + d], [x1 - d, y0], [x1, y0 + d], [x0 + d, y1], [x0, y1 - d], [x1 - d, y1], [x1, y1 - d]]) put(x, y, 0.7);
  }

  drawMarks(ctx, grid, COLS, ROWS, CELL);
}

/** Runs until the canvas leaves the page. */
export function startScan(canvas: HTMLCanvasElement) {
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
    if (!canvas.isConnected) return;
    requestAnimationFrame(tick);
    if (now - last < 66) return; // ~15 fps
    last = now;
    draw(ctx, (now - t0) / 1000 + 0.8);
  };
  if (still) draw(ctx, 1.6);
  else requestAnimationFrame(tick);
}
