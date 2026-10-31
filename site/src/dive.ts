import { drawMarks, type Put } from "../../web/src/art/glyphs";
import { compassStar } from "../../web/src/art/mark";

// The hero: an endless dive into the model, in the app's ASCII marks. Each level is a layer drawn as a frame
// holding a 5×5 field of neurons, wired column to column with signals running along the wires. The neuron at its
// centre is the next layer down, a little bigger than its neighbours, and the camera falls into it forever: every
// K× of zoom the picture repeats, with fresh activations, a quarter-turn's twist per level, and dust streaming
// past. The lodestar sits at the vanishing point.
const CELL = 4;         // css px per grid cell
const K = 4.6;          // how much bigger each level is than the one inside it
const PERIOD = 6.5;     // seconds to fall through one level
const TWIST = 0.36;     // radians each level turns against the one around it
const N = 5, GAP = 0.38, NODE = 0.075; // neurons per side, their spacing and half-size, as fractions of a level

const hash = (...v: number[]) => { let h = 0; for (const x of v) h = Math.sin(h * 1.7 + x * 12.9898 + 78.233) * 43758.5453; return h - Math.floor(h); };
const clamp = (x: number) => Math.max(0, Math.min(1, x));

/** depth: called with the absolute level the camera has reached, each time it passes one. */
export function startDive(canvas: HTMLCanvasElement, depth?: (level: number) => void) {
  let cols = 0, rows = 0, grid = new Float32Array(0), ctx!: CanvasRenderingContext2D;

  const size = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight, dpr = devicePixelRatio || 1;
    if (!w || !h) return false;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cols = Math.ceil(w / CELL);
    rows = Math.ceil(h / CELL);
    grid = new Float32Array(cols * rows);
    return true;
  };

  const put: Put = (x, y, b) => {
    x = Math.round(x);
    y = Math.round(y);
    if (x < 0 || y < 0 || x >= cols || y >= rows || b <= 0) return;
    const i = y * cols + x;
    if (b > grid[i]) grid[i] = b;
  };

  const line = (x0: number, y0: number, x1: number, y1: number, b: number, every = 1) => {
    const n = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
    for (let s = 0; s <= n; s += every) put(x0 + ((x1 - x0) * s) / n, y0 + ((y1 - y0) * s) / n, b);
  };

  // a square frame of half-size h turned by a about (cx, cy): a dotted outline with bright corner brackets
  const frame = (cx: number, cy: number, h: number, a: number, b: number, brackets: number) => {
    const c = Math.cos(a), s = Math.sin(a);
    const at = (u: number, v: number): [number, number] => [cx + u * c - v * s, cy + u * s + v * c];
    const corners: [number, number][] = [[-h, -h], [h, -h], [h, h], [-h, h]];
    for (let k = 0; k < 4; k++) {
      const [u0, v0] = corners[k], [u1, v1] = corners[(k + 1) % 4];
      const [x0, y0] = at(u0, v0), [x1, y1] = at(u1, v1);
      line(x0, y0, x1, y1, b, h > 6 ? 2 : 1);
      if (brackets > 0) { // the corners, solid for a short way along each side
        const arm = Math.min(0.22, 4 / Math.max(h, 1));
        const [ax, ay] = at(u0 + (u1 - u0) * arm, v0 + (v1 - v0) * arm);
        const [bx, by] = at(u1 + (u0 - u1) * arm, v1 + (v0 - v1) * arm);
        line(x0, y0, ax, ay, brackets);
        line(x1, y1, bx, by, brackets);
      }
    }
  };

  const draw = (t: number) => {
    grid.fill(0);
    const cx = cols / 2, cy = rows / 2, R = Math.hypot(cx, cy);
    const z = t / PERIOD;

    // dust streaming out past the camera, faster the further out it is
    for (let i = 0; i < 160; i++) {
      const a = hash(i, 1) * Math.PI * 2, f = (z * (0.8 + hash(i, 2) * 0.6) + hash(i, 3)) % 1;
      const r = 3 * K ** (f * 2.4), len = r * 0.07;
      const fade = clamp(f * 4) * clamp((R * 1.1 - r) / (R * 0.4)) * (0.25 + 0.5 * hash(i, 4));
      for (let u = 0; u <= len; u += 0.8) put(cx + Math.cos(a) * (r - u), cy + Math.sin(a) * (r - u), fade * (1 - u / (len + 1)));
    }

    // the levels, outermost first so the deeper ones draw over them
    const lo = Math.floor(z - Math.log(R * 3) / Math.log(K)), hi = Math.floor(z + 0.2);
    for (let L = lo; L <= hi; L++) {
      const h = K ** (z - L);                      // this level's half-size in cells
      const f = clamp((h - 1) / 5) * clamp(1 - (h - R * 0.9) / (R * 1.6)); // in from the deep, out past the edges
      if (f <= 0) continue;

      const a = L * TWIST, c = Math.cos(a), s = Math.sin(a);
      const at = (u: number, v: number): [number, number] => [cx + u * c - v * s, cy + u * s + v * c];
      frame(cx, cy, h, a, 0.32 * f, 0.75 * f);

      const nodes: [number, number, number][] = []; // x, y, activation
      for (let i = 0; i < N; i++)
        for (let j = 0; j < N; j++) {
          const [x, y] = at((i - 2) * GAP * h, (j - 2) * GAP * h);
          const act = 0.5 + 0.5 * Math.sin(t * (0.7 + hash(L, i, j) * 1.6) + hash(L, j, i) * 6.28);
          nodes.push([x, y, act * (0.35 + 0.65 * hash(L, i, j, 9))]);
        }

      // wires between neighbouring columns, with signals running along the busiest
      if (h > 9)
        for (let i = 0; i < N - 1; i++)
          for (let j = 0; j < N; j++)
            for (const dj of [-1, 0, 1]) {
              const j2 = j + dj;
              if (j2 < 0 || j2 >= N || hash(L, i, j, dj) < 0.45) continue;
              const [x0, y0, a0] = nodes[i * N + j], [x1, y1] = nodes[(i + 1) * N + j2];
              line(x0, y0, x1, y1, 0.12 * f, 2);
              if (a0 > 0.45) {
                const u = (t * 0.55 + hash(L, i, j, dj, 3)) % 1;
                put(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, (0.5 + a0 * 0.5) * f);
              }
            }

      // the neurons: a point when small, a frame lit by how hard it fires when near
      nodes.forEach(([x, y, act], k) => {
        if (k === (N * N - 1) / 2) return; // the centre is the next level down
        const r = NODE * h;
        if (r < 1.2) return put(x, y, (0.25 + act * 0.7) * f);
        frame(x, y, r, a, (0.2 + act * 0.5) * f, act > 0.6 ? 0.9 * f : 0);
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) put(x + dx, y + dy, act * act * 0.95 * f);
      });
    }

    // the lodestar at the vanishing point
    const S = 0.42;
    for (let dy = -8; dy <= 8; dy++)
      for (let dx = -8; dx <= 8; dx++) {
        const b = compassStar(dx / S, dy / S) * (0.75 + 0.25 * Math.sin(t * 1.3));
        if (b > 0.06) put(cx + dx, cy + dy, b);
      }

    ctx.fillStyle = getComputedStyle(canvas).color;
    drawMarks(ctx, grid, cols, rows, CELL);

    const level = Math.floor(z);
    if (level !== lastLevel) ((lastLevel = level), depth?.(level));
  };

  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const t0 = performance.now();
  let raf = 0, last = 0, lastLevel = -1, seen = true;

  const tick = (now: number) => {
    raf = 0;
    if (!seen || document.hidden) return;
    raf = requestAnimationFrame(tick);
    if (now - last < 50) return; // ~20 fps
    last = now;
    draw((now - t0) / 1000 + PERIOD * 0.4);
  };
  const run = () => (still ? draw(PERIOD * 0.4) : raf || (raf = requestAnimationFrame(tick)));

  if (size()) run();
  new IntersectionObserver(([e]) => ((seen = e.isIntersecting), seen && run())).observe(canvas);
  document.addEventListener("visibilitychange", () => !document.hidden && run());
  let rt: ReturnType<typeof setTimeout> | undefined;
  addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => size() && (still ? draw(PERIOD * 0.4) : run()), 150); });
}
