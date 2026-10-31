import { drawMarks, type Put } from "../../web/src/art/glyphs";
import { compassStar } from "../../web/src/art/mark";

// The hero: the lodestar as the pole of a night sky. Its mark burns at the centre, light running out along its
// rays, held in an astrolabe of turning rings: an outer ring of degree ticks, and an inner one of 32 notches, a
// layer each, lit where the model is busy. Around it the sky wheels, every star (a neuron) dragging a trail as in a
// long exposure of the pole, and now and then a few flare together and are joined into a passing constellation.
const CELL = 4;
const STARS = 280;
const SPIN = 0.05; // radians a second the sky turns

const hash = (...v: number[]) => { let h = 0; for (const x of v) h = Math.sin(h * 1.7 + x * 12.9898 + 78.233) * 43758.5453; return h - Math.floor(h); };
const clamp = (x: number) => Math.max(0, Math.min(1, x));

/** at: the pole's place as fractions of the canvas. boost: extra spin, multiplying the turn (scrolling winds it). */
export function startLodestar(canvas: HTMLCanvasElement, opts: { at?: () => [number, number]; boost?: () => number } = {}) {
  const { at = () => [0.5, 0.5], boost = () => 0 } = opts;
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

  const line = (x0: number, y0: number, x1: number, y1: number, b: number, every = 2) => {
    const n = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
    for (let s = 0; s <= n; s += every) put(x0 + ((x1 - x0) * s) / n, y0 + ((y1 - y0) * s) / n, b);
  };

  let turn = 0, lastT = -1;
  const draw = (t: number) => {
    grid.fill(0);
    const [fx, fy] = at(), cx = cols * fx, cy = rows * fy;
    const R = Math.hypot(Math.max(cx, cols - cx), Math.max(cy, rows - cy));
    const extra = boost(), dt = lastT < 0 ? 0 : Math.min(0.2, t - lastT);
    turn += dt * SPIN * (1 + extra);
    lastT = t;
    const star = Math.min(rows * 0.27, cols * 0.2); // the mark's radius, in cells

    // the sky wheeling about the pole, each star trailing an arc behind it
    const trail = 0.22 + extra * 0.05;
    const flares: [number, number, number][] = [];
    for (let i = 0; i < STARS; i++) {
      const r = star * 1.5 + (R - star * 1.2) * Math.sqrt(hash(i, 1)), a = hash(i, 2) * Math.PI * 2 + turn;
      const mag = 0.18 + hash(i, 3) ** 2.5 * 0.6;
      const flare = clamp(Math.sin(t * 0.35 + hash(i, 4) * 40) * 6 - 5.2); // rare, brief
      const steps = Math.max(2, Math.round(r * trail));
      for (let s = 0; s <= steps; s++) {
        const aa = a - (s / steps) * trail;
        put(cx + Math.cos(aa) * r, cy + Math.sin(aa) * r, mag * (1 - s / (steps + 1)) ** 1.6 * 0.55);
      }
      const x = cx + Math.cos(a) * r, y = cy + Math.sin(a) * r, b = Math.min(1, mag + flare * 0.8);
      put(x, y, b);
      if (flare > 0.2) {
        flares.push([x, y, flare]);
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) put(x + dx, y + dy, flare * 0.6);
      }
    }
    // flaring stars near each other join into a constellation while they burn
    for (let i = 0; i < flares.length; i++)
      for (let j = i + 1; j < flares.length; j++) {
        const [x0, y0, f0] = flares[i], [x1, y1, f1] = flares[j];
        if (Math.hypot(x1 - x0, y1 - y0) < star * 2.2) line(x0, y0, x1, y1, Math.min(f0, f1) * 0.45, 2);
      }

    // the astrolabe: degree ticks turning one way, the layers' notches the other
    const r1 = star * 1.32, r2 = star * 1.12;
    for (let d = 0; d < 360; d += 2) {
      const a = (d * Math.PI) / 180 + turn * 0.6, long = d % 30 === 0 ? 4 : d % 10 === 0 ? 2.2 : 1;
      for (let k = 0; k <= long; k += 0.8) put(cx + Math.cos(a) * (r1 + k), cy + Math.sin(a) * (r1 + k), d % 30 === 0 ? 0.5 : 0.24);
    }
    for (let L = 0; L < 32; L++) {
      const a = (L / 32) * Math.PI * 2 - turn * 1.4, busy = clamp(Math.sin(t * 0.9 + L * 1.9) * 1.6 - 0.4);
      for (let k = -1; k <= 1; k += 0.5) put(cx + Math.cos(a + k * 0.03) * r2, cy + Math.sin(a + k * 0.03) * r2, 0.2 + busy * 0.75);
    }
    for (let a = 0; a < Math.PI * 2; a += 1.6 / r2) put(cx + Math.cos(a) * (r2 - 3), cy + Math.sin(a) * (r2 - 3), 0.12);

    // the lodestar itself, light running out along its rays
    const S = star / 15, pulse = 0.88 + 0.12 * Math.sin(t * 1.1);
    for (let dy = -Math.ceil(star); dy <= star; dy++)
      for (let dx = -Math.ceil(star); dx <= star; dx++) {
        const x = dx / S, y = dy / S, r = Math.hypot(x, y);
        if (r > 12 && r < 13.6) continue; // the mark's own ring: the astrolabe stands in for it
        let b = compassStar(x, y);
        if (b < 0.06) continue;
        b *= pulse * (0.82 + 0.18 * Math.sin(r * 0.9 - t * 3.2));
        put(cx + dx, cy + dy, b);
      }

    ctx.fillStyle = getComputedStyle(canvas).color;
    drawMarks(ctx, grid, cols, rows, CELL);
  };

  const still = matchMedia("(prefers-reduced-motion: reduce)").matches, t0 = performance.now();
  let raf = 0, last = 0, seen = true;
  const tick = (now: number) => {
    raf = 0;
    if (!seen || document.hidden) return;
    raf = requestAnimationFrame(tick);
    if (now - last < 45) return; // ~22 fps
    last = now;
    draw((now - t0) / 1000);
  };
  const run = () => (still ? draw(4) : raf || (raf = requestAnimationFrame(tick)));

  if (size()) run();
  new IntersectionObserver(([e]) => ((seen = e.isIntersecting), seen && run())).observe(canvas);
  document.addEventListener("visibilitychange", () => !document.hidden && run());
  let rt: ReturnType<typeof setTimeout> | undefined;
  addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => size() && (still ? draw(4) : run()), 150); });
}
