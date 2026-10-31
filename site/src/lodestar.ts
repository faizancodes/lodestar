import { drawMarks, type Put } from "../../web/src/art/glyphs";
import { compassStar } from "../../web/src/art/mark";

// The hero: an armillary sphere of the model, turning in true perspective with the lodestar at its core. Its
// surface is a lattice of neurons, faint at the back and bright at the front, lit where two slow foci of attention
// pass over them. Great-circle arcs lift off the surface and carry pulses between neurons. Three gimbal rings, one
// of them ticked like an ecliptic, turn on their own axes, and the pole axis runs up through the star. An outer
// astrolabe ring holds it all. Scrolling winds it faster; the pointer tilts it.
const CELL = 4;
const NODES = 900;
const ARCS = 8;
const TILT = 0.41; // the ticked ring's tilt against the equator

const hash = (...v: number[]) => { let h = 0; for (const x of v) h = Math.sin(h * 1.7 + x * 12.9898 + 78.233) * 43758.5453; return h - Math.floor(h); };
const clamp = (x: number) => Math.max(0, Math.min(1, x));
type V = [number, number, number];
const dot = (a: V, b: V) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: V): V => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// neurons spread evenly over the unit sphere
const lattice: V[] = Array.from({ length: NODES }, (_, i) => {
  const y = 1 - (2 * (i + 0.5)) / NODES, r = Math.sqrt(1 - y * y), a = i * Math.PI * (3 - Math.sqrt(5));
  return [Math.cos(a) * r, y, Math.sin(a) * r];
});

/** at: the sphere's centre as fractions of the canvas. boost: extra turn, multiplying the spin (scrolling winds it). */
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

  // the pointer, as -1..1 across the canvas, eased
  let px = 0, py = 0, tx = 0, ty = 0;
  addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    tx = clamp((e.clientX - r.left) / r.width) * 2 - 1;
    ty = clamp((e.clientY - r.top) / r.height) * 2 - 1;
  }, { passive: true });

  let yaw = 0.6, lastT = -1;
  const draw = (t: number) => {
    grid.fill(0);
    const [fx, fy] = at(), cx = cols * fx, cy = rows * fy;
    const Rs = Math.min(rows * 0.3, cols * 0.27), D = Rs * 3.4;
    const dt = lastT < 0 ? 0 : Math.min(0.2, t - lastT);
    lastT = t;
    yaw += dt * 0.16 * (1 + boost());
    px += (tx - px) * 0.06;
    py += (ty - py) * 0.06;
    const pitch = 0.32 + 0.08 * Math.sin(t * 0.21) + py * 0.18, y0 = yaw + px * 0.3;
    const cyw = Math.cos(y0), syw = Math.sin(y0), cp = Math.cos(pitch), sp = Math.sin(pitch);

    // world (unit sphere, y up) → screen cells, with depth 0 (back) .. 1 (front)
    const project = (v: V, r = 1): [number, number, number] => {
      const x1 = v[0] * cyw + v[2] * syw, z1 = -v[0] * syw + v[2] * cyw;
      const y2 = v[1] * cp - z1 * sp, z2 = v[1] * sp + z1 * cp;
      const f = D / (D - z2 * Rs * r);
      return [cx + x1 * Rs * r * f, cy - y2 * Rs * r * f, (z2 + 1) / 2];
    };
    const shade = (d: number) => 0.18 + 0.82 * d ** 1.4;

    // a circle on the sphere's shell: around axis n, at radius r
    const ring = (n: V, r: number, b: number, step = 0.035, ticks = 0) => {
      n = norm(n);
      const u = norm(Math.abs(n[1]) < 0.9 ? [n[2], 0, -n[0]] : [1, 0, 0]);
      const w: V = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
      for (let a = 0; a < Math.PI * 2; a += step) {
        const v: V = [u[0] * Math.cos(a) + w[0] * Math.sin(a), u[1] * Math.cos(a) + w[1] * Math.sin(a), u[2] * Math.cos(a) + w[2] * Math.sin(a)];
        const [x, y, d] = project(v, r);
        put(x, y, b * shade(d));
        if (ticks && Math.round(a / step) % ticks === 0)
          for (let k = 1; k <= 3; k++) { const [x2, y2, d2] = project(v, r + k * 0.018); put(x2, y2, b * 1.4 * shade(d2)); }
      }
    };

    // faint parallels, for the globe's form
    for (let k = -2; k <= 2; k++) {
      const lat = (k * Math.PI) / 7, r = Math.cos(lat), h = Math.sin(lat);
      for (let a = 0; a < Math.PI * 2; a += 0.05) {
        const [x, y, d] = project([Math.cos(a) * r, h, Math.sin(a) * r]);
        put(x, y, 0.13 * shade(d));
      }
    }

    // two slow foci of attention wandering the sphere
    const foci: V[] = [
      norm([Math.cos(t * 0.23), Math.sin(t * 0.17) * 0.8, Math.sin(t * 0.23)]),
      norm([Math.sin(t * 0.13 + 2), Math.cos(t * 0.19) * 0.7, Math.cos(t * 0.11 + 1)]),
    ];

    // the neurons
    lattice.forEach((v, i) => {
      const [x, y, d] = project(v);
      const heat = Math.max(...foci.map((f) => Math.exp(-(1 - dot(v, f)) / 0.11)));
      const flick = 0.5 + 0.5 * Math.sin(t * (1.2 + hash(i, 1) * 2) + hash(i, 2) * 6.3);
      const b = (0.24 + heat * (0.6 + 0.4 * flick) + hash(i, 3) ** 6 * 0.35) * shade(d);
      put(x, y, b);
      if (heat * d > 0.45) for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) put(x + dx, y + dy, b * 0.55);
    });

    // signals: great-circle arcs lifting off the surface, a pulse running along each
    for (let k = 0; k < ARCS; k++) {
      const phase = t / 2.6 + k / ARCS, slot = Math.floor(phase), s = phase - slot;
      const a = lattice[Math.floor(hash(slot, k, 1) * NODES)], c = lattice[Math.floor(hash(slot, k, 2) * NODES)];
      const cos = dot(a, c);
      if (cos < -0.85 || cos > 0.98) continue;
      const om = Math.acos(cos), so = Math.sin(om), drawn = clamp(s * 1.8), fade = clamp((1 - s) * 3);
      const n = Math.ceil(om * Rs * 1.2);
      for (let j = 0; j <= n * drawn; j++) {
        const u = j / n, p1 = Math.sin((1 - u) * om) / so, p2 = Math.sin(u * om) / so;
        const v: V = [a[0] * p1 + c[0] * p2, a[1] * p1 + c[1] * p2, a[2] * p1 + c[2] * p2];
        const [x, y, d] = project(v, 1 + 0.28 * Math.sin(Math.PI * u));
        const head = Math.exp(-((u - drawn) ** 2) / 0.002);
        if (j % 2 === 0 || head > 0.3) put(x, y, (0.34 + head * 0.9) * fade * shade(d));
      }
    }

    // the gimbals: equator, a ticked ecliptic precessing round the pole, and a meridian turning on its own
    ring([0, 1, 0], 1.14, 0.5);
    ring([Math.sin(TILT) * Math.cos(t * 0.09), Math.cos(TILT), Math.sin(TILT) * Math.sin(t * 0.09)], 1.2, 0.62, 0.03, 5);
    ring([Math.cos(t * 0.07), 0, Math.sin(t * 0.07)], 1.26, 0.36);

    // the pole axis, through the star
    for (let h = -1.45; h <= 1.45; h += 0.025) {
      if (Math.abs(h) < 0.3) continue;
      const [x, y, d] = project([0, h, 0]);
      put(x, y, (Math.abs(h) > 1.3 ? 0.6 : 0.26) * shade(d));
    }

    // the astrolabe round it all, in the plane of the page
    const R2 = Rs * 1.5;
    for (let deg = 0; deg < 360; deg += 1.5) {
      const a = (deg * Math.PI) / 180 - yaw * 0.25, major = deg % 30 === 0, minor = deg % 7.5 === 0;
      const len = major ? 4 : minor ? 2 : 0.6;
      for (let k = 0; k <= len; k += 0.8) put(cx + Math.cos(a) * (R2 + k), cy + Math.sin(a) * (R2 + k), major ? 0.42 : 0.16);
    }

    // the lodestar at the core, light running out along its rays
    const star = Rs * 0.4, S = star / 15, pulse = 0.85 + 0.15 * Math.sin(t * 1.1);
    for (let dy = -Math.ceil(star); dy <= star; dy++)
      for (let dx = -Math.ceil(star); dx <= star; dx++) {
        const x = dx / S, y = dy / S, r = Math.hypot(x, y);
        if (r > 11.5) continue; // without its ring: the sphere stands in for it
        const b = compassStar(x, y);
        if (b < 0.06) continue;
        put(cx + dx, cy + dy, b * pulse * (0.8 + 0.2 * Math.sin(r * 0.9 - t * 3.2)));
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
    if (now - last < 40) return; // ~25 fps
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
