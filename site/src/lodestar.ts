import { drawMarks, type Put } from "../../web/src/art/glyphs";
import { compassStar } from "../../web/src/art/mark";

// The hero: an endless fall into an armillary sphere of the model. Each sphere is a lattice of neurons turning in
// perspective, faint at the back and bright at the front, lit where two slow foci of attention pass, held in gimbal
// rings (one ticked like an ecliptic) with great-circle arcs carrying pulses over it. At its heart is the next
// sphere, K× smaller and turning its own way, and inside that the next, forever: the camera zooms steadily in, each
// sphere swelling past and fading as the one inside takes its place. The lodestar burns at the vanishing point,
// and an astrolabe ring holds the frame. Scrolling quickens the fall; the pointer tilts it.
const CELL = 4;
const NODES = 900;
const ARCS = 6;
const K = 2.7;        // how much bigger each sphere is than the one inside it
const PERIOD = 6;     // seconds to fall through one sphere
const TILT = 0.41;    // the ticked ring's tilt against the equator

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

/** at: the vanishing point as fractions of the canvas. boost: extra speed, multiplying the fall (scrolling).
 *  depth: called with the sphere the camera has reached, each time it passes one. */
export function startLodestar(canvas: HTMLCanvasElement, opts: { at?: () => [number, number]; boost?: () => number; depth?: (level: number) => void } = {}) {
  const { at = () => [0.5, 0.5], boost = () => 0, depth } = opts;
  let cols = 0, rows = 0, grid = new Float32Array(0), ctx!: CanvasRenderingContext2D;

  const size = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight, dpr = devicePixelRatio || 1;
    if (!w || !h) return false;
    if (canvas.width === Math.round(w * dpr) && canvas.height === Math.round(h * dpr)) return false;
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
    if (x < 0 || y < 0 || x >= cols || y >= rows || b <= 0.02) return;
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

  let z = 0.35, lastT = -1, lastLevel = NaN;
  const draw = (t: number) => {
    grid.fill(0);
    const [fx, fy] = at(), cx = cols * fx, cy = rows * fy;
    const Rs = Math.min(rows * 0.3, cols * 0.27);
    const dt = lastT < 0 ? 0 : Math.min(0.2, t - lastT);
    lastT = t;
    z += (dt / PERIOD) * (1 + boost());
    px += (tx - px) * 0.06;
    py += (ty - py) * 0.06;
    const pitch = 0.3 + 0.08 * Math.sin(t * 0.21) + py * 0.18, cp = Math.cos(pitch), sp = Math.sin(pitch);
    const shade = (d: number) => 0.18 + 0.82 * clamp(d) ** 1.4;

    // one sphere of the regress: level L, radius r (in Rs), brightness f
    const sphere = (L: number, r: number, f: number) => {
      const spin = t * 0.16 * (L % 2 ? -1.25 : 1) + L * 0.9 + px * 0.3, tilt = 0.35 * Math.sin(L * 1.7);
      const cs = Math.cos(spin), ss = Math.sin(spin), ct = Math.cos(tilt), st = Math.sin(tilt);
      // the sphere's own frame → screen cells, depth 0 (back) .. 1 (front); perspective in proportion to its size
      const project = (v: V, k = 1): [number, number, number] => {
        const y1 = v[1] * ct - v[2] * st, z1 = v[1] * st + v[2] * ct;
        const x2 = v[0] * cs + z1 * ss, z2 = -v[0] * ss + z1 * cs;
        const y3 = y1 * cp - z2 * sp, z3 = y1 * sp + z2 * cp;
        const g = (3.4 / (3.4 - z3 * k)) * Rs * r * k;
        return [cx + x2 * g, cy - y3 * g, (z3 + 1) / 2];
      };
      const px2 = r * Rs; // its radius on screen, in cells

      // a circle on its shell: around axis n, at radius k
      const ring = (n: V, k: number, b: number, ticks = 0) => {
        n = norm(n);
        const u = norm(Math.abs(n[1]) < 0.9 ? [n[2], 0, -n[0]] : [1, 0, 0]);
        const w: V = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
        const step = Math.min(0.25, 1.6 / (px2 * k));
        for (let a = 0, s = 0; a < Math.PI * 2; a += step, s++) {
          const v: V = [u[0] * Math.cos(a) + w[0] * Math.sin(a), u[1] * Math.cos(a) + w[1] * Math.sin(a), u[2] * Math.cos(a) + w[2] * Math.sin(a)];
          const [x, y, d] = project(v, k);
          put(x, y, b * f * shade(d));
          if (ticks && s % ticks === 0 && px2 > 30)
            for (let j = 1; j <= 3; j++) { const [x2, y2, d2] = project(v, k + j * 0.018); put(x2, y2, b * 1.4 * f * shade(d2)); }
        }
      };

      // faint parallels, for the globe's form
      if (px2 > 20)
        for (let k = -2; k <= 2; k++) {
          const lat = (k * Math.PI) / 7, rr = Math.cos(lat), h = Math.sin(lat);
          for (let a = 0; a < Math.PI * 2; a += 1.4 / (px2 * rr)) {
            const [x, y, d] = project([Math.cos(a) * rr, h, Math.sin(a) * rr]);
            put(x, y, 0.13 * f * shade(d));
          }
        }

      // the neurons, lit where its two foci of attention pass
      const tf = t + L * 7.3;
      const foci: V[] = [
        norm([Math.cos(tf * 0.23), Math.sin(tf * 0.17) * 0.8, Math.sin(tf * 0.23)]),
        norm([Math.sin(tf * 0.13 + 2), Math.cos(tf * 0.19) * 0.7, Math.cos(tf * 0.11 + 1)]),
      ];
      const every = Math.max(1, Math.round(NODES / Math.max(30, px2 * px2 * 0.9)));
      for (let i = (L * 5) % every; i < NODES; i += every) {
        const v = lattice[i], [x, y, d] = project(v);
        const heat = Math.max(...foci.map((q) => Math.exp(-(1 - dot(v, q)) / 0.11)));
        const flick = 0.5 + 0.5 * Math.sin(t * (1.2 + hash(i, 1) * 2) + hash(i, L, 2) * 6.3);
        const b = (0.24 + heat * (0.6 + 0.4 * flick) + hash(i, L, 3) ** 6 * 0.35) * f * shade(d);
        put(x, y, b);
        if (heat * d > 0.45 && px2 > 25) for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) put(x + dx, y + dy, b * 0.55);
      }

      // signals: great-circle arcs lifting off the surface, a pulse running along each
      if (px2 > 25)
        for (let k = 0; k < ARCS; k++) {
          const phase = t / 2.6 + k / ARCS, slot = Math.floor(phase), s = phase - slot;
          const a = lattice[Math.floor(hash(slot, k, L, 1) * NODES)], c = lattice[Math.floor(hash(slot, k, L, 2) * NODES)];
          const cos = dot(a, c);
          if (cos < -0.85 || cos > 0.98) continue;
          const om = Math.acos(cos), so = Math.sin(om), drawn = clamp(s * 1.8), fade = clamp((1 - s) * 3);
          const n = Math.ceil(om * px2 * 1.2);
          for (let j = 0; j <= n * drawn; j++) {
            const u = j / n, p1 = Math.sin((1 - u) * om) / so, p2 = Math.sin(u * om) / so;
            const v: V = [a[0] * p1 + c[0] * p2, a[1] * p1 + c[1] * p2, a[2] * p1 + c[2] * p2];
            const [x, y, d] = project(v, 1 + 0.28 * Math.sin(Math.PI * u));
            const head = Math.exp(-((u - drawn) ** 2) / 0.002);
            if (j % 2 === 0 || head > 0.3) put(x, y, (0.34 + head * 0.9) * fade * f * shade(d));
          }
        }

      // its gimbals: an equator, and a ticked ecliptic precessing round the pole
      ring([0, 1, 0], 1.14, 0.5);
      ring([Math.sin(TILT) * Math.cos(t * 0.09 + L), Math.cos(TILT), Math.sin(TILT) * Math.sin(t * 0.09 + L)], 1.22, 0.62, 5);
    };

    // the regress, outermost first; each sphere comes up out of the one before and swells past the camera
    const level = Math.floor(z);
    for (let L = level - 1; L <= level + 4; L++) {
      const r = K ** (z - L);
      const f = clamp((r - 0.05) / 0.3) * clamp((3 - r) / 1.6);
      if (f > 0) sphere(L, r, f);
    }
    if (level !== lastLevel) ((lastLevel = level), depth?.(level));

    // the astrolabe round it all, in the plane of the page
    const R2 = Rs * 1.5;
    for (let deg = 0; deg < 360; deg += 1.5) {
      const a = (deg * Math.PI) / 180 - z * 0.12, major = deg % 30 === 0, minor = deg % 7.5 === 0;
      const len = major ? 4 : minor ? 2 : 0.6;
      for (let k = 0; k <= len; k += 0.8) put(cx + Math.cos(a) * (R2 + k), cy + Math.sin(a) * (R2 + k), major ? 0.42 : 0.16);
    }

    // the lodestar at the vanishing point
    const star = Rs * 0.2, S = star / 15, pulse = 0.85 + 0.15 * Math.sin(t * 1.1);
    for (let dy = -Math.ceil(star); dy <= star; dy++)
      for (let dx = -Math.ceil(star); dx <= star; dx++) {
        const x = dx / S, y = dy / S, r = Math.hypot(x, y);
        if (r > 11.5) continue; // without its ring: the spheres stand in for it
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
  // resize the bitmap whenever the canvas's box changes, not just on a window resize: on a phone the box settles after
  // the first frame (the viewport units shift as the browser's bars appear), and a stale bitmap draws stretched
  new ResizeObserver(() => size() && (still ? draw(4) : lastT >= 0 ? draw(lastT) : run())).observe(canvas);
}
