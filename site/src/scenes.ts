import { drawMarks, type Put } from "../../web/src/art/glyphs";
import { compassStar } from "../../web/src/art/mark";

// The feature scenes under the hero, in the app's ASCII marks. Each is a full-width canvas drawn on a grid of
// cells; one harness sizes it, runs it only while it's on screen, and hands it the pointer.
//   erase   two clouds of activations and the refusal direction between them; a plane closes on them and
//           projects the direction out, so the clouds fall flat into one
//   layers  a stack of layer sheets in isometric view; each token rises through them, lighting neurons, its path
//           wiring sheet to sheet, the logit lens growing surer toward the top
//   steer   the residual stream as flowing lines; a push parts it, and everything downstream stays bent
//   api     clients around the lodestar; requests curve in, the core pulses, and tokens stream back out

type Frame = { put: Put; cols: number; rows: number; t: number; pointer: { x: number; y: number } | null };

const hash = (...v: number[]) => { let h = 0; for (const x of v) h = Math.sin(h * 1.7 + x * 12.9898 + 78.233) * 43758.5453; return h - Math.floor(h); };
const clamp = (x: number, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const ease = (x: number) => { x = clamp(x); return x * x * (3 - 2 * x); };
const gauss = (i: number, k: number) => { // a standard normal sample, fixed per (i, k)
  const u = Math.max(1e-6, hash(i, k, 1)), v = hash(i, k, 2);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

function line(put: Put, x0: number, y0: number, x1: number, y1: number, b: number, every = 1) {
  const n = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
  for (let s = 0; s <= n; s += every) put(x0 + ((x1 - x0) * s) / n, y0 + ((y1 - y0) * s) / n, b);
}

export function scene(canvas: HTMLCanvasElement, cell: number, draw: (f: Frame) => void, still = 2) {
  let cols = 0, rows = 0, grid = new Float32Array(0), ctx!: CanvasRenderingContext2D;
  let pointer: { x: number; y: number } | null = null;

  const size = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight, dpr = devicePixelRatio || 1;
    if (!w || !h) return false;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cols = Math.ceil(w / cell);
    rows = Math.ceil(h / cell);
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

  const frame = (t: number) => {
    grid.fill(0);
    draw({ put, cols, rows, t, pointer });
    ctx.fillStyle = getComputedStyle(canvas).color;
    drawMarks(ctx, grid, cols, rows, cell);
  };

  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches, t0 = performance.now();
  let raf = 0, last = 0, seen = false;
  const tick = (now: number) => {
    raf = 0;
    if (!seen || document.hidden) return;
    raf = requestAnimationFrame(tick);
    if (now - last < 45) return; // ~22 fps
    last = now;
    frame((now - t0) / 1000);
  };
  const run = () => (reduced ? frame(still) : raf || (raf = requestAnimationFrame(tick)));

  canvas.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    pointer = { x: (e.clientX - r.left) / cell, y: (e.clientY - r.top) / cell };
  });
  canvas.addEventListener("pointerleave", () => (pointer = null));
  new IntersectionObserver(([e]) => ((seen = e.isIntersecting), seen && size() && run())).observe(canvas);
  document.addEventListener("visibilitychange", () => !document.hidden && run());
  let rt: ReturnType<typeof setTimeout> | undefined;
  addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => size() && (reduced ? frame(still) : run()), 150); });
  if (size()) frame(still);
}

// ---------------------------------------------------------------- erase: the refusal direction, projected out
type V3 = [number, number, number];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: V3): V3 => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };
const cross3 = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

const DIR = norm([1, 0.35, 0.25]);            // the refusal direction
const U = norm(cross3(DIR, [0, 1, 0])), W = cross3(DIR, U); // the plane it's erased against
const CLOUD = Array.from({ length: 900 }, (_, i) => ({
  side: i % 2 ? 1 : -1, // refused or answered
  p: [gauss(i, 1) * 0.22, gauss(i, 2) * 0.22, gauss(i, 3) * 0.22] as V3,
}));
const CYCLE = 11;

export function erase({ put, cols, rows, t }: Frame) {
  const k = t % CYCLE;
  const apart = 0.25 + 0.75 * ease(k / 2.2);                    // the clouds separate
  const arrow = ease((k - 1.6) / 1.4);                          // the direction draws itself between their means
  const plane = ease((k - 3.6) / 1.2) * (1 - ease((k - 9.6) / 1.2));
  const flat = ease((k - 5) / 2);                               // the projection
  const gone = ease((k - 9.8) / 1.0);                           // everything settles out before it starts again

  const yaw = t * 0.22, pitch = 0.42;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const S = Math.min(rows * 0.5, cols * 0.3), ox = cols / 2, oy = rows / 2;
  const proj = (v: V3): [number, number, number] => {
    const x = v[0] * cy + v[2] * sy, z = -v[0] * sy + v[2] * cy;
    const y = v[1] * cp - z * sp, d = v[1] * sp + z * cp;
    return [ox + x * S, oy - y * S, d];
  };
  const at = (v: V3, b: number) => { const [x, y, d] = proj(v); put(x, y, b * (0.62 + 0.38 * clamp(0.5 - d * 0.6))); };
  const fade = 1 - gone;

  // the plane: a dotted square across the direction, with a finer grid inside
  if (plane > 0) {
    const P = 1.05;
    for (let g = -4; g <= 4; g++) {
      const edge = Math.abs(g) === 4, b = (edge ? 0.38 : 0.13) * plane * fade;
      for (let s = -P; s <= P; s += 0.02) {
        const u = (g / 4) * P;
        if (edge || Math.round(s * 50) % 3 === 0) {
          at([U[0] * u + W[0] * s, U[1] * u + W[1] * s, U[2] * u + W[2] * s], b);
          at([U[0] * s + W[0] * u, U[1] * s + W[1] * u, U[2] * s + W[2] * u], b);
        }
      }
    }
  }

  // the clouds, falling flat onto the plane as the direction is taken out
  for (const [i, c] of CLOUD.entries()) {
    const along = c.side * apart * 0.62 + dot(c.p, DIR);
    const keep = along * (1 - flat);
    const off: V3 = [c.p[0] - dot(c.p, DIR) * DIR[0], c.p[1] - dot(c.p, DIR) * DIR[1], c.p[2] - dot(c.p, DIR) * DIR[2]];
    const v: V3 = [off[0] + DIR[0] * keep, off[1] + DIR[1] * keep, off[2] + DIR[2] * keep];
    const twinkle = 0.85 + 0.15 * Math.sin(t * 2 + i);
    at(v, (c.side > 0 ? 0.62 : 0.3) * (1 - flat * 0.3) * twinkle * fade);
  }

  // the direction: answered mean to refused mean, bright with a halo; once the plane closes it breaks into dust
  if (arrow > 0) {
    const a = -apart * 0.62, b = apart * 0.62, len = (b - a) * arrow;
    for (let s = 0; s <= len; s += 0.008) {
      const u = a + s, crumble = flat > 0 ? hash(Math.round(s * 125), Math.floor(t * 8)) < flat * 0.85 : false;
      if (crumble) continue;
      const v: V3 = [DIR[0] * u * (1 - flat), DIR[1] * u * (1 - flat), DIR[2] * u * (1 - flat)];
      const [x, y] = proj(v);
      put(x, y, (1 - flat * 0.5) * fade);
      put(x, y - 1, 0.35 * (1 - flat) * fade);
      put(x, y + 1, 0.35 * (1 - flat) * fade);
    }
    if (arrow > 0.95 && flat < 0.4) { // the head
      const [hx, hy] = proj([DIR[0] * b, DIR[1] * b, DIR[2] * b]), [tx, ty] = proj([DIR[0] * b * 0.82, DIR[1] * b * 0.82, DIR[2] * b * 0.82]);
      const dx = hx - tx, dy = hy - ty, l = Math.hypot(dx, dy) || 1, nx = -dy / l, ny = dx / l, w = 3.2 * (1 - flat * 2.5);
      line(put, hx, hy, tx + nx * w, ty + ny * w, 1);
      line(put, hx, hy, tx - nx * w, ty - ny * w, 1);
    }
  }
}

// ---------------------------------------------------------------- layers: a token rising through the stack
const SHEETS = 12, NI = 16, NJ = 9, RISE = 3.4;

export function layers({ put, cols, rows, t }: Frame) {
  const a = Math.min(4, (cols - 40) / (NI + NJ), (rows - 8) / ((NI + NJ) * 0.5 + (SHEETS - 1) * 1.3)), b = a * 0.5, gap = a * 1.3;
  const ox = cols / 2 - (NI - NJ) * a * 0.5 - 10, oy = rows / 2 - ((NI + NJ) * b - (SHEETS - 1) * gap) / 2;
  const at = (i: number, j: number, k: number): [number, number] => [ox + (i - j) * a, oy + (i + j) * b - k * gap];

  const token = Math.floor(t / RISE), phase = (t % RISE) / RISE, head = phase * (SHEETS + 2) - 1; // sheet the pulse is at
  const lit = (tok: number, k: number) => { // the neurons a token lights on a sheet
    const out: [number, number, number][] = [];
    for (let n = 0; n < 3; n++) out.push([Math.floor(hash(tok, k, n) * NI), Math.floor(hash(tok, k, n, 1) * NJ), 0.55 + 0.45 * hash(tok, k, n, 2)]);
    return out;
  };

  for (let k = 0; k < SHEETS; k++) {
    // the sheet: its rim and a lattice of resting neurons, brighter as the pulse passes
    const near = Math.exp(-((head - k) ** 2) / 0.8);
    const corners = [at(0, 0, k), at(NI - 1, 0, k), at(NI - 1, NJ - 1, k), at(0, NJ - 1, k)];
    for (let c = 0; c < 4; c++) line(put, ...corners[c], ...corners[(c + 1) % 4], 0.16 + 0.3 * near, 2);
    for (let i = 0; i < NI; i++) for (let j = 0; j < NJ; j++) if ((i + j) % 2 === 0) put(...at(i, j, k), 0.08 + 0.16 * near);

    // the logit lens: how sure this layer is of the next word, growing toward the top once the token has passed
    if (head > k) {
      const conf = 1 / (1 + Math.exp(-(k - SHEETS * 0.62) * 0.9)), [rx, ry] = at(NI - 1, NJ - 1, k);
      const n = Math.round(conf * 14 * clamp(head - k));
      for (let s = 0; s < n; s++) put(rx + 5 + s, ry, s === n - 1 ? 1 : 0.35 + 0.55 * conf);
    }
  }

  // the paths: this token's, wiring lit neurons sheet to sheet as it climbs, and the last one fading
  for (const [tok, age] of [[token - 1, 1], [token, 0]] as const) {
    const reach = age ? SHEETS : head, dim = age ? 0.3 * (1 - phase) ** 2 : 1;
    if (dim <= 0.02) continue;

    for (let k = 0; k < SHEETS && k <= reach; k++) {
      const here = lit(tok, k), pulse = age ? 0 : Math.exp(-((head - k) ** 2) / 0.6);
      for (const [i, j, act] of here) {
        const [x, y] = at(i, j, k);
        put(x, y, (0.6 + 0.4 * act) * dim);
        if (!age) for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) put(x + dx, y + dy, 0.4 * act); // a small plus
        if (pulse > 0.3) for (let d = -2; d <= 2; d++) (put(x + d, y + d, pulse * act), put(x + d, y - d, pulse * act));
      }

      if (k + 1 < SHEETS && k + 1 <= reach + 1) {
        const up = lit(tok, k + 1), grow = age ? 1 : clamp(reach - k);
        here.forEach(([i, j], n) => {
          const [x0, y0] = at(i, j, k), [x1, y1] = at(up[n][0], up[n][1], k + 1);
          line(put, x0, y0, x0 + (x1 - x0) * grow, y0 + (y1 - y0) * grow, 0.5 * dim, age ? 2 : 1);
        });
      }
    }
  }
}

// ---------------------------------------------------------------- steer: a push parting the stream
const LINES = 23;
let push = { x: 0, y: 0, set: false };

export function steer({ put, cols, rows, t, pointer }: Frame) {
  const drift = { x: cols * (0.42 + 0.12 * Math.sin(t * 0.31)), y: rows * (0.5 + 0.22 * Math.sin(t * 0.47 + 1)) };
  const aim = pointer ?? drift;
  if (!push.set) push = { ...aim, set: true };
  push.x += (aim.x - push.x) * 0.12;
  push.y += (aim.y - push.y) * 0.12;

  const A = rows * 0.13, w = rows * 0.2, s = cols * 0.03;
  const bend = (x: number, y0: number) => { // how far the line that starts at y0 is pushed by the time it reaches x:
    const d = y0 - push.y, side = Math.sign(d) || 1;  // most for the nearest, never so much that two lines cross
    return side * A * Math.exp(-Math.abs(d) / w) / (1 + Math.exp(-(x - push.x) / s));
  };

  for (let n = 0; n < LINES; n++) {
    const y0 = rows * (0.08 + (0.84 * n) / (LINES - 1)), near = Math.exp(-Math.abs(y0 - push.y) / (w * 0.7));
    const speed = 9 + 7 * hash(n, 1);

    for (let x = 0; x < cols; x += 1) { // the line, dotted, a little brighter where it's been pushed
      if (x % 2 && near < 0.3) continue;
      put(x, y0 + bend(x, y0), 0.1 + 0.22 * near);
    }

    for (let p = 0; p < 4; p++) { // particles riding the line, each with a tail
      const hx = ((t * speed + hash(n, p, 2) * cols * 1.3) % (cols * 1.15)) - cols * 0.05;
      for (let u = 0; u < 9; u++) {
        const x = hx - u * 1.3;
        put(x, y0 + bend(x, y0), (0.35 + 0.6 * near) * (1 - u / 9) * (0.6 + 0.4 * hash(n, p)));
      }
    }
  }

  // the push: a bright core in a ring, a second ring breathing out from it
  const { x: px, y: py } = push, breathe = (t * 0.9) % 1;
  for (let a = 0; a < Math.PI * 2; a += 0.06) {
    put(px + Math.cos(a) * 5.2, py + Math.sin(a) * 5.2, 0.5);
    put(px + Math.cos(a) * (6 + breathe * 9), py + Math.sin(a) * (6 + breathe * 9), 0.45 * (1 - breathe));
  }
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) put(px + dx, py + dy, Math.hypot(dx, dy) < 1.5 ? 1 : 0.6);
}

// ---------------------------------------------------------------- api: requests in, tokens out
const CLIENTS = 7, TRIP = 2.6;

export function api({ put, cols, rows, t }: Frame) {
  const cx = cols / 2, cy = rows / 2, rx = Math.min(cols * 0.4, rows * 1.15), ry = rows * 0.38;
  const at = (c: number): [number, number] => { const a = -Math.PI / 2 + (c / CLIENTS) * Math.PI * 2 + 0.2; return [cx + Math.cos(a) * rx, cy + Math.sin(a) * ry]; };

  for (let a = 0; a < Math.PI * 2; a += 0.012) if (Math.round(a / 0.012) % 3 === 0) put(cx + Math.cos(a) * rx, cy + Math.sin(a) * ry, 0.12); // the orbit

  // requests: each client sends on its own beat; a packet curves in, the core answers with a stream of tokens
  let pulse = 0;
  for (let c = 0; c < CLIENTS; c++) {
    const [x, y] = at(c), beat = TRIP * (1.6 + hash(c, 1) * 1.6), k = ((t + hash(c, 2) * beat) % beat) / TRIP;
    const busy = k < 1;

    // the client: a small window with lines of text in it
    const edge = busy ? 0.62 : 0.3;
    for (let d = -7; d <= 7; d++) (put(x + d, y - 4, edge), put(x + d, y + 4, edge));
    for (let d = -3; d <= 3; d++) (put(x - 7, y + d, edge), put(x + 7, y + d, edge));
    for (let d = -6; d <= 6; d++) put(x + d, y - 2, 0.16); // its title bar
    for (let r = 0; r <= 2; r++) for (let d = -5; d <= 4 - r * 3; d++) if (hash(c, r, d) > 0.2) put(x + d, y + r, busy ? 0.4 : 0.22);

    if (!busy) continue;
    const mx = (x + cx) / 2 + (cy - y) * 0.25, my = (y + cy) / 2 + (x - cx) * 0.25 * (ry / rx); // a curved path
    const along = (u: number): [number, number] => [(1 - u) ** 2 * x + 2 * (1 - u) * u * mx + u * u * cx, (1 - u) ** 2 * y + 2 * (1 - u) * u * my + u * u * cy];

    for (let u = 0; u <= 1; u += 0.02) put(...along(u), 0.1); // its path, faint
    if (k < 0.4) { // in: one bright packet with a short tail
      const u = k / 0.4;
      for (let s = 0; s < 6; s++) put(...along(Math.max(0, u - s * 0.025)), 1 - s * 0.15);
    } else { // out: tokens streaming back along the same path
      const u0 = (k - 0.4) / 0.6;
      pulse = Math.max(pulse, 1 - clamp((k - 0.4) * 4));
      for (let s = 0; s < 14; s++) {
        const u = 1 - clamp(u0 * 1.5 - s * 0.045);
        if (u <= 0 || u >= 1) continue;
        put(...along(u), s % 4 === 3 ? 0 : 0.8 - s * 0.03);
      }
    }
  }

  // the lodestar at the centre, with a ring going out each time it takes a request
  const S = 0.62 + 0.08 * pulse;
  for (let dy = -12; dy <= 12; dy++)
    for (let dx = -12; dx <= 12; dx++) {
      const b = compassStar(dx / S, dy / S);
      if (b > 0.06) put(cx + dx, cy + dy, b);
    }
  if (pulse > 0) for (let a = 0; a < Math.PI * 2; a += 0.05) put(cx + Math.cos(a) * (13 + (1 - pulse) * 10), cy + Math.sin(a) * (13 + (1 - pulse) * 10), 0.6 * pulse);
}
