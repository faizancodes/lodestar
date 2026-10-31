import type { Put } from "../../web/src/art/glyphs";
import { compassStar } from "../../web/src/art/mark";

// The scroll sequence's scenes, in the app's ASCII marks. Each tells its feature's story as u runs 0 → 1 (how far
// the reader has scrolled through that step); t keeps the ambient motion going (twinkle, flow, traffic).
//   erase   two clouds of activations draw apart, the refusal direction draws between them, a plane closes on
//           them and projects it out, so the clouds fall flat into one
//   layers  the layer stack unfolds from a block into sheets, and a token climbs it, lighting neurons and wiring
//           its path sheet to sheet, the logit lens growing surer toward the top
//   steer   the residual stream; a push slides in and parts it, and everything downstream stays bent
//   chat    one question, two replies writing side by side, a gauge reading each as it goes
//   api     the lodestar; the orbit draws round, clients arrive on it, requests curve in and tokens stream back

export type Frame = { put: Put; cols: number; rows: number; t: number; u: number; pointer: { x: number; y: number } | null };

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

const star = (put: Put, cx: number, cy: number, S: number, k = 1) => {
  const R = Math.ceil(15 * S);
  for (let dy = -R; dy <= R; dy++)
    for (let dx = -R; dx <= R; dx++) {
      const b = compassStar(dx / S, dy / S) * k;
      if (b > 0.06) put(cx + dx, cy + dy, b);
    }
};

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

export function erase({ put, cols, rows, t, u }: Frame) {
  const apart = 0.12 + 0.88 * ease(u / 0.25);   // the clouds separate
  const arrow = ease((u - 0.18) / 0.2);         // the direction draws itself between their means
  const plane = ease((u - 0.4) / 0.15);         // the plane closes in
  const flat = ease((u - 0.55) / 0.3);          // the projection

  const yaw = t * 0.06 + u * 1.5 - 0.4, pitch = 0.42;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const S = Math.min(rows * 0.5, cols * 0.32) * (0.9 + 0.1 * ease(u / 0.4)), ox = cols / 2, oy = rows / 2;
  const proj = (v: V3): [number, number, number] => {
    const x = v[0] * cy + v[2] * sy, z = -v[0] * sy + v[2] * cy;
    const y = v[1] * cp - z * sp, d = v[1] * sp + z * cp;
    return [ox + x * S, oy - y * S, d];
  };
  const at = (v: V3, b: number) => { const [x, y, d] = proj(v); put(x, y, b * (0.62 + 0.38 * clamp(0.5 - d * 0.6))); };

  if (plane > 0) { // a dotted square across the direction, with a finer grid inside, closing in from wide
    const P = 1.05 * (1.6 - 0.6 * plane);
    for (let g = -4; g <= 4; g++) {
      const edge = Math.abs(g) === 4, b = (edge ? 0.4 : 0.13) * plane * (1 + flat * 0.4);
      for (let s = -P; s <= P; s += 0.02) {
        const q = (g / 4) * P;
        if (edge || Math.round(s * 50) % 3 === 0) {
          at([U[0] * q + W[0] * s, U[1] * q + W[1] * s, U[2] * q + W[2] * s], b);
          at([U[0] * s + W[0] * q, U[1] * s + W[1] * q, U[2] * s + W[2] * q], b);
        }
      }
    }
  }

  for (const [i, c] of CLOUD.entries()) { // the clouds, falling flat onto the plane
    const along = (c.side * apart * 0.62 + dot(c.p, DIR)) * (1 - flat);
    const off: V3 = [c.p[0] - dot(c.p, DIR) * DIR[0], c.p[1] - dot(c.p, DIR) * DIR[1], c.p[2] - dot(c.p, DIR) * DIR[2]];
    const twinkle = 0.85 + 0.15 * Math.sin(t * 2 + i);
    at([off[0] + DIR[0] * along, off[1] + DIR[1] * along, off[2] + DIR[2] * along], (c.side > 0 ? 0.62 : 0.3) * (1 - flat * 0.3) * twinkle);
  }

  if (arrow > 0) { // the direction: bright, with a halo; as the plane takes it, it breaks into dust
    const a = -apart * 0.62, b = apart * 0.62, len = (b - a) * arrow;
    for (let s = 0; s <= len; s += 0.008) {
      if (flat > 0 && hash(Math.round(s * 125), Math.floor(t * 8)) < flat * 0.9) continue;
      const q = (a + s) * (1 - flat), [x, y] = proj([DIR[0] * q, DIR[1] * q, DIR[2] * q]);
      put(x, y, 1 - flat * 0.5);
      put(x, y - 1, 0.35 * (1 - flat));
      put(x, y + 1, 0.35 * (1 - flat));
    }
    if (arrow > 0.95 && flat < 0.4) { // the head
      const [hx, hy] = proj([DIR[0] * b, DIR[1] * b, DIR[2] * b]), [tx, ty] = proj([DIR[0] * b * 0.82, DIR[1] * b * 0.82, DIR[2] * b * 0.82]);
      const dx = hx - tx, dy = hy - ty, l = Math.hypot(dx, dy) || 1, nx = -dy / l, ny = dx / l, w = 3.2 * (1 - flat * 2.5);
      line(put, hx, hy, tx + nx * w, ty + ny * w, 1);
      line(put, hx, hy, tx - nx * w, ty - ny * w, 1);
    }
  }
}

// ---------------------------------------------------------------- layers: a token climbing the stack
const SHEETS = 12, NI = 16, NJ = 9;

export function layers({ put, cols, rows, t, u }: Frame) {
  const unfold = ease(u / 0.3);
  const a = Math.min(4, (cols - 40) / (NI + NJ), (rows - 8) / ((NI + NJ) * 0.5 + (SHEETS - 1) * 1.3)), b = a * 0.5;
  const gap = a * (0.25 + 1.05 * unfold);
  const ox = cols / 2 - (NI - NJ) * a * 0.5 - 10, oy = rows / 2 - ((NI + NJ) * b - (SHEETS - 1) * gap) / 2;
  const at = (i: number, j: number, k: number): [number, number] => [ox + (i - j) * a, oy + (i + j) * b - k * gap];

  const head = ease((u - 0.2) / 0.62) * (SHEETS + 0.6) - 0.5; // the sheet the token has reached
  const lit = (k: number) => Array.from({ length: 3 }, (_, n) => [Math.floor(hash(7, k, n) * NI), Math.floor(hash(7, k, n, 1) * NJ), 0.55 + 0.45 * hash(7, k, n, 2)]);

  for (let k = 0; k < SHEETS; k++) {
    const near = Math.exp(-((head - k) ** 2) / 0.8);
    const corners = [at(0, 0, k), at(NI - 1, 0, k), at(NI - 1, NJ - 1, k), at(0, NJ - 1, k)];
    for (let c = 0; c < 4; c++) line(put, ...corners[c], ...corners[(c + 1) % 4], 0.16 + 0.32 * near, 2);
    for (let i = 0; i < NI; i++) for (let j = 0; j < NJ; j++) if ((i + j) % 2 === 0) put(...at(i, j, k), (0.07 + 0.16 * near) * (0.8 + 0.2 * Math.sin(t * 1.5 + i * 0.7 + j + k)));

    if (head > k) { // the logit lens: how sure this layer is of the next word
      const conf = 1 / (1 + Math.exp(-(k - SHEETS * 0.62) * 0.9)), [rx, ry] = at(NI - 1, NJ - 1, k);
      const n = Math.round(conf * 14 * clamp(head - k));
      for (let s = 0; s < n; s++) put(rx + 5 + s, ry, s === n - 1 ? 1 : 0.35 + 0.55 * conf);
    }
  }

  for (let k = 0; k < SHEETS && k <= head; k++) { // the token's path
    const here = lit(k), pulse = Math.exp(-((head - k) ** 2) / 0.6);
    for (const [i, j, act] of here) {
      const [x, y] = at(i, j, k);
      put(x, y, 0.6 + 0.4 * act);
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) put(x + dx, y + dy, 0.4 * act);
      if (pulse > 0.3) for (let d = -2; d <= 2; d++) (put(x + d, y + d, pulse * act), put(x + d, y - d, pulse * act));
    }
    if (k + 1 < SHEETS) {
      const up = lit(k + 1), grow = clamp(head - k);
      here.forEach(([i, j], n) => {
        const [x0, y0] = at(i, j, k), [x1, y1] = at(up[n][0], up[n][1], k + 1);
        line(put, x0, y0, x0 + (x1 - x0) * grow, y0 + (y1 - y0) * grow, 0.5);
      });
    }
  }
}

// ---------------------------------------------------------------- steer: a push parting the stream
const LINES = 23;
let push = { x: 0, y: 0, set: false };

export function steer({ put, cols, rows, t, u, pointer }: Frame) {
  const enter = ease(u / 0.55);
  const drift = { x: cols * (0.12 + 0.34 * enter + 0.04 * Math.sin(t * 0.31)), y: rows * (0.5 + 0.14 * Math.sin(t * 0.47 + 1)) };
  const aim = pointer ?? drift;
  if (!push.set) push = { ...aim, set: true };
  push.x += (aim.x - push.x) * 0.14;
  push.y += (aim.y - push.y) * 0.14;

  const A = rows * 0.13 * ease((u - 0.1) / 0.55), w = rows * 0.2, s = cols * 0.03;
  const bend = (x: number, y0: number) => { // most for the nearest line, never so much that two lines cross
    const d = y0 - push.y, side = Math.sign(d) || 1;
    return side * A * Math.exp(-Math.abs(d) / w) / (1 + Math.exp(-(x - push.x) / s));
  };

  for (let n = 0; n < LINES; n++) {
    const y0 = rows * (0.08 + (0.84 * n) / (LINES - 1)), near = Math.exp(-Math.abs(y0 - push.y) / (w * 0.7)) * enter;
    const speed = 9 + 7 * hash(n, 1);

    for (let x = 0; x < cols; x += 1) {
      if (x % 2 && near < 0.3) continue;
      put(x, y0 + bend(x, y0), 0.1 + 0.22 * near);
    }
    for (let p = 0; p < 4; p++) { // particles riding the line, each with a tail
      const hx = ((t * speed + hash(n, p, 2) * cols * 1.3) % (cols * 1.15)) - cols * 0.05;
      for (let q = 0; q < 9; q++) {
        const x = hx - q * 1.3;
        put(x, y0 + bend(x, y0), (0.35 + 0.6 * near) * (1 - q / 9) * (0.6 + 0.4 * hash(n, p)));
      }
    }
  }

  if (enter > 0.05) { // the push: a bright core in a ring, a second ring breathing out
    const { x: px, y: py } = push, breathe = (t * 0.9) % 1;
    for (let a = 0; a < Math.PI * 2; a += 0.06) {
      put(px + Math.cos(a) * 5.2, py + Math.sin(a) * 5.2, 0.5 * enter);
      put(px + Math.cos(a) * (6 + breathe * 9), py + Math.sin(a) * (6 + breathe * 9), 0.45 * (1 - breathe) * enter);
    }
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) put(px + dx, py + dy, (Math.hypot(dx, dy) < 1.5 ? 1 : 0.6) * enter);
  }
}

// ---------------------------------------------------------------- chat: two replies, side by side
export function chat({ put, cols, rows, t, u }: Frame) {
  const W = Math.min(cols - 8, Math.round(cols * 0.74), 180), x0 = Math.round((cols - W) / 2), y0 = Math.round(rows / 2 - 34);
  const box = (bx0: number, by0: number, bx1: number, by1: number, b: number) => { // a rounded outline in dots
    for (let x = bx0 + 2; x <= bx1 - 2; x += 2) (put(x, by0, b), put(x, by1, b));
    for (let y = by0 + 2; y <= by1 - 2; y += 2) (put(bx0, y, b), put(bx1, y, b));
    put(bx0 + 1, by0 + 1, b); put(bx1 - 1, by0 + 1, b); put(bx0 + 1, by1 - 1, b); put(bx1 - 1, by1 - 1, b);
  };
  const words = (wx0: number, y: number, wx1: number, b: number, seed: number, upto = Infinity) => { // text as runs of marks
    let x = wx0, n = 0;
    while (x < wx1 && n < upto) {
      const w = 2 + Math.floor(hash(seed, x) * 5);
      for (let k = 0; k < w && x + k < wx1 && n < upto; k++, n++) put(x + k, y, b);
      x += w + 1;
    }
    return { x, n };
  };

  // the question, top right
  const ask = ease(u / 0.15), qx0 = x0 + Math.round(W * 0.45), qy = y0;
  if (ask > 0) {
    box(qx0, qy, x0 + W, qy + 8, 0.4 * ask);
    words(qx0 + 3, qy + 4, qx0 + 3 + Math.round((x0 + W - qx0 - 6) * ask), 0.55 * ask, 3);
  }

  // the two replies, writing as the reader scrolls, each with its gauge beneath
  const colW = Math.floor((W - 4) / 2), typed = ease((u - 0.15) / 0.65);
  [0, 1].forEach((c) => {
    const bx0 = x0 + c * (colW + 4), bx1 = bx0 + colW, by0 = qy + 14, by1 = by0 + 46, bright = c ? 0.82 : 0.42;
    if (ask < 1) return;
    box(bx0, by0, bx1, by1, c ? 0.5 : 0.28);
    for (let k = 0; k < 6; k++) put(bx0 + 4 + k, by0 + 4, c ? 0.55 : 0.3); // its header

    let left = typed * 380 * (c ? 1 : 0.9), caret: [number, number] | null = null;
    for (let l = 0; l < 7 && left > 0; l++) {
      const y = by0 + 9 + l * 4, end = bx1 - 4 - (l === 6 ? Math.round(colW * 0.4) : 0);
      const r = words(bx0 + 4, y, end, bright, c * 31 + l, left);
      left -= r.n;
      caret = [Math.min(r.x, end), y];
    }
    if (caret && typed < 1 && Math.floor(t * 2.5) % 2 === 0) put(caret[0] + 1, caret[1], 1);

    // the gauge: a strip of cells, one per word so far, brighter the further toward the pole
    const gy = by1 + 5, cells = Math.floor((colW - 8) / 2), shown = Math.round(cells * typed);
    for (let k = 0; k < cells; k++) {
      const v = c ? 0.45 + 0.4 * hash(k, 5) * (k / cells) + 0.15 : 0.12 + 0.18 * hash(k, 6);
      put(bx0 + 4 + k * 2, gy, k < shown ? v : 0.08);
      put(bx0 + 4 + k * 2, gy + 1, k < shown ? v * 0.8 : 0.06);
    }
  });
}

// ---------------------------------------------------------------- api: requests in, tokens out
const CLIENTS = 7, TRIP = 2.6;

export function api({ put, cols, rows, t, u }: Frame) {
  const cx = cols / 2, cy = rows / 2, rx = Math.min(cols * 0.4, rows * 1.15), ry = rows * 0.38;
  const ring = ease(u / 0.3), arrived = ease((u - 0.2) / 0.5) * CLIENTS;
  const at = (c: number): [number, number] => { const a = -Math.PI / 2 + (c / CLIENTS) * Math.PI * 2 + 0.2; return [cx + Math.cos(a) * rx, cy + Math.sin(a) * ry]; };

  for (let a = 0; a < Math.PI * 2 * ring; a += 0.012) // the orbit, drawing round from the top
    if (Math.round(a / 0.012) % 3 === 0) put(cx + Math.cos(a - Math.PI / 2) * rx, cy + Math.sin(a - Math.PI / 2) * ry, 0.14);

  let pulse = 0;
  for (let c = 0; c < CLIENTS; c++) {
    const here = clamp(arrived - c);
    if (here <= 0) continue;
    const [x, y] = at(c), beat = TRIP * (1.6 + hash(c, 1) * 1.6), k = ((t + hash(c, 2) * beat) % beat) / TRIP;
    const busy = k < 1 && here >= 1;

    const edge = (busy ? 0.62 : 0.3) * here; // the client: a small window with lines of text in it
    for (let d = -7; d <= 7; d++) (put(x + d, y - 4, edge), put(x + d, y + 4, edge));
    for (let d = -3; d <= 3; d++) (put(x - 7, y + d, edge), put(x + 7, y + d, edge));
    for (let d = -6; d <= 6; d++) put(x + d, y - 2, 0.16 * here);
    for (let r = 0; r <= 2; r++) for (let d = -5; d <= 4 - r * 3; d++) if (hash(c, r, d) > 0.2) put(x + d, y + r, (busy ? 0.4 : 0.22) * here);

    if (!busy) continue;
    const mx = (x + cx) / 2 + (cy - y) * 0.25, my = (y + cy) / 2 + (x - cx) * 0.25 * (ry / rx);
    const along = (q: number): [number, number] => [(1 - q) ** 2 * x + 2 * (1 - q) * q * mx + q * q * cx, (1 - q) ** 2 * y + 2 * (1 - q) * q * my + q * q * cy];
    for (let q = 0; q <= 1; q += 0.02) put(...along(q), 0.1);
    if (k < 0.4) { // in: one bright packet with a short tail
      const q = k / 0.4;
      for (let s = 0; s < 6; s++) put(...along(Math.max(0, q - s * 0.025)), 1 - s * 0.15);
    } else { // out: tokens streaming back along the same path
      const q0 = (k - 0.4) / 0.6;
      pulse = Math.max(pulse, 1 - clamp((k - 0.4) * 4));
      for (let s = 0; s < 14; s++) {
        const q = 1 - clamp(q0 * 1.5 - s * 0.045);
        if (q <= 0 || q >= 1) continue;
        put(...along(q), s % 4 === 3 ? 0 : 0.8 - s * 0.03);
      }
    }
  }

  star(put, cx, cy, 0.62 + 0.08 * pulse + 0.15 * (1 - ring)); // the lodestar, with a ring out for each request
  if (pulse > 0) for (let a = 0; a < Math.PI * 2; a += 0.05) put(cx + Math.cos(a) * (13 + (1 - pulse) * 10), cy + Math.sin(a) * (13 + (1 - pulse) * 10), 0.6 * pulse);
}
