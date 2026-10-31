import { drawMarks, type Put } from "../../web/src/art/glyphs";

// Get Lodestar: a model coming down from the hub and assembling itself, layer by layer. Its weights fall from a
// ticked ring above and settle into a tower of layer plates turning slowly in perspective, bottom first. Once it
// stands, tokens climb its residual stream, each plate rippling with attention as one passes. Then it lifts away
// as dust, top first, and the next model comes down: another size, another depth. The pointer tilts the tower.
const CELL = 4;

export type Model = { name: string; layers: number; width: number }; // width: the plate's grid, its hidden size
const MODELS: Model[] = [
  { name: "Llama-3.1-8B", layers: 32, width: 11 },
  { name: "Qwen3-4B", layers: 36, width: 9 },
  { name: "Llama-3.2-1B", layers: 16, width: 7 },
  { name: "Mistral-7B", layers: 32, width: 10 },
  { name: "Qwen3-8B", layers: 36, width: 11 },
  { name: "Llama-3.2-3B", layers: 28, width: 8 },
];
const LOAD = 0.32, RUN = 7, LIFT = 2.4; // seconds per plate loading, running, lifting away

const hash = (...v: number[]) => { let h = 0; for (const x of v) h = Math.sin(h * 1.7 + x * 12.9898 + 78.233) * 43758.5453; return h - Math.floor(h); };
const clamp = (x: number) => Math.max(0, Math.min(1, x));
const ease = (x: number) => { x = clamp(x); return x * x * (3 - 2 * x); };
type V = [number, number, number];

/** status: called with the model on the tower and how many of its layers are in, whenever either changes. */
export function startAssembly(canvas: HTMLCanvasElement, opts: { status?: (m: Model, loaded: number) => void } = {}) {
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
    if (x < 0 || y < 0 || x >= cols || y >= rows || b <= 0.02) return;
    const i = y * cols + x;
    if (b > grid[i]) grid[i] = b;
  };

  let px = 0, tx = 0, py = 0, ty = 0;
  addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    tx = clamp((e.clientX - r.left) / r.width) * 2 - 1;
    ty = clamp((e.clientY - r.top) / r.height) * 2 - 1;
  }, { passive: true });

  // where each model's turn starts, so the cycle can be read from the clock
  const span = (m: Model) => plates(m) * LOAD + 0.8 + RUN + LIFT;
  const plates = (m: Model) => Math.round(m.layers / (rows < 110 ? 4 : 3)); // fewer, on a small canvas
  let shown = "";

  const draw = (t: number) => {
    grid.fill(0);
    let c = t % MODELS.reduce((s, m) => s + span(m), 0), mi = 0;
    while (c > span(MODELS[mi])) c -= span(MODELS[mi++]);
    const m = MODELS[mi], N = plates(m), g = m.width - 3, loadT = N * LOAD + 0.8;

    px += (tx - px) * 0.05;
    py += (ty - py) * 0.05;
    const yaw = t * 0.16 + px * 0.5, cy_ = Math.cos(yaw), sy = Math.sin(yaw);
    const pitch = 0.36 + py * 0.1, cp = Math.cos(pitch), sp = Math.sin(pitch);
    const S = Math.min(rows * 0.37, cols * 0.5), cx = cols / 2, cy = rows * 0.55;
    const project = (v: V): [number, number, number] => {
      const x2 = v[0] * cy_ + v[2] * sy, z2 = -v[0] * sy + v[2] * cy_;
      const y3 = v[1] * cp - z2 * sp, z3 = v[1] * sp + z2 * cp;
      const k = 4 / (4 - z3);
      return [cx + x2 * S * k, cy - y3 * S * k, clamp((z2 + 1) / 2)];
    };
    const shade = (d: number) => 0.3 + 0.7 * d ** 1.3;
    const dotAt = (v: V, b: number) => { const [x, y, d] = project(v); put(x, y, b * shade(d)); };
    const line = (a: V, c: V, b: number, every = 1) => {
      const [x1, y1] = project(a), [x2, y2] = project(c), n = Math.ceil(Math.hypot(x2 - x1, y2 - y1));
      for (let j = 0; j <= n; j += every) {
        const u = j / Math.max(n, 1);
        dotAt([a[0] + (c[0] - a[0]) * u, a[1] + (c[1] - a[1]) * u, a[2] + (c[2] - a[2]) * u], b);
      }
    };

    // the tower's shape: N plates from y = -1 to 1, half-width h, a g×g lattice on each
    const h = 0.22 + g * 0.05, Y = (j: number) => -1 + (2 * j) / Math.max(1, N - 1) * 0.92;
    const node = (i: number, j: number): V => {
      const a = i % g, b = Math.floor(i / g);
      return [(-1 + (2 * a) / (g - 1)) * h, Y(j), (-1 + (2 * b) / (g - 1)) * h];
    };
    const source: V = [0, 1.22, 0];

    // phase: loading (plates arrive), running (tokens climb), lifting (plates leave, top first)
    const loaded = c < loadT ? Math.min(N, c / LOAD) : N;
    const running = c >= loadT && c < loadT + RUN ? (c - loadT) / RUN : c >= loadT + RUN ? 1 : 0;
    const lift = c >= loadT + RUN ? (c - loadT - RUN) / LIFT : 0;
    const status = `${mi}:${Math.min(m.layers, Math.ceil((loaded / N) * m.layers))}`;
    if (status !== shown) ((shown = status), opts.status?.(m, Math.min(m.layers, Math.ceil((loaded / N) * m.layers))));

    // the ring the weights come down from, turning against the tower
    const ringR = h * 0.75;
    for (let deg = 0; deg < 360; deg += 4) {
      const a = (deg * Math.PI) / 180 - t * 0.4, tick = deg % 24 === 0;
      const v: V = [Math.cos(a) * ringR, source[1], Math.sin(a) * ringR];
      dotAt(v, (lift > 0 ? 0.3 : 0.55) * (tick ? 1.3 : 0.7));
      if (tick) dotAt([v[0] * 1.12, v[1], v[2] * 1.12], 0.5);
    }

    const gone = (j: number) => ease(lift * 1.6 - ((N - 1 - j) / N) * 0.6); // top first

    // tokens up the residual stream, while it runs
    const pulses = [0, 1, 2].map((k) => (running > 0 && running < 1 ? ((c - loadT) / 2.3 + k / 3) % 1 : -9));
    const pulseAt = (j: number) => Math.max(...pulses.map((p) => Math.exp(-(((p * (N + 3) - 1.5) - j) ** 2) / 1.1)));

    for (let j = 0; j < N; j++) {
      const arrive = clamp(loaded - j); // 0..1, how far this plate's weights are in
      if (arrive <= 0) continue;
      const leave = gone(j);
      const e = running > 0 ? pulseAt(j) : 0;
      const fade = 1 - leave;

      // the lattice: each weight falls from the ring to its place
      for (let i = 0; i < g * g; i++) {
        const home = node(i, j), at = hash(i, j, mi) * 0.65, s = ease((arrive - at) / 0.35);
        if (s <= 0) continue;
        let v: V;
        if (s < 1) {
          const a = hash(i, j, 7) * Math.PI * 2, from: V = [Math.cos(a) * ringR, source[1], Math.sin(a) * ringR];
          v = [from[0] + (home[0] - from[0]) * s, from[1] + (home[1] - from[1]) * s ** 1.6, from[2] + (home[2] - from[2]) * s];
          dotAt(v, 0.85);
          for (let k = 1; k <= 3; k++) { const s2 = Math.max(0, s - k * 0.05); dotAt([from[0] + (home[0] - from[0]) * s2, from[1] + (home[1] - from[1]) * s2 ** 1.6, from[2] + (home[2] - from[2]) * s2], 0.5 - k * 0.12); }
          continue;
        }
        v = home;
        if (leave > 0) {
          const dz = leave * (0.5 + hash(i, j, 3));
          v = [home[0] * (1 + leave * 0.6) + (hash(i, j, 4) - 0.5) * leave, home[1] + dz, home[2] * (1 + leave * 0.6) + (hash(i, j, 5) - 0.5) * leave];
        }
        const rd = Math.hypot(home[0], home[2]) / h;
        const ripple = e * (0.5 + 0.5 * Math.cos(rd * 7 - t * 6));
        const settle = Math.exp(-(arrive - at - 0.35) * 6) * (arrive < 1 ? 1 : 0);
        const flick = hash(i, j, mi, 9) ** 5 * 0.3 * (0.5 + 0.5 * Math.sin(t * 2 + i));
        dotAt(v, (0.32 + ripple * 0.7 + settle * 0.6 + flick) * fade);
      }

      // its frame, drawn in as the plate fills
      if (leave < 1) {
        const y = Y(j), cs: V[] = [[-h, y, -h], [h, y, -h], [h, y, h], [-h, y, h]].map(([a, b, c2]) => [a * 1.12, b, c2 * 1.12] as V);
        const drawn = ease(arrive * 1.2);
        for (let k = 0; k < 4; k++) {
          const a = cs[k], b = cs[(k + 1) % 4], u = drawn;
          line(a, [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u], (0.36 + e * 0.5) * fade, 1);
        }
        // and its weight matrix, faint, once it is in
        if (arrive >= 1)
          for (let k = 0; k < g; k++) {
            line(node(k, j), node(k + g * (g - 1), j), (0.1 + e * 0.2) * fade, 2);
            line(node(k * g, j), node(k * g + g - 1, j), (0.1 + e * 0.2) * fade, 2);
          }
      }

      // attention: arcs between a plate's weights as a token passes
      if (e > 0.3 && leave === 0)
        for (let k = 0; k < 3; k++) {
          const slot = Math.floor((c - loadT) / 2.3 + j * 0.37);
          const a = node(Math.floor(hash(slot, j, k, 1) * g * g), j), b = node(Math.floor(hash(slot, j, k, 2) * g * g), j);
          const n = 28;
          for (let q = 0; q <= n; q++) {
            const u = q / n, lift2 = Math.sin(Math.PI * u) * 0.18;
            dotAt([a[0] + (b[0] - a[0]) * u, a[1] + lift2, a[2] + (b[2] - a[2]) * u], (0.3 + 0.5 * e) * (q % 2 ? 0.7 : 1));
          }
        }
    }

    let top = Math.min(N - 1, Math.floor(loaded));
    while (top >= 0 && gone(top) > 0.5) top--;

    // each token carries a ticked ring up round the tower, like a gimbal riding it
    for (const p of pulses) {
      const jf = p * (N + 3) - 1.5;
      if (jf < -0.5 || jf > N - 0.5 || top < 0) continue;
      const y = Y(Math.max(0, Math.min(N - 1, jf))), R = h * 1.5, b = clamp(Math.min(jf + 0.5, N - 0.5 - jf));
      for (let deg = 0; deg < 360; deg += 2) {
        const a = (deg * Math.PI) / 180 + p * 3;
        dotAt([Math.cos(a) * R, y, Math.sin(a) * R], (deg % 30 === 0 ? 0.95 : 0.5) * b);
        if (deg % 30 === 0) dotAt([Math.cos(a) * R * 1.07, y, Math.sin(a) * R * 1.07], 0.75 * b);
      }
    }

    // the residual stream: a spine through the plates that are in, carrying the tokens
    if (loaded >= 1 && top >= 0) {
      const y0 = Y(0), y1 = Y(top);
      const [, sy0] = project([0, y0, 0]), [, sy1] = project([0, y1, 0]), n = Math.ceil(Math.abs(sy0 - sy1));
      for (let q = 0; q <= n; q++) {
        const y = y0 + (y1 - y0) * (q / Math.max(n, 1)), jf = ((y + 1) / 1.84) * (N - 1);
        dotAt([0, y, 0], 0.32 + pulseAt(jf) * 0.8);
      }
      // corner posts, faint
      for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) line([a * h * 1.12, y0, b * h * 1.12], [a * h * 1.12, y1, b * h * 1.12], 0.1, 3);
    }

    ctx.fillStyle = getComputedStyle(canvas).color;
    drawMarks(ctx, grid, cols, rows, CELL);
  };

  // its own clock, which only runs while it can be seen, so a reader arriving finds the first model coming down
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let raf = 0, last = 0, seen = false, clock = 0;
  const tick = (now: number) => {
    raf = 0;
    if (!seen || document.hidden) return (last = 0);
    raf = requestAnimationFrame(tick);
    if (last && now - last < 40) return; // ~25 fps
    clock += last ? Math.min(0.1, (now - last) / 1000) : 0;
    last = now;
    draw(clock);
  };
  const run = () => (still ? draw(span(MODELS[0]) - LIFT - 2) : raf || (raf = requestAnimationFrame(tick)));

  size();
  new IntersectionObserver(([e]) => ((seen = e.isIntersecting), seen && run())).observe(canvas);
  document.addEventListener("visibilitychange", () => !document.hidden && run());
  let rt: ReturnType<typeof setTimeout> | undefined;
  addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => size() && (still ? run() : draw(clock)), 150); });
}
