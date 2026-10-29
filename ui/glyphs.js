// Every ASCII scene draws through here: one mark per grid cell, picked by brightness, so a scene reads like a
// figure set in type: a dot where it's faint, then a plus, an ×, and a solid block where it's brightest.
const DOT = 0.3, PLUS = 0.55, CROSS = 0.8;

/** Draws `grid` (cols × rows of brightness 0..1) as marks in the canvas's current fill colour. */
export function drawMarks(ctx, grid, cols, rows, cell) {
  ctx.clearRect(0, 0, cols * cell, rows * cell);
  const plus = new Path2D(), cross = new Path2D();
  const arm = cell * 0.4, d = cell * 0.3;
  for (let y = 0; y < rows; y++)
    for (let x = 0; x < cols; x++) {
      const b = Math.min(1, grid[y * cols + x]);
      if (b < 0.05) continue;
      const cx = x * cell + cell / 2, cy = y * cell + cell / 2;
      if (b < DOT) {
        const s = cell * (0.16 + 0.55 * b);
        ctx.fillRect(cx - s / 2, cy - s / 2, s, s);
      } else if (b < PLUS) {
        plus.moveTo(cx - arm, cy); plus.lineTo(cx + arm, cy);
        plus.moveTo(cx, cy - arm); plus.lineTo(cx, cy + arm);
      } else if (b < CROSS) {
        cross.moveTo(cx - d, cy - d); cross.lineTo(cx + d, cy + d);
        cross.moveTo(cx - d, cy + d); cross.lineTo(cx + d, cy - d);
      } else {
        const s = cell * 0.84;
        ctx.fillRect(cx - s / 2, cy - s / 2, s, s);
      }
    }
  ctx.strokeStyle = ctx.fillStyle;
  ctx.lineCap = "square";
  ctx.lineWidth = Math.max(0.6, cell * 0.2);
  ctx.globalAlpha = 0.75;
  ctx.stroke(plus);
  ctx.globalAlpha = 1;
  ctx.lineWidth = Math.max(0.7, cell * 0.24);
  ctx.stroke(cross);
}

const hash = (x, y) => { const s = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453; return s - Math.floor(s); };

/** The sidebar's night, as a long exposure: star trails wheeling around the lodestar, the one star that holds still,
 * over a dark mountain range with a lit ridge. Turns slowly (~6 fps, paused while hidden); still for reduced motion. */
export function startSideArt(canvas) {
  const cell = 4, TURN = (2 * Math.PI) / 480; // one full turn of the sky every eight minutes
  let cols, rows, grid, ctx, stars, front, back, pole, navRow;
  const size = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight, dpr = devicePixelRatio || 1;
    if (!w || !h) return false;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cols = Math.ceil(w / cell);
    rows = Math.ceil(h / cell);
    grid = new Float32Array(cols * rows);
    pole = { x: Math.round(cols * 0.68), y: Math.round(rows * 0.42) };
    const tabs = document.querySelector(".tabs")?.getBoundingClientRect(), top = canvas.getBoundingClientRect().top;
    navRow = tabs ? (tabs.bottom - top) / cell : 0; // trails fade out behind the brand and the tabs
    const base = rows - 38; // the ranges rise behind the model card and just above it
    front = Array.from({ length: cols }, (_, x) => Math.round(base - 11 * Math.exp(-((x - cols * 0.72) ** 2) / 110)
      - 6 * Math.exp(-((x - cols * 0.18) ** 2) / 70) + 1.3 * Math.sin(x * 0.45)));
    back = Array.from({ length: cols }, (_, x) => Math.round(base - 15 + 5 * Math.sin(x * 0.085 + 1.2) + 3 * Math.sin(x * 0.27)));
    const far = Math.hypot(Math.max(pole.x, cols - pole.x), Math.max(pole.y, rows - pole.y));
    stars = Array.from({ length: 170 }, (_, i) => ({
      r: 4 + hash(i, 1) ** 1.25 * far,
      a: hash(i, 2) * Math.PI * 2,
      len: 0.45 + hash(i, 3) * 0.9, // radians of trail behind the head
      head: 0.22 + hash(i, 4) ** 2.2 * 0.78, // most faint, a few bright
    }));
    return true;
  };
  const put = (x, y, b) => {
    if (x < 0 || y < 0 || x >= cols || y >= rows || y >= Math.min(front[x], back[x])) return; // behind the mountains
    const i = y * cols + x;
    if (b > grid[i]) grid[i] = b;
  };
  const draw = (t) => {
    grid.fill(0);
    for (const s of stars) { // each trail brightens from a faint tail to its head
      const head = s.a + t * TURN, step = 0.8 / s.r;
      for (let u = 0; u <= s.len; u += step) {
        const k = 1 - u / s.len;
        const y = Math.round(pole.y + Math.sin(head - u) * s.r * 0.92);
        put(Math.round(pole.x + Math.cos(head - u) * s.r), y, s.head * k ** 1.6 * Math.min(1, Math.max(0.1, (y - navRow) / 16)));
      }
    }
    // the lodestar: a bright core, its north ray longest
    put(pole.x, pole.y, 1);
    for (const [dx, dy, b] of [[1, 0, 0.7], [-1, 0, 0.7], [0, 1, 0.7], [0, -1, 0.75], [0, -2, 0.45], [0, -3, 0.25], [2, 0, 0.3], [-2, 0, 0.3], [0, 2, 0.3],
      [1, 1, 0.2], [1, -1, 0.2], [-1, 1, 0.2], [-1, -1, 0.2]]) put(pole.x + dx, pole.y + dy, b);
    for (let x = 0; x < cols; x++) { // two ranges: a dim far ridge, and a near one lit along its edge
      const f = front[x], bk = back[x];
      for (let y = Math.max(0, bk); y < rows; y++) {
        let b = 0;
        if (y === bk && bk < f) b = 0.36;
        else if (y === f) b = hash(x, 9) < 0.18 ? 0.95 : 0.7;
        else if (y > f) b = (x + y) % 3 === 0 ? Math.max(0, 0.14 - (y - f) * 0.004) : 0;
        else if (y > bk) b = (x + 2 * y) % 5 === 0 ? 0.08 : 0;
        if (b) grid[y * cols + x] = b;
      }
    }
    ctx.fillStyle = getComputedStyle(canvas).color;
    drawMarks(ctx, grid, cols, rows, cell);
  };
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches, t0 = performance.now();
  let last = 0, raf = 0;
  const tick = (now) => {
    raf = 0;
    if (document.hidden) return;
    raf = requestAnimationFrame(tick);
    if (now - last < 160) return; // ~6 fps is plenty for a sky this slow
    last = now;
    draw((now - t0) / 1000 + 40);
  };
  const run = () => (still ? draw(40) : raf || (raf = requestAnimationFrame(tick)));
  if (size()) run();
  let resizeT = 0;
  addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => size() && (still ? draw(40) : null), 200); });
  document.addEventListener("visibilitychange", () => !document.hidden && run());
}
