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

/** The sidebar's night as a navigator's chart, where the start screen's scene is the photograph: a faint sky
 * atlas grid sweeping in from the pole, and scattered stars that twinkle. Nothing turns; ~4 fps for the twinkle,
 * paused while hidden. */
export function startSideArt(canvas) {
  const cell = 4;
  let cols, rows, grid, ctx, navRow, pole;
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
    const tabs = document.querySelector(".tabs")?.getBoundingClientRect(), top = canvas.getBoundingClientRect().top;
    navRow = tabs ? (tabs.bottom - top) / cell : 0;
    pole = { x: cols * 1.25, y: -rows * 0.08 }; // the chart's pole, off past the top right, so its grid sweeps in
    return true;
  };
  const put = (x, y, b) => {
    if (x < 0 || y < 0 || x >= cols || y >= rows) return;
    b *= Math.min(1, Math.max(0.15, (y - navRow) / 16)); // quiet behind the brand and tabs
    const i = y * cols + x;
    if (b > grid[i]) grid[i] = b;
  };
  const draw = (t) => {
    grid.fill(0);
    // the atlas grid: circles of declination and meridians, dotted
    for (let r = 30; r < cols * 3; r += 26) {
      const step = 3 / r;
      for (let a = Math.PI * 0.5; a < Math.PI * 1.1; a += step) put(Math.round(pole.x + Math.cos(a) * r), Math.round(pole.y + Math.sin(a) * r), 0.14);
    }
    for (let k = 0; k < 9; k++) {
      const a = Math.PI * (0.55 + k * 0.06);
      for (let r = 20; r < cols * 3; r += 4) put(Math.round(pole.x + Math.cos(a) * r), Math.round(pole.y + Math.sin(a) * r), 0.11);
    }
    // scattered stars, each twinkling on its own beat
    for (let i = 0; i < 70; i++) {
      const x = Math.floor(hash(i, 11) * cols), y = Math.floor(hash(i, 12) * rows);
      put(x, y, (0.12 + hash(i, 13) ** 3 * 0.45) * (0.65 + 0.35 * Math.sin(t * (0.6 + hash(i, 14)) + i)));
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
    if (now - last < 250) return; // a twinkle needs only a few frames a second
    last = now;
    draw((now - t0) / 1000);
  };
  const run = () => (still ? draw(0) : raf || (raf = requestAnimationFrame(tick)));
  if (size()) run();
  let resizeT = 0;
  addEventListener("resize", () => { clearTimeout(resizeT); resizeT = setTimeout(() => size() && (still ? draw(0) : null), 200); });
  document.addEventListener("visibilitychange", () => !document.hidden && run());
}
