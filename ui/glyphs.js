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

/** The sidebar's night: scattered stars over engraved hills, all drawn in marks. Static; redrawn on resize. */
export function drawSideArt(canvas) {
  const cell = 4, w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  const dpr = devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = getComputedStyle(canvas).color;
  const cols = Math.ceil(w / cell), rows = Math.ceil(h / cell);
  const grid = new Float32Array(cols * rows);
  const base = rows - 62; // where the hills start: they rise behind and above the model card
  const ridge = (x, k) => base + 9 - 7 * Math.exp(-((x - cols * 0.7) ** 2) / 140) - 4 * Math.exp(-((x - cols * 0.22) ** 2) / 90)
    + k * 2.6 + Math.sin(x * 0.21 + k) * 0.5;
  for (let y = 0; y < rows; y++)
    for (let x = 0; x < cols; x++) {
      const top = ridge(x, 0);
      let b = 0;
      if (y >= top) {
        // contour lines that follow the ridge, closer and brighter toward the front, broken like a cut line
        const v = y - top, k = Math.floor(v / 2.6);
        const onLine = Math.abs(y - ridge(x, k)) < 0.55 && Math.sin(x * 0.9 + k * 2.3) < 0.82;
        b = onLine ? 0.35 + Math.min(0.5, k * 0.06) : (v < 1 ? 0.6 : 0);
      } else {
        // stars, thinning out toward the hills
        const r = hash(x, y);
        if (r < 0.011 * (1 - y / rows * 0.6)) b = r < 0.0012 ? 0.7 : 0.08 + r * 18;
      }
      grid[y * cols + x] = b;
    }
  ctx.globalAlpha = 1;
  drawMarks(ctx, grid, cols, rows, cell);
}
