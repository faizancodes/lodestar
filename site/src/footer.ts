import { drawMarks, type Put } from "../../web/src/art/glyphs";
import { compassStar } from "../../web/src/art/mark";

// The foot of the page: the lodestar rising over the horizon at the centre of a great astrolabe, its rings turning
// at their own rates, meridians fanning out from it, and the sky round it scattered with stars that slowly wheel
// about it, as the sky does about the pole star, a few joined into constellations.
const CELL = 4;
const STARS = 420;

const hash = (...v: number[]) => { let h = 0; for (const x of v) h = Math.sin(h * 1.7 + x * 12.9898 + 78.233) * 43758.5453; return h - Math.floor(h); };
const clamp = (x: number) => Math.max(0, Math.min(1, x));

// "Lodestar" in the page's face, as coverage per cell, filling the band below the horizon (which sits at 0.58 × 0.78)
function spell(cols: number, rows: number) {
  const c = document.createElement("canvas"), g = c.getContext("2d", { willReadFrequently: true })!;
  c.width = cols;
  c.height = rows;
  const family = getComputedStyle(document.body).fontFamily, top = rows * 0.58 * 0.78 + rows * 0.09, room = rows - top - rows * 0.06;
  g.font = `600 100px ${family}`;
  const m = g.measureText("Lodestar"), cap = m.actualBoundingBoxAscent;
  const px = Math.min((cols * 0.88) / m.width, room / cap) * 100;
  g.font = `600 ${px}px ${family}`;
  g.letterSpacing = `${-px * 0.03}px`;
  g.textAlign = "center";
  g.textBaseline = "alphabetic";
  g.fillStyle = "#fff";
  const h = g.measureText("Lodestar").actualBoundingBoxAscent;
  g.fillText("Lodestar", cols / 2, top + (room - h) * 0.4 + h); // centred in its band, a little high
  const d = g.getImageData(0, 0, cols, rows).data, out = new Float32Array(cols * rows);
  for (let i = 0; i < out.length; i++) out[i] = d[i * 4 + 3] / 255;
  return out;
}

export function startFooter(canvas: HTMLCanvasElement) {
  let cols = 0, rowsAll = 0, grid = new Float32Array(0), name = new Float32Array(0), ctx!: CanvasRenderingContext2D;

  const size = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight, dpr = devicePixelRatio || 1;
    if (!w || !h) return false;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cols = Math.ceil(w / CELL);
    rowsAll = Math.ceil(h / CELL);
    grid = new Float32Array(cols * rowsAll);
    name = spell(cols, rowsAll);
    return true;
  };

  const put: Put = (x, y, b) => {
    x = Math.round(x);
    y = Math.round(y);
    if (x < 0 || y < 0 || x >= cols || y >= rowsAll || b <= 0.02) return;
    const i = y * cols + x;
    if (b > grid[i]) grid[i] = b;
  };

  const draw = (t: number) => {
    grid.fill(0);
    const rows = Math.round(rowsAll * 0.58); // the sky's share; the name stands below it
    const cx = cols / 2, cy = rows * 0.78, R = Math.max(cols * 0.55, rows);
    // things fade toward the page's edges and below the horizon
    const fall = (x: number, y: number) => clamp(1.25 - Math.hypot((x - cx) / (cols * 0.55), (y - cy) / (rows * 0.95))) * (y > cy ? clamp(1 - (y - cy) / (rows * 0.2)) : 1);
    const at = (x: number, y: number, b: number) => put(x, y, b * fall(x, y));

    // the sky, wheeling slowly about the star
    const sky: [number, number][] = [];
    const n = Math.min(STARS, Math.round(cols * 1.6)); // sparser on a narrow screen
    for (let i = 0; i < n; i++) {
      const r = R * (0.12 + 0.95 * Math.sqrt(hash(i, 1))), a = hash(i, 2) * Math.PI * 2 - t * 0.012;
      const x = cx + Math.cos(a) * r * 1.4, y = cy - Math.abs(Math.sin(a)) * r * 0.9 - 2;
      sky.push([x, y]);
      const tw = 0.5 + 0.5 * Math.sin(t * (0.6 + hash(i, 3) * 1.8) + hash(i, 4) * 6.3);
      at(x, y, (0.12 + hash(i, 5) ** 4 * 0.6) * (0.5 + 0.5 * tw));
    }
    // a few constellations among them
    for (let k = 0; k < 7; k++) {
      let p = Math.floor(hash(k, 9) * n);
      for (let s = 0; s < 4; s++) {
        let best = -1, bd = Infinity;
        for (let q = 0; q < n; q += 3) {
          const d = Math.hypot(sky[q][0] - sky[p][0], sky[q][1] - sky[p][1]);
          if (q !== p && d > 8 && d < bd) ((bd = d), (best = q));
        }
        if (best < 0 || bd > 34) break;
        const [x1, y1] = sky[p], [x2, y2] = sky[best];
        for (let j = 0; j <= bd; j += 2) at(x1 + ((x2 - x1) * j) / bd, y1 + ((y2 - y1) * j) / bd, 0.2);
        at(x2, y2, 0.75);
        p = best;
      }
    }

    // meridians, fanning out from the star across the sky
    for (let k = 0; k <= 24; k++) {
      const a = Math.PI + (k / 24) * Math.PI, major = k % 6 === 0;
      for (let r = 20; r < R * 1.3; r += major ? 1.5 : 3) at(cx + Math.cos(a) * r * 1.4, cy + Math.sin(a) * r * 0.9, major ? 0.24 : 0.12);
    }

    // the astrolabe's rings, each ticked and turning at its own rate
    const rings = [0.3, 0.48, 0.7, 0.95];
    rings.forEach((f, n) => {
      const r = R * f, spin = t * 0.05 * (n % 2 ? -1 : 1) / (n + 1);
      for (let deg = 180; deg <= 360; deg += 0.6) {
        const a = (deg * Math.PI) / 180, x = cx + Math.cos(a) * r * 1.4, y = cy + Math.sin(a) * r * 0.9;
        at(x, y, 0.3);
        const tick = ((deg - spin * 57.3) % (n % 2 ? 5 : 10) + 360) % (n % 2 ? 5 : 10);
        if (tick < 0.6) for (let j = 1; j <= (n % 2 ? 2 : 4); j++) at(cx + Math.cos(a) * (r + j) * 1.4, cy + Math.sin(a) * (r + j) * 0.9, 0.5);
      }
      // a bead riding each ring
      const b = Math.PI + ((t * 0.07 * (n + 1) + n * 0.37) % 1) * Math.PI;
      for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) if (Math.abs(i) + Math.abs(j) <= 2) at(cx + Math.cos(b) * r * 1.4 + i, cy + Math.sin(b) * r * 0.9 + j, 0.9 - (Math.abs(i) + Math.abs(j)) * 0.2);
    });

    // the name, standing in the ground below the horizon, a slow light passing along it
    for (let y = 0; y < rowsAll; y++)
      for (let x = 0; x < cols; x++) {
        const a = name[y * cols + x];
        if (a < 0.3) continue;
        const sweep = Math.exp(-(((x - ((t * 0.08) % 1.6 - 0.3) * cols) / (cols * 0.1)) ** 2));
        put(x, y, a * (0.34 + 0.16 * hash(x, y) + sweep * 0.5) * (1 - 0.4 * clamp((y - cy) / (rowsAll - cy))));
      }

    // the horizon
    for (let x = 0; x < cols; x += 1) at(x, cy, 0.3 + 0.25 * Math.exp(-(((x - cx) / (cols * 0.12)) ** 2)));

    // and the lodestar, on it
    const star = Math.min(rowsAll * 0.58 * 0.3, cols * 0.12, 36), S = star / 15, pulse = 0.85 + 0.15 * Math.sin(t * 1.1);
    for (let dy = -Math.ceil(star); dy <= star; dy++)
      for (let dx = -Math.ceil(star); dx <= star; dx++) {
        const b = compassStar(dx / S, dy / S);
        if (b >= 0.06) put(cx + dx, cy - star * 0.35 + dy, b * pulse * (0.8 + 0.2 * Math.sin(Math.hypot(dx, dy) / S * 0.9 - t * 3.2)));
      }

    ctx.fillStyle = getComputedStyle(canvas).color;
    drawMarks(ctx, grid, cols, rowsAll, CELL);
  };

  const still = matchMedia("(prefers-reduced-motion: reduce)").matches, t0 = performance.now();
  let raf = 0, last = 0, seen = false;
  const tick = (now: number) => {
    raf = 0;
    if (!seen || document.hidden) return;
    raf = requestAnimationFrame(tick);
    if (now - last < 40) return; // ~25 fps
    last = now;
    draw((now - t0) / 1000);
  };
  const run = () => (still ? draw(4) : raf || (raf = requestAnimationFrame(tick)));

  size();
  document.fonts.ready.then(() => size() && draw((performance.now() - t0) / 1000));
  new IntersectionObserver(([e]) => ((seen = e.isIntersecting), seen && run())).observe(canvas);
  document.addEventListener("visibilitychange", () => !document.hidden && run());
  let rt: ReturnType<typeof setTimeout> | undefined;
  addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => size() && draw((performance.now() - t0) / 1000), 150); });
}
