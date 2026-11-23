import { drawMarks, type Put } from "../../web/src/art/glyphs";
import { startAssembly } from "./assemble";
import { startFooter } from "./footer";
import { startLodestar } from "./lodestar";
import { api, chat, erase, layers, steer, type Frame } from "./scenes";

// capture.html's script: puts one animation on the canvas. The animated ones run on the page's stepped clock; a
// feature scene is drawn by __frame(t, u) instead, u being how far through its story it is (on the site, the scroll).
const q = new URLSearchParams(location.search), canvas = document.querySelector<HTMLCanvasElement>("#art")!;
const parseDim = (key: string, fallback: number) => {
  const n = Number(q.get(key));
  return Number.isFinite(n) && n > 0 ? Math.round(n) : fallback;
};
canvas.style.width = `${parseDim("w", 800)}px`;
canvas.style.height = `${parseDim("h", 420)}px`;
const scene = q.get("scene") ?? "hero";
const SCENES: Record<string, (f: Frame) => void> = { erase, layers, steer, chat, api };

if (scene === "hero") startLodestar(canvas);
else if (scene === "get") startAssembly(canvas);
else if (scene === "foot") startFooter(canvas);
else if (scene in SCENES) {
  const CELL = 4, dpr = devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d")!, cols = Math.ceil(w / CELL), rows = Math.ceil(h / CELL), grid = new Float32Array(cols * rows);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const color = getComputedStyle(canvas).color;
  const drawScene = SCENES[scene];
  const put: Put = (x, y, b) => {
    x = Math.round(x);
    y = Math.round(y);
    if (x < 0 || y < 0 || x >= cols || y >= rows || b <= 0.02) return;
    const i = y * cols + x;
    if (b > grid[i]) grid[i] = b;
  };
  (window as unknown as { __frame: (t: number, u: number) => void }).__frame = (t, u) => {
    grid.fill(0);
    drawScene({ put, cols, rows, t, u, pointer: null });
    ctx.fillStyle = color;
    drawMarks(ctx, grid, cols, rows, CELL);
  };
} else {
  throw new Error(`Unknown capture scene: ${scene}`);
}
