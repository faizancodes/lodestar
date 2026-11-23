import { createDecoder } from "../../web/src/art/decode";
import { drawMarks, type Put } from "../../web/src/art/glyphs";
import type { Frame } from "./scenes";
import { clamp } from "./util";

// The features as one scroll sequence: a stage pinned for the length of the section, one canvas on it, and a step
// per feature. Scrolling moves a progress p through the steps; each scene plays its story as its step's u runs
// 0 → 1, and at the seam the next scene comes up through the last one's marks. The step's words change with it,
// its title decoding in mark by mark as in the app's replies.
const CELL = 4, SEAM = 0.16; // the last part of a step, where it hands over to the next

export function startSequence(root: HTMLElement, scenes: ((f: Frame) => void)[]) {
  const canvas = root.querySelector<HTMLCanvasElement>(".seq-art")!;
  const steps = [...root.querySelectorAll<HTMLElement>(".step")];
  const ticks = [...root.querySelectorAll<HTMLElement>(".rail a")];
  const meter = root.querySelector<HTMLElement>(".rail-fill");
  const N = scenes.length;

  let cols = 0, rows = 0, grid = new Float32Array(0), ctx: CanvasRenderingContext2D | null = null, color = "";
  let pointer: { x: number; y: number } | null = null, weight = 1;

  const size = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight, dpr = devicePixelRatio || 1;
    if (!w || !h) return false;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx = canvas.getContext("2d");
    if (!ctx) return false;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    color = getComputedStyle(canvas).color;
    cols = Math.ceil(w / CELL);
    rows = Math.ceil(h / CELL);
    grid = new Float32Array(cols * rows);
    return true;
  };

  const put: Put = (x, y, b) => {
    x = Math.round(x);
    y = Math.round(y);
    b *= weight;
    if (x < 0 || y < 0 || x >= cols || y >= rows || b <= 0.02) return;
    const i = y * cols + x;
    if (b > grid[i]) grid[i] = b;
  };

  // how far through the section the reader is, in steps
  const target = () => {
    const r = root.getBoundingClientRect(), run = r.height - innerHeight;
    return clamp((-r.top / Math.max(run, 1)) * N, 0, N - 0.0001);
  };

  let shown = -1;
  const activate = (i: number) => {
    if (i === shown) return;
    shown = i;
    steps.forEach((s, k) => s.classList.toggle("on", k === i));
    ticks.forEach((a, k) => a.classList.toggle("on", k === i));
    const title = steps[i].querySelector<HTMLElement>("h2")!;
    title.textContent = title.dataset.text ??= title.textContent!;
    createDecoder().update(title);
  };

  let p = target(), lastT = 0;
  const frame = (t: number, snap = false) => {
    if (!ctx) return;
    const goal = target(), dt = Math.min(0.25, Math.max(0, t - lastT));
    lastT = t;
    p = snap ? goal : p + (goal - p) * (1 - Math.exp(-dt * 5)); // eased, so the scrub glides
    const i = Math.floor(p), u = p - i;
    activate(Math.min(N - 1, Math.floor(p + SEAM * 0.5)));
    if (meter) meter.style.transform = `scaleY(${(p / (N - 0.0001)).toFixed(4)})`;

    grid.fill(0);
    const hand = i < N - 1 ? clamp((u - (1 - SEAM)) / SEAM, 0, 1) : 0; // how far into the seam
    weight = 1 - hand;
    scenes[i]({ put, cols, rows, t, u: clamp(u / (1 - SEAM), 0, 1), pointer });
    if (hand > 0) {
      weight = hand;
      scenes[i + 1]({ put, cols, rows, t, u: 0, pointer });
    }
    weight = 1;

    ctx.fillStyle = color;
    drawMarks(ctx, grid, cols, rows, CELL);
  };

  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches, t0 = performance.now();
  let raf = 0, last = 0, seen = false;
  const tick = (now: number) => {
    raf = 0;
    if (!seen || document.hidden) return;
    raf = requestAnimationFrame(tick);
    if (now - last < 40) return; // ~25 fps
    last = now;
    frame(reduced ? 2 : (now - t0) / 1000, reduced);
  };
  const run = () => raf || (raf = requestAnimationFrame(tick));

  const onMove = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    pointer = { x: (e.clientX - r.left) / CELL, y: (e.clientY - r.top) / CELL };
  };
  const onLeave = () => (pointer = null);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);
  const io = new IntersectionObserver(([e]) => ((seen = e.isIntersecting), seen && run()));
  io.observe(root);
  const onVis = () => !document.hidden && run();
  document.addEventListener("visibilitychange", onVis);
  let rt: ReturnType<typeof setTimeout> | undefined;
  const onResize = () => {
    clearTimeout(rt);
    rt = setTimeout(() => size() && frame(reduced ? 2 : lastT || 2, true), 150);
  };
  addEventListener("resize", onResize);

  // the rail jumps to a step's middle
  const tickHandlers = ticks.map((a, k) => {
    const fn = (e: MouseEvent) => {
      e.preventDefault();
      const top = root.getBoundingClientRect().top + scrollY, runDist = root.offsetHeight - innerHeight;
      scrollTo({ top: top + ((k + 0.45) / N) * runDist, behavior: reduced ? "auto" : "smooth" });
    };
    a.addEventListener("click", fn);
    return [a, fn] as const;
  });

  if (size()) frame(2, true);

  return () => {
    cancelAnimationFrame(raf);
    clearTimeout(rt);
    io.disconnect();
    canvas.removeEventListener("pointermove", onMove);
    canvas.removeEventListener("pointerleave", onLeave);
    document.removeEventListener("visibilitychange", onVis);
    removeEventListener("resize", onResize);
    for (const [a, fn] of tickHandlers) a.removeEventListener("click", fn);
  };
}
