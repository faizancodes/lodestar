import { startAblScene } from "../../web/src/art/ablscene";
import { createDecoder } from "../../web/src/art/decode";
import { startSideArt, startSpinner } from "../../web/src/art/glyphs";
import { render as renderHomeArt, type HomeArt } from "../../web/src/art/home";
import { finishScene, setLayers, setProgress, startScene, stopScene, wake } from "../../web/src/art/loader";
import { markSVG } from "../../web/src/art/mark";
import { startScan } from "../../web/src/art/scan";

const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = <T extends HTMLElement>(s: string) => document.querySelector<T>(s)!;
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Calls on(true) when el comes into view and on(false) when it leaves. */
function whileSeen(el: Element, on: (seen: boolean) => void, margin = "0px") {
  new IntersectionObserver(([e]) => on(e.isIntersecting), { rootMargin: margin }).observe(el);
}

// ---------------------------------------------------------------- the mark
for (const el of document.querySelectorAll<HTMLElement>("[data-mark]")) {
  const px = +el.dataset.mark!;
  el.innerHTML = markSVG(px, px < 30 ? 21 : 31);
}

// ---------------------------------------------------------------- hero: the model arriving, layer by layer, on repeat
const LAYERS = 32, LOAD_S = 7.5;
async function hero() {
  const canvas = $<HTMLCanvasElement>("#hero-art"), status = $("#hero-status"), count = $("#hero-layers");
  const spin = startSpinner($<HTMLCanvasElement>("#hero-spin"), () => !status.dataset.ready);

  for (;;) {
    startScene(canvas);
    setLayers(LAYERS, null);
    wake();
    delete status.dataset.ready;
    $("#readout").classList.remove("ready");
    status.textContent = "Loading Qwen3.5-4B";
    spin();

    const t0 = performance.now();
    for (let p = 0; p < 1; p = (performance.now() - t0) / 1000 / LOAD_S) {
      setProgress(0.04 + p * 0.86);
      count.textContent = `${Math.min(LAYERS, Math.floor(p * LAYERS))} / ${LAYERS} layers`;
      await wait(120);
    }

    count.textContent = `${LAYERS} / ${LAYERS} layers`;
    await finishScene();
    status.dataset.ready = "1";
    $("#readout").classList.add("ready");
    status.textContent = "Qwen3.5-4B ready";
    if (still) return;

    await wait(5000);
    stopScene();
  }
}

// ---------------------------------------------------------------- the home cards' scenes, playing while on screen
function homeArts() {
  for (const canvas of document.querySelectorAll<HTMLCanvasElement>("canvas[data-art]")) {
    const kind = canvas.dataset.art as HomeArt;
    let raf = 0, last = 0;
    const t0 = performance.now() - Math.random() * 4000;
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      if (now - last < 50) return;
      last = now;
      renderHomeArt(canvas, kind, (now - t0) / 1000);
    };

    renderHomeArt(canvas, kind, 0);
    if (still) continue;
    whileSeen(canvas, (seen) => (seen ? raf || (raf = requestAnimationFrame(tick)) : (cancelAnimationFrame(raf), (raf = 0))));
    addEventListener("resize", () => renderHomeArt(canvas, kind, (performance.now() - t0) / 1000));
  }
}

// ---------------------------------------------------------------- abliterate: the four stages, in turn
function abliterate() {
  const scene = startAblScene($<HTMLCanvasElement>("#abl-art"));
  const items = [...document.querySelectorAll<HTMLElement>("#abl-stages li")];
  let at = 0, timer = 0;

  const show = (i: number) => {
    at = i;
    scene.set(i);
    items.forEach((li) => li.classList.toggle("on", +li.dataset.i! === i));
  };
  const next = () => { clearTimeout(timer); timer = setTimeout(() => (show((at + 1) % items.length), next()), 5200); };

  items.forEach((li) => li.addEventListener("click", () => (show(+li.dataset.i!), next())));
  show(0);
  if (!still) next();
}

// ---------------------------------------------------------------- chat: two replies decoding in, gauges reading along
const REPLIES = [
  "To the long night, and to every fool who thought they'd live to see the dawn. Drink deep, friends; by morning the city is mine.",
  "Raise yer cups, ye glorious bilge rats, to plunder, salt and sunshine! Drink up, for come the dawn the whole harbour be ours!",
];
const LEAN = [-0.34, 0.46]; // where each reply's Cheerful reading settles

function gaugeColor(v: number) {
  const k = Math.round(Math.min(1, Math.abs(v) / 0.5) * 100);
  return `color-mix(in srgb, var(${v >= 0 ? "--g-pos" : "--g-neg"}) ${k}%, var(--wash))`;
}

async function chat() {
  const els = REPLIES.map((_, i) => $(`[data-reply="${i}"]`));
  const strips = REPLIES.map((_, i) => $(`[data-strip="${i}"]`)), vals = REPLIES.map((_, i) => $(`[data-gv="${i}"]`));
  const tokens = REPLIES.map((r) => r.match(/\S+\s*/g)!);

  for (;;) {
    const decoders = REPLIES.map(() => createDecoder());
    els.forEach((el) => (el.textContent = ""));
    strips.forEach((s) => (s.textContent = ""));
    const shown = [0, 0], sums = [0, 0];

    while (shown.some((n, i) => n < tokens[i].length)) {
      tokens.forEach((toks, i) => {
        if (shown[i] >= toks.length || (i === 0 && Math.random() < 0.25)) return; // the two run at slightly different speeds
        const n = ++shown[i];
        els[i].textContent = toks.slice(0, n).join("");
        decoders[i].update(els[i]);

        const k = n / toks.length, v = LEAN[i] * (0.4 + 0.6 * k) + (Math.sin(n * 1.9 + i) * 0.12);
        sums[i] += v;
        const cell = document.createElement("i");
        cell.style.background = gaugeColor(v);
        strips[i].append(cell);
        vals[i].textContent = (sums[i] / n >= 0 ? "+" : "−") + Math.abs(sums[i] / n).toFixed(2);
      });

      await wait(still ? 0 : 95);
    }

    if (still) return;
    await wait(6500);
  }
}

// ---------------------------------------------------------------- the rest
function reveal() {
  const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add("in"), io.unobserve(e.target))), { rootMargin: "0px 0px -8% 0px" });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
}

function copyUrl() {
  const b = $<HTMLButtonElement>("#copy-url");
  b.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText("http://127.0.0.1:8792/v1"); } catch { return; }
    b.textContent = "Copied";
    setTimeout(() => (b.textContent = "Copy"), 1400);
  });
}

reveal();
hero();
homeArts();
abliterate();
startScan($<HTMLCanvasElement>("#scan-art"));
startSideArt($<HTMLCanvasElement>("#sky-art"));
copyUrl();

let chatStarted = false;
whileSeen($("#chat-demo"), (seen) => seen && !chatStarted && ((chatStarted = true), chat()), "0px 0px -20% 0px");
