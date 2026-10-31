import { createDecoder } from "../../web/src/art/decode";
import { startSideArt, startSpinner } from "../../web/src/art/glyphs";
import { markSVG } from "../../web/src/art/mark";
import { startDive } from "./dive";
import { api, erase, layers, scene, steer } from "./scenes";

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

// ---------------------------------------------------------------- hero: falling through layer after layer, forever
function hero() {
  const readout = $("#hero-depth");
  startSpinner($<HTMLCanvasElement>("#hero-spin"), () => true);
  startDive($<HTMLCanvasElement>("#hero-art"), (level) => {
    const layer = 1 + (((level * 7) % 32) + 32) % 32, neuron = 100 + Math.floor(Math.abs(Math.sin(level * 91.7)) * 14000);
    readout.textContent = `L${layer} · n${neuron}`;
  });
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

reveal();
hero();
scene($<HTMLCanvasElement>("#art-erase"), 4, erase, 8.5);
scene($<HTMLCanvasElement>("#art-layers"), 4, layers, 2.4);
scene($<HTMLCanvasElement>("#art-steer"), 4, steer);
scene($<HTMLCanvasElement>("#art-api"), 4, api, 1.8);
startSideArt($<HTMLCanvasElement>("#sky-art"));

let chatStarted = false;
whileSeen($("#chat-demo"), (seen) => seen && !chatStarted && ((chatStarted = true), chat()), "0px 0px -20% 0px");
