import { startSideArt, startSpinner } from "../../web/src/art/glyphs";
import { markSVG } from "../../web/src/art/mark";
import { startDive } from "./dive";
import { api, chat, erase, layers, steer } from "./scenes";
import { startSequence } from "./sequence";

const $ = <T extends HTMLElement>(s: string) => document.querySelector<T>(s)!;

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

// ---------------------------------------------------------------- the rest
function reveal() {
  const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && (e.target.classList.add("in"), io.unobserve(e.target))), { rootMargin: "0px 0px -8% 0px" });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
}

reveal();
hero();
startSequence($("#features"), [erase, layers, steer, chat, api]);
startSideArt($<HTMLCanvasElement>("#sky-art"));

