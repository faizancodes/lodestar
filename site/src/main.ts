import { startSideArt, startSpinner } from "../../web/src/art/glyphs";
import { createDecoder } from "../../web/src/art/decode";
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
  const readout = $("#hero-depth"), act = $("#hero-act"), wide = matchMedia("(min-width: 901px)");
  startSpinner($<HTMLCanvasElement>("#hero-spin"), () => true);

  // scrolling pushes the fall faster, easing back when the page stops
  let boost = 0, lastY = scrollY;
  addEventListener("scroll", () => { boost = Math.min(6, boost + Math.abs(scrollY - lastY) / 40); lastY = scrollY; }, { passive: true });
  startDive($<HTMLCanvasElement>("#hero-art"), {
    at: () => (wide.matches ? 0.64 : 0.5),
    boost: () => (boost *= 0.9),
    depth: (level) => {
      const layer = 1 + (((level * 7) % 32) + 32) % 32, neuron = 100 + Math.floor(Math.abs(Math.sin(level * 91.7)) * 14000);
      readout.textContent = `L${layer} · n${neuron}`;
    },
  });
  setInterval(() => {
    const v = Math.sin(performance.now() / 900) * 0.6 + (Math.random() - 0.5) * 0.25;
    act.textContent = (v < 0 ? "−" : "+") + Math.abs(v).toFixed(3);
  }, 120);

  // the title types in, each letter decoding through the marks
  const title = $("#hero-title"), text = title.textContent!, dec = createDecoder();
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  title.style.minHeight = `${title.offsetHeight}px`;
  let n = 0;
  const type = () => {
    title.textContent = text.slice(0, ++n);
    dec.update(title);
    if (n < text.length) setTimeout(type, 38);
  };
  title.textContent = "";
  setTimeout(type, 250);
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

