import { startSideArt } from "../../web/src/art/glyphs";
import { createDecoder } from "../../web/src/art/decode";
import { markSVG } from "../../web/src/art/mark";
import { startLodestar } from "./lodestar";
import { api, chat, erase, layers, steer } from "./scenes";
import { startSequence } from "./sequence";

const $ = <T extends HTMLElement>(s: string) => document.querySelector<T>(s)!;

// ---------------------------------------------------------------- the mark
for (const el of document.querySelectorAll<HTMLElement>("[data-mark]")) {
  const px = +el.dataset.mark!;
  el.innerHTML = markSVG(px, px < 30 ? 21 : 31);
}

// ---------------------------------------------------------------- hero: the lodestar, and the sky turning about it
function hero() {
  const readout = $("#hero-depth"), act = $("#hero-act"), wide = matchMedia("(min-width: 901px)");

  // scrolling winds the sky faster, easing back when the page stops
  let boost = 0, lastY = scrollY;
  addEventListener("scroll", () => { boost = Math.min(8, boost + Math.abs(scrollY - lastY) / 30); lastY = scrollY; }, { passive: true });
  startLodestar($<HTMLCanvasElement>("#hero-art"), {
    at: () => (wide.matches ? [0.7, 0.5] : [0.5, 0.5]),
    boost: () => (boost *= 0.9),
  });
  const probe = () => (readout.textContent = `L${1 + Math.floor(Math.random() * 32)} · n${100 + Math.floor(Math.random() * 14000)}`);
  probe();
  setInterval(probe, 1800);
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
    else title.style.minHeight = "";
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

