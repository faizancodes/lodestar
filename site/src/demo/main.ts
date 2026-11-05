// Lodestar in the browser: the app's Chat view, with Qwen3-0.6B running on this page's GPU instead of the Mac app's
// server. The markup, styles and scenes are the app's own (web/src); what's here is the wiring.
import "../../../web/src/styles/fonts/fonts.css";
import "../../../web/src/styles/style.css";
import "./demo.css";
import { startSideArt } from "../../../web/src/art/glyphs";
import { finishScene, setLayers, setProgress, startScene, stopScene, wake } from "../../../web/src/art/loader";
import { markSVG } from "../../../web/src/art/mark";
import { cleanReply, esc, fmt, lensLabel, nid, showTok } from "../../../web/src/lib/format";
import { md } from "../../../web/src/lib/markdown";
import { CFG, gpu, Qwen, type Inspect } from "./qwen";
import { STOP, Tok, type Msg } from "./tok";

type Pin = { layer: number; neuron: number; dose: number; lens?: string };
type User = { role: "user"; content: string };
type Bot = {
  role: "assistant"; content: string; prompt: number[]; ids: number[]; pieces?: string[];
  mods: { steer: Pin[]; abl: boolean }; wps?: number; stopped?: boolean; error?: string;
};

// the app's starters and their marks (web/src/chat/engine.ts, views/Chat.tsx)
const STARTERS = ["Explain how a rainbow forms", "Write a haiku about Mondays", "Give me three dinner ideas", "What would you do with a free afternoon?"];
const MARKS = ["·+×", "+×▪", "×+·", "▪×+"];
const DOTS = `<span class="dots"><i></i><i></i><i></i></span>`;
const CROSS = `<svg viewBox="0 0 10 10" aria-hidden="true"><path d="m2.5 2.5 5 5m0-5-5 5" /></svg>`;

const $ = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector<T>(s)!;
const still = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const words = (s: string) => (s.match(/\S+/g) ?? []).length;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const st = {
  qwen: null as Qwen | null, tok: null as Tok | null,
  msgs: [] as (User | Bot)[], busy: false, abort: null as AbortController | null,
  pins: [] as Pin[], steer: true, erase: false,
  side: null as null | "neurons" | "inspect", inspect: false,
  picked: null as null | { m: number; k: number; data?: Inspect; lens?: string[]; err?: string },
  addErr: "",
};

// ------------------------------------------------------------------ loading
async function boot() {
  $("#brand-mark").innerHTML = markSVG(36, 15);
  startSideArt($<HTMLCanvasElement>("#side-art"));
  startScene($<HTMLCanvasElement>("#start-art"));
  renderStarters();
  wire();

  const dev = await gpu();
  const pick = $("#pick");
  if (typeof dev === "string") {
    pick.classList.add("off");
    $("#pick-sub").textContent = "Needs WebGPU: a recent Chrome, Edge or Safari";
    $("#pick-go").hidden = true;
    $("#start-status").textContent = dev;
    return;
  }

  const go = () => load(dev);
  pick.addEventListener("click", go, { once: true });
  pick.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && go(), { once: true });
}

function status(cls: string, msg: string) {
  $("#model-card").className = "model-card " + cls;
  $("#model-status .txt").textContent = msg;
}

async function load(dev: GPUDevice) {
  const start = $("#start");
  if (start.classList.contains("loading")) return;
  status("busy", "Loading Qwen3-0.6B…");

  if (!still()) {
    const card = $("#pick");
    card.classList.add("picked");
    await card.animate([{ opacity: 1, transform: "none" }, { opacity: 0, transform: "translateY(12px)" }],
      { duration: 280, delay: 120, easing: "cubic-bezier(0.4, 0, 1, 1)", fill: "forwards" }).finished;
  }
  start.classList.add("loading");
  setTimeout(wake, 300);

  const t0 = Date.now();
  let line = "Opening Qwen3-0.6B";
  const clock = () => {
    const s = Math.floor((Date.now() - t0) / 1000);
    $("#start-status").textContent = `${line} ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };
  clock();
  const timer = setInterval(clock, 1000);

  try {
    const [pack, tok] = await Promise.all([fetch("./demo/qwen3-0.6b.json").then((r) => r.json()), Tok.load()]);
    const qwen = new Qwen(dev, pack);
    await qwen.load((p) => {
      setLayers(CFG.L, p.lit);
      setProgress(p.done / p.total);
      line = `Reading weights · ${(p.done / 1e9).toFixed(1)} of ${(p.total / 1e9).toFixed(1)} GB`;
    });
    Object.assign(st, { qwen, tok });
  } catch (e) {
    clearInterval(timer);
    stopScene();
    start.classList.remove("loading");
    $("#pick").classList.remove("picked");
    $("#pick").getAnimations().forEach((a) => a.cancel());
    status("err", "Couldn’t load: " + (e as Error).message);
    $("#start-status").textContent = (e as Error).message;
    $("#pick").addEventListener("click", () => load(dev), { once: true });
    return;
  }

  clearInterval(timer);
  $("#start-status").textContent = "Ready";
  status("ready", "Ready · on this page’s GPU");
  await finishScene();
  if (!still()) {
    start.classList.add("leaving");
    await sleep(240);
  }
  stopScene();
  start.hidden = true;
  $("#main").classList.add("is-chat");
  $("#view-chat").hidden = false;
  renderAll();
  $("#chat-input").focus({ preventScroll: true });
}

// ------------------------------------------------------------------ chat
function history(upto: number): Msg[] {
  const sys = $<HTMLTextAreaElement>("#chat-system").value.trim();
  const out: Msg[] = sys ? [{ role: "system", content: sys }] : [];
  for (const m of st.msgs.slice(0, upto)) out.push({ role: m.role, content: m.role === "assistant" ? cleanReply(m.content) : m.content });
  return out;
}

async function send(text: string) {
  const { qwen, tok } = st;
  if (!qwen || !tok || st.busy || !text.trim()) return;

  st.msgs.push({ role: "user", content: text.trim() });
  const prompt = tok.prompt(history(st.msgs.length));
  const steer = st.steer ? st.pins.map((p) => ({ ...p })) : [];
  const bot: Bot = { role: "assistant", content: "", prompt, ids: [], mods: { steer, abl: st.erase } };
  st.msgs.push(bot);
  $<HTMLTextAreaElement>("#chat-input").value = "";
  growInput();

  qwen.setPushes(steer);
  qwen.erase = st.erase;
  st.busy = true;
  st.abort = new AbortController();
  renderAll();
  toBottom(true);

  const t0 = performance.now();
  try {
    const max = +$<HTMLInputElement>("#chat-max").value, temp = +$<HTMLInputElement>("#chat-temp").value;
    for await (const id of qwen.generate(prompt, { max, temp, stop: STOP, signal: st.abort.signal })) {
      bot.ids.push(id);
      bot.content = tok.decode(bot.ids);
      const secs = (performance.now() - t0) / 1000;
      if (secs > 0.5) bot.wps = words(bot.content) / secs;
      renderLast();
    }
  } catch (e) {
    bot.error = (e as Error).message;
  }

  bot.stopped = st.abort.signal.aborted;
  bot.pieces = bot.ids.map((_, k) => tok.decode(bot.ids.slice(0, k + 1)).slice(tok.decode(bot.ids.slice(0, k)).length));
  st.busy = false;
  st.abort = null;
  renderAll();
}

function retry() {
  if (st.busy) return;
  const last = st.msgs.length - 1;
  const user = st.msgs[last - 1];
  if (st.msgs[last]?.role !== "assistant" || user?.role !== "user") return;
  st.msgs.splice(last - 1, 2);
  send(user.content);
}

function newChat() {
  st.abort?.abort();
  st.msgs = [];
  st.picked = null;
  renderAll();
}

// ------------------------------------------------------------------ inspect
async function pickToken(m: number, k: number) {
  const bot = st.msgs[m] as Bot;
  if (!st.qwen || !st.tok || st.busy) return;
  st.picked = { m, k };
  st.side = "inspect";
  renderAll();

  try {
    const data = await st.qwen.inspect(bot.prompt.concat(bot.ids), bot.prompt.length + k);
    const lens = await st.qwen.neuronLens(data.neurons.map((n) => ({ layer: n.layer, neuron: n.neuron, sign: Math.sign(n.score) })));
    if (st.picked?.m !== m || st.picked.k !== k) return;
    st.picked.data = data;
    st.picked.lens = lens.map((ids) => lensLabel(ids.map((i) => st.tok!.decode([i]))));
  } catch (e) {
    if (st.picked) st.picked.err = (e as Error).message;
  }
  renderSide();
}

function togglePin(layer: number, neuron: number, dose: number, lens?: string) {
  const i = st.pins.findIndex((p) => p.layer === layer && p.neuron === neuron);
  if (i >= 0) st.pins.splice(i, 1);
  else st.pins.push({ layer, neuron, dose, lens });
  renderAll();
}

async function labelPins() {
  const want = st.pins.filter((p) => p.lens === undefined);
  if (!want.length || !st.qwen) return;
  want.forEach((p) => (p.lens = ""));
  const lens = await st.qwen.neuronLens(want.map((p) => ({ layer: p.layer, neuron: p.neuron, sign: Math.sign(p.dose) })));
  want.forEach((p, i) => (p.lens = lensLabel(lens[i].map((id) => st.tok!.decode([id])))));
  renderSide();
}

// "L9 N2578 +1.2", several separated by commas: as the app's Neurons box takes them
function addNeurons(text: string) {
  const out: Pin[] = [];
  for (const part of text.split(/[,;\n]+/).map((s) => s.trim()).filter(Boolean)) {
    const m = /^L?\s*(\d+)\s*[·:/\s]\s*N?\s*(\d+)(?:\s+([+−–-]?\s*\d*\.?\d+))?$/i.exec(part);
    if (!m) return void (st.addErr = `Couldn’t read “${part}”. Write a layer, a neuron and a dose, like L9 N2578 +1.2.`);
    const [layer, neuron] = [+m[1], +m[2]], dose = m[3] ? parseFloat(m[3].replace(/[−–]/, "-").replace(/\s/g, "")) : 1;
    if (layer >= CFG.L) return void (st.addErr = `Qwen3-0.6B has layers 0 to ${CFG.L - 1}.`);
    if (neuron >= CFG.N) return void (st.addErr = `Each layer has neurons 0 to ${CFG.N - 1}.`);
    if (dose) out.push({ layer, neuron, dose });
  }

  st.addErr = "";
  for (const p of out) {
    const old = st.pins.find((q) => q.layer === p.layer && q.neuron === p.neuron);
    if (old) old.dose = p.dose;
    else st.pins.push(p);
  }
  $<HTMLInputElement>("#chat-add").value = "";
  renderAll();
  labelPins();
}

// ------------------------------------------------------------------ rendering
function modsLine(mods: Bot["mods"]) {
  if (!mods.steer.length && !mods.abl) return "";
  return `<div class="msg-mods">${mods.steer.map((p) => `<span class="msg-mod"><b>${esc(p.lens?.split(" · ")[0]?.trim() || nid(p.layer, p.neuron))}</b> ${p.dose > 0 ? "+" : "−"}${+Math.abs(p.dose).toFixed(2)}</span>`).join("")}${mods.abl ? `<span class="msg-mod"><b>Refusals erased</b></span>` : ""}</div>`;
}

function botBody(m: Bot, i: number, typing: boolean) {
  if (st.inspect && !typing && m.pieces?.length) {
    const on = st.picked?.m === i ? st.picked.k : -1;
    return `<div class="md tks">${m.pieces.map((p, k) => `<span class="tk${k === on ? " on" : ""}" data-m="${i}" data-k="${k}">${esc(p)}</span>`).join("")}</div>`;
  }
  return `<div class="md">${m.content ? md(cleanReply(m.content)) : typing ? DOTS : `<p class="msg-none">No reply</p>`}</div>`;
}

function renderLog() {
  const log = $("#chat-log");
  log.innerHTML = st.msgs.map((m, i) => {
    if (m.role === "user") return `<div class="msg user"><div class="bubble">${esc(m.content)}</div></div>`;
    const last = i === st.msgs.length - 1, typing = last && st.busy;
    const rate = m.wps ? `${m.wps.toFixed(1)} words/s` : "";
    const acts = typing ? "" : `<button data-a="copy" data-i="${i}">Copy</button>${last ? `<button data-a="retry">Retry</button>` : ""}`;
    return `<div class="msg bot${typing ? " typing" : ""}" data-i="${i}">${botBody(m, i, typing)}${modsLine(m.mods)}
      <div class="msg-acts">${acts}${m.stopped ? `<span class="msg-note">Stopped</span>` : ""}${m.error ? `<span class="msg-note">${esc(m.error)}</span>` : ""}<span class="msg-rate">${rate}</span></div></div>`;
  }).join("");
  $("#chat-empty").hidden = st.msgs.length > 0;
}

function renderLast() {
  const i = st.msgs.length - 1, m = st.msgs[i] as Bot;
  const el = document.querySelector(`#chat-log .msg[data-i="${i}"]`);
  if (!el) return renderLog();
  el.querySelector(".md")!.innerHTML = m.content ? md(cleanReply(m.content)) : DOTS;
  el.querySelector(".msg-rate")!.textContent = m.wps ? `${m.wps.toFixed(1)} words/s` : "";
  toBottom();
}

function renderMods() {
  const n = st.pins.length;
  $("#chat-mods").innerHTML = `<button class="mod${st.erase ? " on" : ""}" data-mod="abl" title="Erase the refusal direction in replies">Refusals erased</button>`
    + (n ? `<button class="mod${st.steer ? " on" : ""}" data-mod="steer" title="${esc(st.pins.map((p) => nid(p.layer, p.neuron)).join(", "))}">Steered · ${n} neuron${n > 1 ? "s" : ""}</button>` : "");
  const nb = $("#chat-neurons-btn"), ib = $("#chat-inspect-btn");
  nb.classList.toggle("on", st.side === "neurons");
  nb.setAttribute("aria-expanded", String(st.side === "neurons"));
  ib.classList.toggle("on", st.inspect);
  ib.setAttribute("aria-pressed", String(st.inspect));
}

function renderSide() {
  const side = $("#chat-side");
  side.hidden = !st.side;
  if (!st.side) return;
  const head = (t: string) => `<div class="nr-h"><span class="eyebrow light">${t}</span><button class="nr-x" data-a="close" aria-label="Close">×</button></div>`;

  if (st.side === "neurons") {
    side.innerHTML = head("Neurons") + `
      <form class="nr-add${st.addErr ? " bad" : ""}" id="nr-add"><input id="chat-add" placeholder="L9 N2578 +1.2" spellcheck="false" autocomplete="off" /><button class="btn sm">Add</button></form>
      <p class="nr-err"${st.addErr ? "" : " hidden"}>${esc(st.addErr)}</p>
      <div class="nr-pins">${st.pins.map((p, i) => `
        <div class="nr-pin" data-pin="${i}">
          <div class="nr-id"><b>L${p.layer}</b><span>N${p.neuron}</span></div>
          <div class="nr-dose${p.dose < 0 ? " neg" : ""}"><button data-a="less" aria-label="Less">−</button><input data-a="dose" aria-label="Dose" value="${(p.dose > 0 ? "+" : "") + +p.dose.toFixed(2)}" spellcheck="false" /><button data-a="more" aria-label="More">+</button></div>
          <button class="nr-del" data-a="del" aria-label="Remove">${CROSS}</button>
          <div class="nr-lens">${esc(p.lens ?? "")}</div>
        </div>`).join("")}</div>`;
    return;
  }

  const pk = st.picked, bot = pk && (st.msgs[pk.m] as Bot | undefined);
  if (!pk || !bot?.pieces) {
    side.innerHTML = head("Inspect") + `<p class="nr-note">Pick a word in a reply.</p>`;
    return;
  }

  const word = bot.pieces[pk.k];
  let body = `<div class="insp-tok"><b>${esc(showTok(word))}</b><span>token ${pk.k + 1} of ${bot.pieces.length}</span></div>`;
  if (pk.err) body += `<p class="nr-err">${esc(pk.err)}</p>`;
  else if (!pk.data) body += `<p class="nr-note">Reading every layer…</p>`;
  else {
    const on = new Set(st.pins.map((p) => `${p.layer}:${p.neuron}`));
    body += `<div class="insp-sec"><span class="eyebrow light">Firing hardest</span></div><div class="nr-found">${pk.data.neurons.map((n, i) => `
      <button class="nr-card" role="switch" aria-checked="${on.has(`${n.layer}:${n.neuron}`)}" data-l="${n.layer}" data-n="${n.neuron}" data-s="${Math.sign(n.score)}" data-lens="${esc(pk.lens?.[i] ?? "")}">
        <div class="nr-id"><b>L${n.layer}</b><span>N${n.neuron}</span><em>${fmt(n.score)}</em></div><i></i>
        <div class="nr-lens">${esc(pk.lens?.[i] ?? "")}</div>
      </button>`).join("")}</div>
      <div class="insp-sec"><span class="eyebrow light">Next word, layer by layer</span></div>
      <ol class="lens">${pk.data.lens.map((top, l) => `<li><span class="lens-l">L${l}</span><span class="lens-t">${esc(showTok(st.tok!.decode([top[0].id])))}</span><span class="lens-p">${Math.round(top[0].p * 100)}%</span></li>`).join("")}</ol>`;
  }
  side.innerHTML = head("Inspect") + body;
}

function renderStarters() {
  $("#chat-starters").innerHTML = STARTERS.map((s, i) => `<button class="starter" data-starter="${esc(s)}"><span class="starter-mark" aria-hidden="true">${MARKS[i]}</span><span>${esc(s)}</span><span class="starter-go" aria-hidden="true">↗</span></button>`).join("");
}

function renderComposer() {
  const empty = !$<HTMLTextAreaElement>("#chat-input").value.trim();
  $<HTMLButtonElement>("#chat-send").hidden = st.busy;
  $<HTMLButtonElement>("#chat-send").disabled = empty;
  $<HTMLButtonElement>("#chat-stop").hidden = !st.busy;
}

function renderAll() {
  renderLog();
  renderMods();
  renderSide();
  renderComposer();
}

function toBottom(force = false) {
  const s = $("#chat-scroll");
  if (force || s.scrollHeight - s.scrollTop - s.clientHeight < 120) s.scrollTop = s.scrollHeight;
}

function growInput() {
  const t = $<HTMLTextAreaElement>("#chat-input");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 220) + "px";
  renderComposer();
}

// ------------------------------------------------------------------ events
function wire() {
  const input = $<HTMLTextAreaElement>("#chat-input");
  input.addEventListener("input", growInput);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) (e.preventDefault(), send(input.value));
  });
  $("#chat-form").addEventListener("submit", (e) => (e.preventDefault(), send(input.value)));
  $("#chat-stop").addEventListener("click", () => st.abort?.abort());
  $("#chat-new").addEventListener("click", newChat);
  $("#chat-starters").addEventListener("click", (e) => {
    const b = (e.target as Element).closest<HTMLElement>("[data-starter]");
    if (b) send(b.dataset.starter!);
  });

  $("#chat-neurons-btn").addEventListener("click", () => ((st.side = st.side === "neurons" ? null : "neurons"), renderAll(), st.side && $("#chat-add").focus({ preventScroll: true })));
  $("#chat-inspect-btn").addEventListener("click", () => {
    st.inspect = !st.inspect;
    st.side = st.inspect ? "inspect" : st.side === "inspect" ? null : st.side;
    if (!st.inspect) st.picked = null;
    renderAll();
  });

  $("#chat-mods").addEventListener("click", (e) => {
    const b = (e.target as Element).closest<HTMLElement>("[data-mod]");
    if (b?.dataset.mod === "abl") st.erase = !st.erase;
    if (b?.dataset.mod === "steer") st.steer = !st.steer;
    renderMods();
  });

  $("#chat-log").addEventListener("click", async (e) => {
    const t = e.target as HTMLElement;
    const tk = t.closest<HTMLElement>(".tk");
    if (tk) return pickToken(+tk.dataset.m!, +tk.dataset.k!);
    const a = t.closest<HTMLElement>("[data-a]")?.dataset.a;
    if (a === "retry") retry();
    if (a === "copy") {
      const b = t.closest<HTMLElement>("[data-a]")!;
      await navigator.clipboard.writeText(cleanReply((st.msgs[+b.dataset.i!] as Bot).content)).catch(() => {});
      b.textContent = "Copied";
      setTimeout(() => (b.textContent = "Copy"), 1400);
    }
  });

  const side = $("#chat-side");
  side.addEventListener("submit", (e) => (e.preventDefault(), addNeurons($<HTMLInputElement>("#chat-add").value)));
  side.addEventListener("click", (e) => {
    const t = e.target as HTMLElement;
    const card = t.closest<HTMLElement>(".nr-card");
    if (card) return togglePin(+card.dataset.l!, +card.dataset.n!, +card.dataset.s! || 1, card.dataset.lens);
    const a = t.closest<HTMLElement>("[data-a]")?.dataset.a, pin = st.pins[+(t.closest<HTMLElement>("[data-pin]")?.dataset.pin ?? -1)];
    if (a === "close") {
      if (st.side === "inspect") (st.inspect = false), (st.picked = null);
      st.side = null;
      renderAll();
    }
    if (!pin) return;
    if (a === "del") st.pins.splice(st.pins.indexOf(pin), 1);
    if (a === "less" || a === "more") {
      const d = Math.round((pin.dose + (a === "more" ? 0.25 : -0.25)) / 0.25) * 0.25;
      if (!d) return;
      if (Math.sign(d) !== Math.sign(pin.dose)) pin.lens = undefined;
      pin.dose = d;
    }
    if (a === "del" || a === "less" || a === "more") (renderAll(), labelPins());
  });
  side.addEventListener("change", (e) => {
    const t = e.target as HTMLInputElement;
    if (t.dataset.a !== "dose") return;
    const pin = st.pins[+(t.closest<HTMLElement>("[data-pin]")?.dataset.pin ?? -1)];
    const d = parseFloat(t.value.replace(/[−–]/g, "-"));
    if (pin && Number.isFinite(d) && d) {
      if (Math.sign(d) !== Math.sign(pin.dose)) pin.lens = undefined;
      pin.dose = d;
    }
    renderAll();
    labelPins();
  });

  for (const [id, digits] of [["chat-temp", 2], ["chat-max", 0]] as const) {
    const r = $<HTMLInputElement>("#" + id), fill = () => {
      r.style.setProperty("--fill", `${((+r.value - +r.min) / (+r.max - +r.min)) * 100}%`);
      $("#" + id + "-v").textContent = (+r.value).toFixed(digits);
    };
    r.addEventListener("input", fill);
    fill();
  }
  document.addEventListener("click", (e) => !(e.target as Element).closest("#chat-opts") && $("#chat-opts").removeAttribute("open"));
}

// for poking at the engine from the console while developing
if ((import.meta as { env?: { DEV?: boolean } }).env?.DEV) Object.assign(window, { demo: { st, send } });

boot();
