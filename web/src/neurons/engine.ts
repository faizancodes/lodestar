// Neurons: push the neurons you name by hand, and talk to the model with them on, beside its plain self.
import { flushSync } from "react-dom";
import { createDecoder } from "../art/decode";
import { replyHtml } from "../chat/engine";
import { api } from "../lib/api";
import { modelName } from "../lib/format";
import { load, save } from "../lib/storage";
import { stream } from "../lib/stream";
import { still } from "../shell/engine";
import { useApp } from "../store/app";
import type { Col, Msg } from "../store/chat";
import { touchNeurons, useNeurons, type Pin } from "../store/neurons";
import { parsePins } from "./parse";

const app = useApp.getState;
const get = useNeurons.getState;
const set = useNeurons.setState;

const scroller = () => document.getElementById("nr-scroll");
const input = () => document.getElementById("nr-input") as HTMLTextAreaElement | null;
const lastMds = () => [...document.querySelectorAll<HTMLElement>("#nr-log .msg.bot:last-child .md")];
const render = () => flushSync(touchNeurons);
const toBottom = () => scroller()?.scrollTo({ top: scroller()!.scrollHeight, behavior: still() ? "auto" : "smooth" });

// ------------------------------------------------------------------ the neurons
// kept per model, by name: neuron numbers mean nothing in another one
const key = () => `neurons:${modelName(app().info?.model) ?? ""}`;
const keep = () => save(key(), JSON.stringify(get().pins.map(({ lens: _, ...p }) => p)));

export function addPins() {
  const { layers = 0, neurons = 0 } = app().info ?? {};
  const found = parsePins(get().add);
  if (!found) return set({ err: "Write each one as L14 N8949 +1." });

  const bad = found.find((p) => p.layer >= layers || p.neuron >= neurons);
  if (bad) {
    return set({ err: bad.layer >= layers ? `There’s no layer ${bad.layer}: this model has layers 0–${layers - 1}.`
      : `There’s no neuron ${bad.neuron}: each layer has neurons 0–${neurons - 1}.` });
  }

  const pins = [...get().pins];
  for (const f of found) {
    const i = pins.findIndex((p) => p.layer === f.layer && p.neuron === f.neuron);
    if (i >= 0) pins[i] = { ...pins[i], dose: f.dose, on: true };
    else pins.push({ ...f, on: true });
  }

  set({ pins, add: "", err: "" });
  keep();
  pins.forEach(fetchLens);
}

async function fetchLens(p: Pin) {
  if (p.lens) return;

  try {
    const r = await api<{ lens: Pin["lens"] }>(`/api/neuron?layer=${p.layer}&n=${p.neuron}`);
    set({ pins: get().pins.map((q) => (q.layer === p.layer && q.neuron === p.neuron ? { ...q, lens: r.lens } : q)) });
  } catch {}
}

export function updatePin(i: number, p: Partial<Pin>) {
  set({ pins: get().pins.map((q, j) => (j === i ? { ...q, ...p } : q)) });
  keep();
}

export function removePin(i: number) {
  set({ pins: get().pins.filter((_, j) => j !== i) });
  keep();
}

/** The model is ready, or another one is: its own neurons, and a fresh conversation. */
export function onNeuronsModel() {
  let pins: Pin[] = [];
  try { pins = JSON.parse(load(key()) ?? "[]"); } catch {}

  set({ pins, messages: [], err: "" });
  pins.forEach(fetchLens);
}

const on = () => get().pins.filter((p) => p.on && p.dose);
const pushes = () => on().map((p) => ({ layer: p.layer, neuron: p.neuron, dose: p.dose }));

// ------------------------------------------------------------------ the conversation
const parts = (m: Msg): { content: string }[] => m.cols ?? [m];

// a side-by-side turn continues each column's own thread
function history(key: Col["key"]) {
  return get().messages.slice(-24).map(({ role, content, cols }) => ({
    role,
    content: cols ? (cols.find((c) => c.key === key) ?? cols[0]).content : content,
  }));
}

async function reply() {
  const many = get().compare && on().length > 0;
  const body = many
    ? { rows: [{ messages: history("plain"), pushes: [], abl: false }, { messages: history("steer"), pushes: pushes(), abl: false }], temperature: 0.7, max_new: 512 }
    : { messages: history("steer"), pushes: pushes(), abl: false, temperature: 0.7, max_new: 512 };

  const m: Msg = { role: "assistant", content: "", ...(many && { cols: [{ key: "plain", label: "Plain", content: "" }, { key: "steer", label: "Steered", content: "" }] }) };
  get().messages.push(m);
  set({ streaming: true });
  const decoders = parts(m).map(() => createDecoder());
  render();
  toBottom();

  let raf = 0;
  const draw = () => {
    raf = 0;
    lastMds().forEach((el, i) => {
      const part = parts(m)[i];
      if (!part?.content) return;
      el.innerHTML = replyHtml(part);
      decoders[i]?.update(el);
    });
    toBottom();
  };

  const ok = await stream("/api/chat", body, (ev) => {
    if (ev.type !== "delta") return;
    parts(m)[many ? (ev.row as number) : 0].content += ev.delta as string;
    if (!raf) raf = requestAnimationFrame(draw);
  });

  cancelAnimationFrame(raf);
  set({ streaming: false });

  if (!parts(m).some((p) => p.content.trim())) get().messages.pop();
  else if (!ok) m.stopped = true;
  render();

  lastMds().forEach((el, i) => decoders[i]?.update(el));
  input()?.focus();
}

export function send(text?: string) {
  text = (text ?? get().input).trim();
  if (!text || get().streaming) return;
  if (app().busy) return app().toastMsg("Something is already running. Stop it first.", true);

  get().messages.push({ role: "user", content: text });
  flushSync(() => set({ input: "" }));
  reply();
}

export function retry() {
  if (get().streaming) return;
  get().messages.pop();
  reply();
}

export function clearChat() {
  if (get().streaming) api("/api/cancel", {});
  set({ messages: [] });
  render();
  input()?.focus();
}

export function toggleCompare() {
  save("neurons-compare", get().compare ? "0" : "1");
  set({ compare: !get().compare });
}

/** The tab was opened. */
export function onNeuronsShown() {
  render();
  scroller()?.scrollTo({ top: scroller()!.scrollHeight });
  requestAnimationFrame(() => (get().pins.length ? input() : document.getElementById("nr-add"))?.focus());
}
