// Chat: talk to the loaded model, with the recipe that's on and the erased refusal direction switched on or off.
import { flushSync } from "react-dom";
import { setAblActive } from "../abl/engine";
import { createDecoder } from "../art/decode";
import { api } from "../lib/api";
import { cleanReply, dirReady, isDir, isWord, itemKey, pushes, searchDose } from "../lib/format";
import { md } from "../lib/markdown";
import { save } from "../lib/storage";
import { stream } from "../lib/stream";
import { setTab, still } from "../shell/engine";
import { ablActive, useApp } from "../store/app";
import type { ChatSummary, NeuronItem, WordSearch } from "../lib/types";
import { addItems, removeAt } from "../steer/recipe";
import { parsePins } from "./parse";
import { touchChat, useChat, type Col, type ColKey, type Mods, type Msg, type Reading } from "../store/chat";

const app = useApp.getState;
const get = useChat.getState;
const set = useChat.setState;
const toast = (msg: string, err = false) => app().toastMsg(msg, err);

export const STARTERS = ["Explain how a rainbow forms", "Write a haiku about Mondays", "Give me three dinner ideas", "What would you do with a free afternoon?"];
const DOTS = `<span class="dots"><i></i><i></i><i></i></span>`;
const NONE = `<p class="msg-none">No reply</p>`;

/** A reply as HTML: the dots while it's coming, and a note when a finished one came out empty. */
export const replyHtml = (m: { content: string }, done = false) => (m.content ? md(cleanReply(m.content)) : done ? NONE : DOTS);

const scroller = () => document.getElementById("chat-scroll")!;
const input = () => document.getElementById("chat-input") as HTMLTextAreaElement | null;
const lastMds = () => [...document.querySelectorAll<HTMLElement>("#chat-log .msg.bot:last-child .md")];
const lastRates = () => [...document.querySelectorAll<HTMLElement>("#chat-log .msg.bot:last-child .msg-rate")];

// ------------------------------------------------------------------ speed
// When each streaming part got its first and latest words. Kept off the messages, which are saved as they are.
const timing = new WeakMap<object, { t0: number; t1: number; w0: number }>(); // w0: words it had by its first
const words = (s: string) => (s.trim() ? s.trim().split(/\s+/).length : 0);

/** Words a second: measured from the first word, so the wait before it (reading the prompt) doesn't count. */
export function rateOf(part: { content: string; wps?: number }) {
  if (part.wps) return part.wps;

  const t = timing.get(part);
  if (!t || t.t1 - t.t0 < 500) return null;
  return (words(part.content) - t.w0) / ((t.t1 - t.t0) / 1000);
}

export const rateText = (r: number | null) => (r == null ? "" : `${r.toFixed(1)} words/s`);

function arrived(part: object) {
  const now = performance.now(), t = timing.get(part);
  if (t) t.t1 = now;
  else timing.set(part, { t0: now, t1: now, w0: words((part as { content: string }).content) });
}

// keep following the reply unless the user has scrolled up to read
const nearBottom = () => { const s = scroller(); return s.scrollHeight - s.scrollTop - s.clientHeight < 80; };
let follow = true;
export function onScroll() {
  follow = nearBottom();
  set({ jump: !follow });
}

const toBottom = (smooth: boolean) => scroller().scrollTo({ top: scroller().scrollHeight, behavior: smooth && !still() ? "smooth" : "auto" });
export const jumpDown = () => ((follow = true), toBottom(true));

let decoders: ReturnType<typeof createDecoder>[] = []; // the replies being written decode in, mark by mark

// the parts of a reply that stream: the message itself, or its columns side by side
const parts = (m: Msg): { content: string; wps?: number; cut?: boolean; read?: Reading }[] => m.cols ?? [m];

function updateLastBot() {
  const m = get().messages[get().messages.length - 1];

  lastMds().forEach((el, i) => {
    const part = parts(m)[i];
    if (!part?.content) return;

    el.innerHTML = replyHtml(part);
    decoders[i]?.update(el);
  });

  lastRates().forEach((el, i) => {
    const part = parts(m)[i];
    if (part) el.textContent = rateText(rateOf(part));
  });

  set({ gv: get().gv + 1 });
  if (follow) toBottom(false);
}

const render = () => flushSync(touchChat);

/** What Compare puts side by side: the plain model, then each switch that's on, alone. */
export function compareCols(): Omit<Col, "content">[] {
  const a = app();
  const cols: Omit<Col, "content">[] = [{ key: "plain", label: "Plain" }];

  if (get().steer && a.recipe.items.length) cols.push({ key: "steer", label: "Steered" });
  if (ablActive(a)) cols.push({ key: "abl", label: "Refusals erased" });
  return cols;
}

// the column a single reply continues: what its switches make it
const soloKey = (): ColKey => (get().steer && app().recipe.items.length ? "steer" : ablActive(app()) ? "abl" : "plain");

/** The conversation as one column saw it: an earlier side-by-side turn gives that column's reply. */
function history(key: ColKey) {
  const sys = get().system.trim();

  const turns = get().messages.slice(-24).map(({ role, content, cols }) => ({
    role,
    content: cols ? (cols.find((c) => c.key === key) ?? cols[0]).content : content,
  }));

  return (sys ? [{ role: "system", content: sys }] : []).concat(turns);
}

/** The gauges that can be read now: the ones picked whose direction exists for the loaded model. */
export const gaugeDirs = () => {
  const dirs = app().dirs ?? [];
  return get().gauges.flatMap((id) => dirs.filter((d) => d.id === id && d.ready));
};

export function toggleGauge(id: string) {
  const g = get().gauges, next = g.includes(id) ? g.filter((x) => x !== id) : [...g, id];
  save("chat-gauges", JSON.stringify(next));
  set({ gauges: next });
}

// what a continued reply's gauges read
const live0 = (going: ReturnType<typeof parts> | null) => {
  const r = going?.[0]?.read;
  return r ? r.ids.map((id, i) => ({ id, name: r.names[i] })) : [];
};

/** What the switches put on a reply now: the recipe's neurons and ready directions as they're dosed, and the erasure. */
function modsNow(): Mods {
  const { recipe, dirs } = app();
  const steer = get().steer
    ? recipe.items.flatMap((it) => isDir(it)
      ? dirReady(it, dirs) ? [{ label: it.name || it.dir, dose: it.dose, ...(it.op === "erase" && { erase: true }) }] : []
      : [{ label: `L${it.layer} N${it.neuron}`, dose: it.dose }])
    : [];

  return { ...(steer.length && { steer }), ...(ablActive(app()) && { abl: true }) };
}

// a part of a reply that wrote to the length limit before it finished
const cutParts = (m: Msg) => parts(m).filter((p) => p.cut);

/** The next reply, or with `more`, the rest of the last one: each part that ran out of length carries on from where it stopped. */
async function reply(more = false) {
  const { recipe, dirs } = app();
  const last = get().messages[get().messages.length - 1];
  const m: Msg = more ? last : { role: "assistant", content: "" };
  const cols = more ? (m.cols ?? []) : get().compare ? compareCols() : [];
  const many = cols.length > 1;
  const going = more ? cutParts(m) : null; // the parts this continues, in row order

  const row = (key: ColKey, steer: boolean, abl: boolean) => ({
    messages: history(key),
    pushes: steer ? pushes(recipe.items, dirs, 1) : [],
    abl,
  });

  const body = many
    ? {
        rows: ((going ?? cols) as Col[]).map((c) => row(c.key, c.key === "steer", c.key === "abl")),
        temperature: +get().temp,
        max_new: +get().max,
        ...(more && { continue: true }),
      }
    : { ...row(soloKey(), get().steer, ablActive(app())), temperature: +get().temp, max_new: +get().max, ...(more && { continue: true }) };

  if (!more) {
    if (many) m.cols = cols.map((c) => ({ ...c, content: "" }));
    get().messages.push(m);
  }

  // a continued reply goes on reading what it read; a new one, the gauges that are on
  const reads = more ? live0(going) : gaugeDirs().map((d) => ({ id: d.id, name: d.name }));
  const read = reads.map((d) => d.id);
  if (read.length) Object.assign(body, { read });

  const live = going ?? parts(m); // row i streams into live[i]
  if (!more) live.forEach((p) => (p.read = read.length ? { ids: read, names: reads.map((d) => d.name), toks: [] } : undefined));
  live.forEach((p) => ((p.cut = false), delete p.wps, timing.delete(p))); // its speed is the new stretch's
  m.stopped = false;
  m.mods = modsNow();

  set({ streaming: true });
  decoders = parts(m).map(() => createDecoder());
  render();
  follow = true;
  toBottom(true);

  let raf = 0;
  const ok = await stream("/api/chat", body, (ev) => {
    if (ev.type === "done") {
      const cut = ev.cut as boolean | boolean[];
      live.forEach((p, i) => (p.cut = Array.isArray(cut) ? cut[i] : cut));
      return;
    }
    if (ev.type !== "delta") return;

    const part = live[many ? (ev.row as number) : 0];
    part.content += ev.delta as string;
    const rs = ev.read as number[][] | undefined;
    if (rs?.length && part.read) part.read.toks.push({ t: ev.delta as string, v: part.read.ids.map((_, d) => rs.reduce((a, r) => a + r[d], 0) / rs.length) });
    arrived(part);
    if (!raf) raf = requestAnimationFrame(() => ((raf = 0), updateLastBot()));
  });

  cancelAnimationFrame(raf);
  set({ streaming: false });

  for (const p of live) {
    const r = rateOf(p);
    if (r) p.wps = Math.round(r * 10) / 10;
  }

  const kept = parts(m).some((p) => p.content.trim());
  if (!kept) get().messages.pop();
  else if (!ok) m.stopped = true;
  render();
  if (kept) saveChat();

  lastMds().forEach((el, i) => decoders[i]?.update(el)); // let the last few letters finish decoding
  decoders = [];
  if (follow) toBottom(false);
  input()?.focus();
}

export function sendChat(text?: string) {
  text = (text ?? get().input).trim();
  if (!text || get().streaming) return;
  if (app().busy) return toast("Something is already running. Stop it first.", true);

  get().messages.push({ role: "user", content: text });
  flushSync(() => set({ input: "" }));
  growInput();
  reply();
}

export function retry() {
  if (get().streaming) return;
  get().messages.pop();
  reply();
}

/** Carry on writing a reply that ran out of length. */
export function continueReply() {
  const m = get().messages[get().messages.length - 1];
  if (get().streaming || !m || m.role !== "assistant" || !cutParts(m).length) return;
  if (app().busy) return toast("Something is already running. Stop it first.", true);
  reply(true);
}

export async function copyReply(text: string) {
  await navigator.clipboard.writeText(cleanReply(text));
}

export function growInput() {
  const t = input()!;
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 220) + "px";
}

export function newChat() {
  if (get().streaming) api("/api/cancel", {});
  set({ messages: [], id: null }); // the last one is already saved
  render();
  input()?.focus();
}

// ------------------------------------------------------------------ saved chats
export async function refreshChats() {
  try { set({ list: await api<ChatSummary[]>("/api/chats") }); } catch {}
}

/** Save the conversation, with the settings, switches and recipe it ran under. After every reply. */
async function saveChat() {
  const s = get(), a = app();
  const id = s.id ?? `c${Date.now()}`;
  if (!s.id) set({ id });

  const first = s.messages.find((m) => m.role === "user")?.content ?? "";

  try {
    await api("/api/chats", {
      id,
      title: first.length > 60 ? first.slice(0, 57).trimEnd() + "…" : first,
      model: a.info?.model ?? null,
      messages: s.messages,
      system: s.system, temp: s.temp, max: s.max, steer: s.steer, compare: s.compare,
      abl: ablActive(a),
      recipe: a.recipe.items.length ? a.recipe : null,
    });
    refreshChats();
  } catch (e) {
    toast("Couldn’t save the chat: " + (e as Error).message, true);
  }
}

type Saved = {
  id: string; messages: Msg[]; system: string; temp: string; max: string; steer: boolean; compare: boolean; abl: boolean;
  recipe: ReturnType<typeof app>["recipe"] | null;
};

/** Reopen a saved conversation as it ran: its settings, its switches, and the recipe that was on. */
export async function openChat(id: string) {
  if (get().streaming) return toast("A reply is still coming. Stop it first.", true);

  let c: Saved;
  try { c = await api<Saved>(`/api/chats/${id}`); } catch (e) { return toast((e as Error).message, true); }

  set({ id: c.id, messages: c.messages, system: c.system ?? "", temp: c.temp ?? "0.7", max: c.max ?? "512", steer: c.steer ?? true, compare: !!c.compare });
  if (c.recipe) useApp.setState({ recipe: c.recipe });
  if (app().abl?.ready && ablActive(app()) !== !!c.abl) setAblActive(!!c.abl);

  setTab("chat");
  render();
  toBottom(false);
}

export async function deleteChat(id: string) {
  try { await api(`/api/chats/${id}`, null, "DELETE"); } catch (e) { return toast((e as Error).message, true); }

  if (get().id === id) set({ id: null });
  refreshChats();
}

export const toggleSteer = () => set({ steer: !get().steer });

export function toggleCompare() {
  save("chat-compare", get().compare ? "0" : "1");
  set({ compare: !get().compare });
}

// ------------------------------------------------------------------ neurons by hand
export function toggleNeurons() {
  save("chat-neurons", get().neurons ? "0" : "1");
  set({ neurons: !get().neurons });
  if (get().neurons) requestAnimationFrame(() => document.getElementById("chat-add")?.focus());
}

/** Add the neurons typed in the panel ("L14 N8949 +1 and L13 N7956 -1") to the recipe that's on, and steer with it. */
export function addNeurons() {
  const { layers = 0, neurons = 0 } = app().info ?? {};
  const found = parsePins(get().add);
  if (!found) return set({ addErr: "Write each one as L14 N8949 +1." });

  const bad = found.find((p) => p.layer >= layers || p.neuron >= neurons);
  if (bad) {
    return set({ addErr: bad.layer >= layers ? `There’s no layer ${bad.layer}: this model has layers 0–${layers - 1}.`
      : `There’s no neuron ${bad.neuron}: each layer has neurons 0–${neurons - 1}.` });
  }

  addItems(found);
  set({ add: "", addErr: "", steer: true });
  found.forEach(fetchLens);
}

export function setPickBy(by: "type" | "map") {
  save("chat-pick-by", by);
  set({ pickBy: by });
  requestAnimationFrame(() => document.getElementById(by === "map" ? "chat-find" : "chat-add")?.focus());
}

/** The neurons whose push most directly raises a word, for the map. */
export async function findNeurons() {
  const word = get().find.trim();
  if (!word || get().finding) return;
  if (!isWord(word)) return set({ found: null, addErr: "Try a word or two, like evil or pirate." });

  set({ finding: true, addErr: "" });
  try {
    // the last ~10% of layers hold blunt "say this exact word" neurons; mid-late layers hold the concept (as in Steer)
    const skip = Math.max(2, Math.round((app().info?.layers ?? 0) / 9));
    const r = await api<WordSearch>(`/api/search?word=${encodeURIComponent(word)}&k=24&skip_last=${skip}`);
    set({ found: { ...r, word } });
  } catch (e) {
    set({ addErr: (e as Error).message });
  } finally {
    set({ finding: false });
  }
}

/** A neuron on the map, clicked: into the recipe at a dose its push is noticeable but not yet a loop, or out again. */
export function toggleFound(x: { layer: number; neuron: number; score: number }) {
  const i = app().recipe.items.findIndex((it) => itemKey(it) === `${x.layer}:${x.neuron}`);
  if (i >= 0) return removeAt(i);

  addItems([{ layer: x.layer, neuron: x.neuron, dose: searchDose(x.score) }]);
  set({ steer: true });
  fetchLens(x);
}

async function fetchLens(it: { layer: number; neuron: number }) {
  const k = `${it.layer}:${it.neuron}`;
  if (get().lens[k]) return;

  try {
    const r = await api<{ lens: { up: string[]; down: string[] } }>(`/api/neuron?layer=${it.layer}&n=${it.neuron}`);
    set({ lens: { ...get().lens, [k]: r.lens } });
  } catch {}
}

/** The recipe's neurons, each with where it sits in the recipe (its directions are Steer's to show). */
export const recipeNeurons = (items: ReturnType<typeof app>["recipe"]["items"]) =>
  items.flatMap((it, i) => (isDir(it) ? [] : [{ it: it as NeuronItem, i }]));

/** The Chat tab was opened, or a model got ready while it's open. */
export function onChatShown() {
  recipeNeurons(app().recipe.items).forEach(({ it }) => fetchLens(it));
  render();
  toBottom(false);
  requestAnimationFrame(() => input()?.focus());
}
