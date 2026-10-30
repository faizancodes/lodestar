// Chat: talk to the loaded model, with the recipe that's on and the erased refusal direction switched on or off.
import { flushSync } from "react-dom";
import { setAblActive } from "../abl/engine";
import { createDecoder } from "../art/decode";
import { api } from "../lib/api";
import { cleanReply, pushes } from "../lib/format";
import { md } from "../lib/markdown";
import { save } from "../lib/storage";
import { stream } from "../lib/stream";
import { setTab, still } from "../shell/engine";
import { ablActive, useApp } from "../store/app";
import type { ChatSummary } from "../lib/types";
import { touchChat, useChat, type Col, type ColKey, type Msg } from "../store/chat";

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
const timing = new WeakMap<object, { t0: number; t1: number }>();
const words = (s: string) => (s.trim() ? s.trim().split(/\s+/).length : 0);

/** Words a second: measured from the first word, so the wait before it (reading the prompt) doesn't count. */
export function rateOf(part: { content: string; wps?: number }) {
  if (part.wps) return part.wps;

  const t = timing.get(part);
  if (!t || t.t1 - t.t0 < 500) return null;
  return (words(part.content) - 1) / ((t.t1 - t.t0) / 1000);
}

export const rateText = (r: number | null) => (r == null ? "" : `${r.toFixed(1)} words/s`);

function arrived(part: object) {
  const now = performance.now(), t = timing.get(part);
  if (t) t.t1 = now;
  else timing.set(part, { t0: now, t1: now });
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
const parts = (m: Msg): { content: string; wps?: number }[] => m.cols ?? [m];

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

async function reply() {
  const { recipe, dirs } = app();
  const cols = get().compare ? compareCols() : [];
  const many = cols.length > 1;

  const body = many
    ? {
        rows: cols.map((c) => ({ messages: history(c.key), pushes: c.key === "steer" ? pushes(recipe.items, dirs, 1) : [], abl: c.key === "abl" })),
        temperature: +get().temp,
        max_new: +get().max,
      }
    : {
        messages: history(soloKey()),
        pushes: get().steer ? pushes(recipe.items, dirs, 1) : [],
        abl: ablActive(app()),
        temperature: +get().temp,
        max_new: +get().max,
      };

  const m: Msg = { role: "assistant", content: "", ...(many && { cols: cols.map((c) => ({ ...c, content: "" })) }) };
  get().messages.push(m);
  set({ streaming: true });
  decoders = parts(m).map(() => createDecoder());
  render();
  follow = true;
  toBottom(true);

  let raf = 0;
  const ok = await stream("/api/chat", body, (ev) => {
    if (ev.type !== "delta") return;
    const part = parts(m)[many ? (ev.row as number) : 0];
    part.content += ev.delta as string;
    arrived(part);
    if (!raf) raf = requestAnimationFrame(() => ((raf = 0), updateLastBot()));
  });

  cancelAnimationFrame(raf);
  set({ streaming: false });

  for (const p of parts(m)) {
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

/** The Chat tab was opened, or a model got ready while it's open. */
export function onChatShown() {
  render();
  toBottom(false);
  requestAnimationFrame(() => input()?.focus());
}
