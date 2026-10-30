// Chat: talk to the loaded model, with the recipe that's on and the erased refusal direction switched on or off.
import { flushSync } from "react-dom";
import { createDecoder } from "../art/decode";
import { api } from "../lib/api";
import { cleanReply, pushes } from "../lib/format";
import { md } from "../lib/markdown";
import { stream } from "../lib/stream";
import { still } from "../shell/engine";
import { ablActive, useApp } from "../store/app";
import { touchChat, useChat, type Msg } from "../store/chat";

const app = useApp.getState;
const get = useChat.getState;
const set = useChat.setState;
const toast = (msg: string, err = false) => app().toastMsg(msg, err);

export const STARTERS = ["Explain how a rainbow forms", "Write a haiku about Mondays", "Give me three dinner ideas", "What would you do with a free afternoon?"];
export const DOTS = `<span class="dots"><i></i><i></i><i></i></span>`;
export const replyHtml = (m: Msg) => (m.content ? md(cleanReply(m.content)) : DOTS);

const scroller = () => document.getElementById("chat-scroll")!;
const input = () => document.getElementById("chat-input") as HTMLTextAreaElement | null;
const lastMd = () => document.querySelector<HTMLElement>("#chat-log .msg.bot:last-child .md");

// keep following the reply unless the user has scrolled up to read
const nearBottom = () => { const s = scroller(); return s.scrollHeight - s.scrollTop - s.clientHeight < 80; };
let follow = true;
export function onScroll() {
  follow = nearBottom();
  set({ jump: !follow });
}
const toBottom = (smooth: boolean) => scroller().scrollTo({ top: scroller().scrollHeight, behavior: smooth && !still() ? "smooth" : "auto" });
export const jumpDown = () => ((follow = true), toBottom(true));

let decoder: ReturnType<typeof createDecoder> | null = null; // the reply being written decodes in, mark by mark

function updateLastBot() {
  const el = lastMd();
  const m = get().messages[get().messages.length - 1];
  if (el && m.content) {
    el.innerHTML = replyHtml(m);
    decoder?.update(el);
  }
  if (follow) toBottom(false);
}

const render = () => flushSync(touchChat);

async function reply() {
  const sys = get().system.trim();
  const history = get().messages.slice(-24).map(({ role, content }) => ({ role, content }));
  const { recipe, dirs } = app();
  const body = {
    messages: (sys ? [{ role: "system", content: sys }] : []).concat(history),
    pushes: get().steer ? pushes(recipe.items, dirs, 1) : [],
    abl: ablActive(app()),
    temperature: +get().temp,
    max_new: +get().max,
  };
  const m: Msg = { role: "assistant", content: "" };
  get().messages.push(m);
  set({ streaming: true });
  decoder = createDecoder();
  render();
  follow = true;
  toBottom(true);
  let raf = 0;
  const ok = await stream("/api/chat", body, (ev) => {
    if (ev.type !== "delta") return;
    m.content += ev.delta as string;
    if (!raf) raf = requestAnimationFrame(() => ((raf = 0), updateLastBot()));
  });
  cancelAnimationFrame(raf);
  set({ streaming: false });
  if (!m.content.trim()) get().messages.pop();
  else if (!ok) m.stopped = true;
  render();
  const el = lastMd();
  if (el) decoder?.update(el); // let the last few letters finish decoding
  decoder = null;
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

export async function copyReply(m: Msg) {
  await navigator.clipboard.writeText(cleanReply(m.content));
}

export function growInput() {
  const t = input()!;
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 220) + "px";
}

export function newChat() {
  if (get().streaming) api("/api/cancel", {});
  set({ messages: [] });
  render();
  input()?.focus();
}

export const toggleSteer = () => set({ steer: !get().steer });

/** The Chat tab was opened, or a model got ready while it's open. */
export function onChatShown() {
  render();
  toBottom(false);
  requestAnimationFrame(() => input()?.focus());
}
