// Try it and refine: a conversation with the model as the recipe steers it, read top to bottom like a chat. Every
// question and reply stays, and the model sees the conversation so far. If a reply isn't right, the buttons under it
// say what's wrong ("too strong", or anything else): the neurons are retuned and the reply is written again as a new
// version, and the arrows beside it switch back. New chat starts over.
import { createElement, Fragment, type ReactNode } from "react";
import { flushSync } from "react-dom";
import { fmt, isDir, labelOf, pushes } from "../lib/format";
import { save } from "../lib/storage";
import { stream } from "../lib/stream";
import type { DirItem, Item, NeuronItem, NeuronRow } from "../lib/types";
import { ablActive, useApp } from "../store/app";
import { touch, useSteer, type Thread, type Turn, type Version } from "../store/steer";
import { lines } from "./describe";
import { inspect } from "./Found";
import { addItems } from "./recipe";
import { sink, streamInto } from "./sink";

const app = useApp.getState;
const get = useSteer.getState;
const set = useSteer.setState;
const toast = (msg: string, err = false) => app().toastMsg(msg, err);
let ids = 0;

export const PROMPTS = ["Describe the way you look.", "Hi! Who are you?", "What's your favorite food?", "Tell me about your weekend.", "Write a short poem about the ocean."];
export const FIX_CHIPS = ["It's too strong", "It's too weak", "It repeats itself", "It went off topic"];

const snapshot = () => app().recipe.items.map((i) => ({ ...i }));
const scrollToLatest = () => requestAnimationFrame(() => document.querySelector("#tl .tl-a.cur")?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
export const shownOf = (t: Turn) => t.versions[t.shown];

/** "2 neurons", "1 direction", "2 neurons, 1 direction". */
export function nOn(items: Item[]) {
  const nd = items.filter(isDir).length, nn = items.length - nd;
  return [nn && `${nn} neuron${nn === 1 ? "" : "s"}`, nd && `${nd} direction${nd === 1 ? "" : "s"}`].filter(Boolean).join(", ") || "0 neurons";
}

export function setCompare(on: boolean) {
  save("tl-compare", on ? "1" : "0");
  set({ compare: on });
}

/** The conversation before turn `upto`, as the model sees it: each question and the reply that's shown for it. */
function historyOf(tl: Thread, upto: number) {
  return tl.turns.slice(0, upto).flatMap((t) => [{ role: "user", content: t.prompt }, { role: "assistant", content: shownOf(t).out.text }]);
}

/** Writes a reply for turn `ti` into `v`, with the recipe that's on and the conversation before it. */
async function reply(tl: Thread, ti: number, v: Version, withNormal: boolean) {
  const turn = tl.turns[ti], steered = v.items.length > 0;
  const cols = steered ? (withNormal ? ["base", "up"] : ["up"]) : turn.abl ? ["abl"] : ["base"];
  cols.forEach((c) => tl.streaming.add(c === "base" && steered ? "normal" : "main"));
  flushSync(touch); // drawn before the first words stream into it
  scrollToLatest();

  const history = historyOf(tl, ti);
  const { recipe, dirs } = app();
  await stream("/api/steer", { prompt: turn.prompt, max_new: 160, columns: cols, pushes: pushes(recipe.items, dirs, 1), ...(history.length ? { history } : {}) }, (ev) => {
    if (ev.type !== "delta") return;
    streamInto(ev.col === "base" && steered ? turn.normal : v.out, ev.delta as string);
  });

  tl.streaming.clear();
  touch();
}

const busy = () => (app().busy ? (toast("Something is already running. Stop it first.", true), true) : false);

/** The box was sent: the next message in the conversation. */
export function submit() {
  const text = get().tInput.trim();
  if (text) send(text);
}

/** Puts a question in the box, ready to send (used by plans that are still running). */
export function setQuestion(prompt: string) {
  set({ tInput: prompt });
}

/** Asks a question with the recipe that's on, as the next message in the conversation. */
export async function send(prompt: string) {
  prompt = prompt.trim();
  if (!prompt || busy()) return;

  const tl = get().tl ?? { id: ++ids, turns: [], fixing: null, fixLive: null, other: false, streaming: new Set() };
  const v: Version = { items: snapshot(), out: sink(), fix: null, note: "" };
  tl.turns.push({ prompt, versions: [v], shown: 0, normal: sink(), abl: ablActive(app()) });
  tl.other = false;
  set({ tl, tInput: "", fixInput: "" });

  await reply(tl, tl.turns.length - 1, v, true);
}

/** Starts the conversation over. */
export function newChat() {
  if (busy()) return;
  set({ tl: null, tInput: "", fixInput: "" });
  document.getElementById("t-prompt")?.focus();
}

/** The latest question again, with the recipe that's on now: a new version of its reply. */
export async function retry() {
  const tl = get().tl!, ti = tl.turns.length - 1, turn = tl.turns[ti];
  if (busy()) return;

  const v: Version = { items: snapshot(), out: sink(), fix: null, note: "Asked again" };
  turn.versions.push(v);
  turn.shown = turn.versions.length - 1;
  await reply(tl, ti, v, false);
}

/** The arrows beside the latest reply: show another version of it, and put its neurons back on. */
export function showVersion(k: number) {
  const tl = get().tl!, turn = tl.turns.at(-1)!;
  if (k < 0 || k >= turn.versions.length || k === turn.shown || busy()) return;

  turn.shown = k;
  addItems(turn.versions[k].items.map((i) => ({ ...i })), true);
  touch();
}

/** Opens the inside of the latest reply under it. */
export function inspectAnswer() {
  const tl = get().tl!, ti = tl.turns.length - 1, turn = tl.turns[ti], v = shownOf(turn);
  inspect(`tl:${tl.id}:${ti}`, turn.prompt, v.out.text, pushes(v.items, app().dirs, 1), v.items.length ? nOn(v.items) : turn.abl ? "Refusals erased" : "Normal", turn.abl);
}

/** "Something else": the box for saying what's wrong in your own words. */
export function toggleOther() {
  const tl = get().tl!;
  tl.other = !tl.other;
  touch();
  if (tl.other) requestAnimationFrame(() => document.getElementById("tl-fix-input")?.focus());
}

/** What's wrong with the latest reply, from a button or in words. */
export function fixLatest(feedback: string) {
  feedback = feedback.trim();
  if (feedback && !busy()) runFix(feedback);
}

// a fix: retune the neurons of the latest answer toward what the person asked for, then answer again
async function runFix(feedback: string) {
  const tl = get().tl!, ti = tl.turns.length - 1, turn = tl.turns[ti], v = shownOf(turn);
  const prev = v.items.filter((i): i is NeuronItem => !isDir(i)).map((i) => ({ ...i }));
  if (!prev.length) return toast("Fixing retunes neurons, and none are on. Change a direction's strength instead.", true);

  const dirsOn = v.items.filter(isDir) as DirItem[];
  type Refined = { improved: boolean; note: string; recipe: (NeuronRow & { dose: number })[] };
  let label = "", result = null as Refined | null;
  const say = (text: string, running: boolean) => ((tl.fixLive = { text, running }), touch());

  tl.fixing = feedback;
  tl.other = false;
  say("Starting…", true);
  requestAnimationFrame(() => document.querySelector("#tl .tl-now")?.scrollIntoView({ behavior: "smooth", block: "nearest" }));

  const prompts = lines(get().prompts);
  const ok = await stream("/api/tune/refine", { recipe: prev, feedback, prompt: turn.prompt, bad: v.out.text, behavior: app().recipe.name || "", prompts }, (ev) => {
    if (ev.type === "stage") say((label = ev.label as string) + "…", true);
    else if (ev.type === "progress") say(`${label} · ${ev.done} of ${ev.total}`, true);
    else if (ev.type === "result") result = ev as unknown as Refined;
  });

  tl.fixing = null;
  tl.fixLive = null;
  const fail = !ok || !result ? "Stopped. Nothing was changed." : !result.improved ? result.note : "";
  if (fail) {
    touch();
    return toast(fail, !ok);
  }

  const r: Refined = result!;

  // what changed, neuron by neuron
  const k = (i: { layer: number; neuron: number }) => `${i.layer}:${i.neuron}`;
  const known = new Map(prev.map((i) => [k(i), i])), after = new Map(r.recipe.map((x) => [k(x), x]));
  const b = (it: Item) => createElement("b", null, labelOf(it));
  const out: ReactNode[] = [];
  const line = (...parts: ReactNode[]) => out.push(createElement(Fragment, { key: out.length }, ...parts));

  for (const i of prev) {
    const a = after.get(k(i));
    if (!a) line(b(i), " off");
    else if (Math.abs(a.dose - i.dose) > 0.005) line(b(i), ` ${fmt(i.dose)} → ${fmt(a.dose)}`);
  }
  for (const x of r.recipe) if (!known.has(k(x))) line("added ", b(x as NeuronItem), ` ${fmt(x.dose)}`);

  addItems([...r.recipe.map((x) => ({ ...(known.get(k(x)) || { lens: x.lens }), layer: x.layer, neuron: x.neuron, dose: x.dose })), ...dirsOn], true);

  const nv: Version = { items: snapshot(), out: sink(), fix: { feedback, lines: out }, note: "" };
  turn.versions.push(nv);
  turn.shown = turn.versions.length - 1;
  set({ fixInput: "" });
  await reply(tl, ti, nv, false);
}
