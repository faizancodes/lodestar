// Try it and refine: a short thread, read top to bottom like a chat: the question, the answer, then each thing you
// said should change and the answer after it. One box at the bottom does both jobs: once there's a steered answer it
// takes a fix ("what should change?"), which retunes the neurons and answers again; "New question" switches it back
// to asking. Older answers fold to a few lines and can be gone back to; the normal answer shows on request.
import { createElement, Fragment, type ReactNode } from "react";
import { flushSync } from "react-dom";
import { fmt, isDir, labelOf, pushes } from "../lib/format";
import { save } from "../lib/storage";
import { stream } from "../lib/stream";
import type { DirItem, Item, NeuronItem, NeuronRow } from "../lib/types";
import { ablActive, useApp } from "../store/app";
import { touch, useSteer, type Thread, type Version } from "../store/steer";
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
export const FIX_CHIPS = ["It repeats itself", "It's too strong", "It's too weak", "It went off topic"];

const snapshot = () => app().recipe.items.map((i) => ({ ...i }));
export const tlMode = (tl: Thread | null) => (tl && tl.versions.length && !tl.asking ? "fix" : "ask");
const scrollToLatest = () => requestAnimationFrame(() => document.querySelector("#tl .tl-a.cur")?.scrollIntoView({ behavior: "smooth", block: "nearest" }));

/** "2 neurons", "1 direction", "2 neurons, 1 direction". */
export function nOn(items: Item[]) {
  const nd = items.filter(isDir).length, nn = items.length - nd;
  return [nn && `${nn} neuron${nn === 1 ? "" : "s"}`, nd && `${nd} direction${nd === 1 ? "" : "s"}`].filter(Boolean).join(", ") || "0 neurons";
}

export function setCompare(on: boolean) {
  save("tl-compare", on ? "1" : "0");
  set({ compare: on });
}

/** The box was sent: a fix once there's a steered answer, otherwise a question. */
export function submit() {
  const text = get().tInput.trim();
  if (!text) return;
  if (app().busy) return toast("Something is already running. Stop it first.", true);
  if (tlMode(get().tl) === "fix") runFix(text);
  else send(text);
}

/** "New question" turns the box to asking; "Cancel" turns it back to fixing. */
export function newQuestion() {
  const tl = get().tl!;
  tl.asking = tlMode(tl) === "fix";
  set({ tInput: "" });
  touch();
  document.getElementById("t-prompt")?.focus();
}

/** Puts a question in the box, ready to ask (used by plans that are still running). */
export function setQuestion(prompt: string) {
  const tl = get().tl;
  if (tl) tl.asking = true;
  set({ tInput: prompt });
  touch();
}

/** Asks a question with the recipe that's on: a new question starts a new thread; the same one again (after
 *  changing the recipe by hand) adds an answer to it. */
export async function send(prompt: string) {
  prompt = prompt.trim();
  if (!prompt) return;
  if (app().busy) return toast("Something is already running. Stop it first.", true);
  const items = snapshot();
  const { recipe, dirs } = app();
  // with no recipe on there's nothing to refine: a one-off answer (normal, or with refusals erased)
  let tl = get().tl;
  const fresh = !tl || tl.prompt !== prompt || !items.length || !tl.versions.length;
  if (fresh || !tl) tl = { id: ++ids, prompt, normal: sink(), other: sink(), abl: ablActive(app()), versions: [], asking: false, fixing: null, fixLive: null, streaming: new Set() };
  tl.asking = false;
  const v: Version = { items, out: sink(), fix: null, note: fresh ? "" : "Asked again with the recipe on now", open: false };
  if (items.length) tl.versions.push(v);
  const cols = items.length ? (fresh ? ["base", "up"] : ["up"]) : tl.abl ? ["abl"] : ["base"];
  cols.forEach((c) => tl!.streaming.add(c === "base" && items.length ? "normal" : "main"));
  const t = tl;
  flushSync(() => set({ tl: t, tInput: "" })); // drawn before the first words stream into it
  touch();
  scrollToLatest();
  await stream("/api/steer", { prompt, max_new: 160, columns: cols, pushes: pushes(recipe.items, dirs, 1) }, (ev) => {
    if (ev.type !== "delta") return;
    const d = ev.delta as string;
    if (ev.col === "base" && items.length) streamInto(t.normal, d);
    else streamInto(items.length ? v.out : t.other, d);
  });
  t.streaming.clear();
  touch();
}

export function undo() {
  const tl = get().tl!;
  tl.versions.pop();
  addItems(tl.versions.at(-1)!.items.map((i) => ({ ...i })), true);
  touch();
}

export function goBack(i: number) {
  const tl = get().tl!, v = tl.versions[i];
  addItems(v.items.map((x) => ({ ...x })), true);
  const out = sink();
  out.text = v.out.text;
  tl.versions.push({ items: v.items.map((x) => ({ ...x })), out, fix: null, note: `Went back to v${i + 1}`, open: false });
  touch();
  scrollToLatest();
}

/** Opens the inside of an answer under it: a version (by index), or the one-off answer. */
export function inspectAnswer(i: number | null) {
  const tl = get().tl!, v = i === null ? null : tl.versions[i];
  const reply = (v ? v.out : tl.other).text;
  inspect(`tl:${tl.id}:${i ?? "x"}`, tl.prompt, reply, v ? pushes(v.items, app().dirs, 1) : [], v ? `v${i! + 1}` : tl.abl ? "Refusals erased" : "Normal", tl.abl);
}

// a fix: retune the neurons of the latest answer toward what the person asked for, then answer again
async function runFix(feedback: string) {
  const tl = get().tl!, v = tl.versions.at(-1)!;
  const prev = v.items.filter((i): i is NeuronItem => !isDir(i)).map((i) => ({ ...i }));
  if (!prev.length) return toast("Fixing retunes neurons, and none are on. Change a direction's strength instead.", true);
  const dirsOn = v.items.filter(isDir) as DirItem[];
  type Refined = { improved: boolean; note: string; recipe: (NeuronRow & { dose: number })[] };
  let label = "", result = null as Refined | null;
  const say = (text: string, running: boolean) => ((tl.fixLive = { text, running }), touch());
  set({ tInput: "" });
  tl.fixing = feedback;
  say("Starting…", true);
  requestAnimationFrame(() => document.querySelector("#tl .tl-now")?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
  const prompts = lines(get().prompts);
  const ok = await stream("/api/tune/refine", { recipe: prev, feedback, prompt: tl.prompt, bad: v.out.text, behavior: app().recipe.name || "", prompts }, (ev) => {
    if (ev.type === "stage") say((label = ev.label as string) + "…", true);
    else if (ev.type === "progress") say(`${label} · ${ev.done} of ${ev.total}`, true);
    else if (ev.type === "result") result = ev as unknown as Refined;
  });
  tl.fixing = null;
  tl.fixLive = null;
  const fail = !ok || !result ? "Stopped. Nothing was changed." : !result.improved ? result.note : "";
  if (fail) {
    set({ tInput: feedback });
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
  const nv: Version = { items: snapshot(), out: sink(), fix: { feedback, lines: out }, note: "", open: false };
  tl.versions.push(nv);
  tl.streaming.add("main");
  flushSync(touch);
  scrollToLatest();
  const { recipe, dirs } = app();
  await stream("/api/steer", { prompt: tl.prompt, max_new: 160, columns: ["up"], pushes: pushes(recipe.items, dirs, 1) }, (ev) => {
    if (ev.type === "delta") streamInto(nv.out, ev.delta as string);
  });
  tl.streaming.clear();
  touch();
}
