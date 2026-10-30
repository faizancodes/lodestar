// Ask it something: the reply with the recipe on beside the model as it normally is, and fixing a steered reply
// by saying what's wrong with it. Carried over from ui/app.js.
import { createElement, Fragment, type ReactNode } from "react";
import { flushSync } from "react-dom";
import { fmt, isDir, labelOf, pushes } from "../lib/format";
import { stream } from "../lib/stream";
import type { DirItem, Item, NeuronItem, NeuronRow } from "../lib/types";
import { still } from "../shell/engine";
import { ablActive, useApp } from "../store/app";
import { touch, useSteer, type Col, type Fix, type Run } from "../store/steer";
import { lines } from "./describe";
import { inspect } from "./Found";
import { panelAt } from "./panels";
import { addItems } from "./recipe";
import { sink, streamInto } from "./sink";

const app = useApp.getState;
const get = useSteer.getState;
const set = useSteer.setState;
const toast = (msg: string, err = false) => app().toastMsg(msg, err);
let ids = 0;

export const PROMPTS = ["Describe the way you look.", "Hi! Who are you?", "What's your favorite food?", "Tell me about your weekend.", "Write a short poem about the ocean."];
export const NAMES: Record<Col, string> = { base: "Normal", up: "Changed", down: "Opposite", abl: "Ablated" };

export async function send() {
  const prompt = get().tPrompt.trim();
  if (!prompt || app().busy) return;
  const { recipe, dirs } = app();
  const up = pushes(recipe.items, dirs, 1), down = pushes(recipe.items, dirs, -1);
  const withAb = ablActive(app());
  // when ablation is on: the changed column carries it (the server decides), and with no neurons on, the
  // comparison reads as Normal beside the ablated model
  const opposite = (document.getElementById("t-opposite") as HTMLInputElement).checked;
  const cols: Col[] = up.length ? (["base", "up"] as Col[]).concat(opposite ? ["down"] : []) : withAb ? ["base", "abl"] : ["base"];
  const desc = up.length ? recipe.items.map(labelOf).slice(0, 3).join(" + ") : withAb ? "refusal direction erased" : "nothing changed";
  const run: Run = {
    id: ++ids, prompt, desc, cols, outs: Object.fromEntries(cols.map((c) => [c, sink()])), done: false, fix: null, fixAt: 0,
    inspect: (c) => inspect(run.id, prompt, run.outs[c].text, c === "up" ? up : c === "down" ? down : [], NAMES[c],
      c === "abl" || (c === "up" && withAb), !!run.fix && run.fixAt > (panelAt(`run:${run.id}`)?.at ?? Infinity)),
  };
  flushSync(() => set({ runs: [run, ...get().runs] })); // drawn before the first words stream into it
  const body = { prompt, max_new: 160, columns: cols, pushes: up.map((p) => ({ ...p })) };
  await stream("/api/steer", body, (ev) => { if (ev.type === "delta") streamInto(run.outs[ev.col as string], ev.delta as string); });
  run.done = true;
  touch();
}

// ------------------------------------------------------------------ fix a steered reply
// Say what's wrong with a Changed reply: the model writes the answer it should have given, the neurons that are on
// are retuned toward it (turned up, down or off, or new ones added), and the question runs again. Undo puts the
// neurons back.
export const FIX_CHIPS = ["It repeats itself", "It's too strong", "It's too weak", "It went off topic"];

export function openFix(run: Run) {
  const f = run.fix;
  if (f) {
    f.open = !f.open;
    if (f.open) f.focus++;
  } else {
    run.fix = { open: true, input: "", focus: 1, busy: false, live: null, aim: null, done: null };
    run.fixAt = performance.now();
  }
  touch();
}

export async function runFix(run: Run, fix: Fix) {
  const feedback = fix.input.trim(), prompt = run.prompt, bad = run.outs.up.text;
  if (!feedback) return (fix.focus++, touch());
  const prev = app().recipe.items.filter((i): i is NeuronItem => !isDir(i)).map((i) => ({ ...i }));
  if (!prev.length) return toast("Fix retunes neurons, and none are on. Change a direction's strength instead.", true);
  const dirsOn = () => app().recipe.items.filter(isDir) as DirItem[];
  const say = (text: ReactNode, running: boolean) => ((fix.live = { text, running }), touch());
  fix.aim = null;
  fix.done = null;
  fix.busy = true;
  type Refined = { improved: boolean; note: string; recipe: (NeuronRow & { dose: number })[] };
  let label = "", result = null as Refined | null;
  say("Starting…", true);
  const prompts = lines(get().prompts);
  const ok = await stream("/api/tune/refine", { recipe: prev, feedback, prompt, bad, behavior: app().recipe.name || "", prompts }, (ev) => {
    if (ev.type === "stage") say((label = ev.label as string) + "…", true);
    else if (ev.type === "progress") say(`${label} · ${ev.done} of ${ev.total}`, true);
    else if (ev.type === "aim") {
      if (!fix.aim) ((fix.aim = sink()), flushSync(touch));
      streamInto(fix.aim, ev.delta as string);
    } else if (ev.type === "result") result = ev as unknown as Refined;
  });
  fix.busy = false;
  touch();
  if (!ok || !result) return say("Stopped. Nothing was changed.", false);
  const r: Refined = result;
  if (!r.improved) return say(r.note, false);

  // what changed, neuron by neuron, from before the fix to after it
  const k = (i: { layer: number; neuron: number }) => `${i.layer}:${i.neuron}`;
  const known = new Map(prev.map((i) => [k(i), i])), after = new Map(r.recipe.map((x) => [k(x), x]));
  const b = (it: Item) => createElement("b", null, labelOf(it));
  const out: ReactNode[] = [];
  const line = (...parts: ReactNode[]) => out.push(createElement(Fragment, null, ...parts));
  for (const i of prev) {
    const a = after.get(k(i));
    if (!a) line("Turned off ", b(i));
    else if (Math.abs(a.dose - i.dose) > 0.005) line(`${Math.abs(a.dose) < Math.abs(i.dose) ? "Turned down" : "Turned up"} `, b(i), ` from ${fmt(i.dose)} to ${fmt(a.dose)}`);
  }
  for (const x of r.recipe) if (!known.has(k(x))) line("Added ", b(x as NeuronItem), ` at ${fmt(x.dose)}`);
  addItems([...r.recipe.map((x) => ({ ...(known.get(k(x)) || { lens: x.lens }), layer: x.layer, neuron: x.neuron, dose: x.dose })), ...dirsOn()], true);
  say("Fixed. The same question runs again above, with the new neurons.", false);
  fix.done = {
    lines: out,
    undo: () => {
      addItems([...prev, ...dirsOn()], true);
      fix.done = "undone";
      touch();
    },
  };
  touch();
  set({ tPrompt: prompt });
  await send();
  document.querySelector("#runs .run")?.scrollIntoView({ behavior: still() ? "auto" : "smooth", block: "start" });
}
