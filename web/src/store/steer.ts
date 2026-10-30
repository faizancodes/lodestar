import type { ReactNode } from "react";
import { create } from "zustand";
import type { NeuronRow, StreamEvent } from "../lib/types";
import type { Sink } from "../steer/sink";

// Steer's own state. The long-running parts (a plan, a run, a fix, the new-direction panel) are mutable objects
// the engine writes to as their streams arrive, the way ui/app.js kept them; `v` changes after each write so the
// views draw them again. The recipe itself lives in the app store: the sidebar and Chat read it too.

/** A status line (setLive in ui/app.js): empty until first set, then the spinner and its text. */
export type Live = { text: ReactNode; running: boolean } | null;

export type Cand = NeuronRow & { sign: number; est: number; dose?: number; dj: number; dfluency?: number; tested?: boolean };
export type Plan = {
  goal: string;
  cands: Map<string, Cand>;
  recipe: (NeuronRow & { dose: number; dfluency?: number })[];
  stage: number;
  frac: number;
  example: string;
  examplePrompt: string;
  proof: { prompt?: string; baseline?: string; steered?: string };
  done: boolean;
  label?: string;
  noEffect?: boolean;
  ordinary?: boolean; // the example's heading was swapped for "The model's ordinary answer"; it stays swapped
  resume?: (() => void) | null;
  stopped?: string;
  result?: StreamEvent & { recipe: Plan["recipe"]; note?: string; hit_test?: number; hit_test0?: number };
  applied?: boolean;
  clock: string;
  renders: number; // full redraws, which rebuilt the stage steps in ui/app.js (progress alone didn't)
};
export type Results = { n: number; word?: (NeuronRow & { score: number; dose: number })[]; plan?: Plan };

export type Fix = {
  open: boolean;
  input: string;
  focus: number; // bumped to focus the input
  busy: boolean; // the Fix button is disabled while it runs
  live: Live;
  aim: Sink | null; // shown once the model starts writing what it's aiming for
  done: { lines: ReactNode[]; undo: () => void } | "undone" | null;
};
export type Col = "base" | "up" | "down" | "abl";
export type Run = {
  id: number;
  prompt: string;
  desc: string;
  cols: Col[];
  outs: Record<string, Sink>;
  done: boolean;
  inspect: (c: Col) => void;
  fix: Fix | null;
  fixAt: number; // when the fix box was made: it and an inspect panel sit after the replies newest first
};

export type Dn = {
  open: boolean;
  focus: number; // bumped to focus the name
  focusPath: number; // bumped to focus the import path
  typing: boolean; // the three tries are being written
  kind: "persona" | "topic";
  name: string;
  pos: string;
  neg: string;
  path: string;
  live: Live;
  result: { facts: [string, string][]; q: string; tries: Sink[] } | null;
  made: (StreamEvent & { entry?: { id: string; name: string }; band: number[]; prompt: string; auc: number[]; cos_split: number[] }) | null;
};

export type Export = { page: "abl" | "st" | null; line: string; width: string; running: boolean; saved: Partial<Record<"abl" | "st", { name: string; dst: string }>> };

export type Body = { node: ReactNode; n: number; reveal: boolean };
export type Panel = { id: number; anchor: string; kind: "inspect" | "neuron"; title: string; body: Body; at: number };

type Steer = {
  v: number;
  input: string;
  note: string; // #c-status
  results: Results | null;
  prompts: string; // Options → Test questions
  tPrompt: string;
  runs: Run[];
  dn: Dn;
  exp: Export;
  panels: Panel[];
  recipeGen: number; // bumped where ui/app.js redrew the recipe list, which dropped any panel open under a row
};

export const useSteer = create<Steer>(() => ({
  v: 0,
  input: "",
  note: "",
  results: null,
  prompts: "",
  tPrompt: "Describe the way you look.",
  runs: [],
  dn: { open: false, focus: 0, focusPath: 0, typing: false, kind: "persona", name: "", pos: "", neg: "", path: "", live: null, result: null, made: null },
  exp: { page: null, line: "", width: "", running: false, saved: {} },
  panels: [],
  recipeGen: 0,
}));

/** Draw the mutable parts again after writing to them. */
export const touch = () => useSteer.setState((s) => ({ v: s.v + 1 }));
