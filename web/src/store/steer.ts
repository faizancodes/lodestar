import type { ReactNode } from "react";
import { create } from "zustand";
import { load } from "../lib/storage";
import type { Item, NeuronRow, StreamEvent } from "../lib/types";
import type { Sink } from "../steer/sink";

// Steer's own state. The long-running parts (a plan, the try-it thread, the new-direction panel) are mutable objects
// the engine writes to as their streams arrive; `v` changes after each write so the
// views draw them again. The recipe itself lives in the app store: the sidebar and Chat read it too.

/** A status line: empty until first set, then the spinner and its text. */
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
  renders: number; // full redraws, which rebuild the stage steps (progress alone doesn't)
};
type Results = { n: number; word?: (NeuronRow & { score: number; dose: number })[]; plan?: Plan };

/** One answer to the thread's question, with the recipe that was on for it and how it came about. */
export type Version = {
  items: Item[];
  out: Sink;
  fix: { feedback: string; lines: ReactNode[] } | null; // the fix that made it
  note: string; // or why it's here otherwise ("Went back to v2")
  open: boolean; // an older answer, unfolded
};
/** Try it and refine: one question, and its answers as the recipe is fixed. */
export type Thread = {
  id: number; // panels opened under its answers are named by it, so a new thread starts with none
  prompt: string;
  normal: Sink; // the model as it normally is, shown on request
  other: Sink; // the one-off answer when no recipe is on: normal, or with refusals erased
  abl: boolean;
  versions: Version[];
  asking: boolean; // the box asks a new question instead of taking a fix
  fixing: string | null; // the fix being made
  fixLive: Live;
  streaming: Set<"main" | "normal">;
};

export type Dn = {
  open: boolean;
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
export type Panel = { id: number; anchor: string; kind: "inspect" | "neuron"; title: string; body: Body };

type Steer = {
  v: number;
  input: string;
  note: string; // #c-status
  results: Results | null;
  prompts: string; // Options → Test questions
  tInput: string; // the box under the thread: a question, or what should change
  tl: Thread | null;
  compare: boolean; // the normal answer shows under the latest one (here and on Abliterate)
  dn: Dn;
  exp: Export;
  panels: Panel[];
  recipeGen: number; // bumped when the recipe list is redrawn as a whole, which drops any panel open under a row
};

export const useSteer = create<Steer>(() => ({
  v: 0,
  input: "",
  note: "",
  results: null,
  prompts: "",
  tInput: "",
  tl: null,
  compare: load("tl-compare") === "1",
  dn: { open: false, typing: false, kind: "persona", name: "", pos: "", neg: "", path: "", live: null, result: null, made: null },
  exp: { page: null, line: "", width: "", running: false, saved: {} },
  panels: [],
  recipeGen: 0,
}));

/** Draw the mutable parts again after writing to them. */
export const touch = () => useSteer.setState((s) => ({ v: s.v + 1 }));
