import { create } from "zustand";
import type { Pt } from "../probe/map";

export type Metrics = {
  separation: number; words_only: number; shuffled: number; caught: number; layer: number; layers_picked: number[];
  new_groups?: number; words_only_new_groups?: number; per_group?: Record<string, number>;
};
export type ProbeRow = { id: string; name: string; labels: [string, string]; created: number; metrics: Metrics };
export type Probe = ProbeRow & { model: string; items: { text: string; side: 0 | 1; group: string | null }[]; map: Pt[][] };

type Prompt = string | { text: string; group?: string | null };
/** An experiment file (interp/experiment.py). */
export type Experiment = {
  lodestar: string; name: string; about?: string; model?: string;
  probe: { a: { label: string; prompts: Prompt[] }; b: { label: string; prompts: Prompt[] } };
  try?: string[]; expect?: Record<string, [number, number]>;
};
export type Check = { metric: string; want: [number, number]; got: number | null; ok: boolean };
export type Example = { file: string; name: string; about: string; model: string };

/** The question being asked of a probe: its score, its place at every layer, and Inkling's answer. */
type Test = { q: string; p: number | null; path: Pt[] | null; at: number; answer: string; asked: string[] };

type ProbeState = {
  list: ProbeRow[] | null;
  examples: Example[];
  form: { name: string; aLabel: string; a: string; bLabel: string; b: string };
  exp: Experiment | null; // the experiment the form came from, whose `expect` a run is checked against
  live: string;
  pct: number | null;
  cur: Probe | null;
  checks: Check[] | null;
  sameModel: boolean | null;
  test: Test;
};

export const useProbe = create<ProbeState>(() => ({
  list: null,
  examples: [],
  form: { name: "", aLabel: "", a: "", bLabel: "", b: "" },
  exp: null,
  live: "",
  pct: null,
  cur: null,
  checks: null,
  sameModel: null,
  test: { q: "", p: null, path: null, at: 0, answer: "", asked: [] },
}));
