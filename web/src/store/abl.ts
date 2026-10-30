import { create } from "zustand";
import type { Sink } from "../steer/sink";
import type { Live } from "./steer";

// Abliterate's own state. The saved direction itself (/api/abl) lives in the app store: the sidebar, Home and
// Chat read it too. As on Steer, the parts streams write to are mutable objects, redrawn when `v` changes.

/** A question with its reply before and after the refusal direction is erased, side by side. */
export type Pair = { prompt: string; outs: { base: Sink; abl: Sink }; verdict: { base?: boolean; abl?: boolean }; typing: boolean };
export type Rank = { layer: number; score: number; tried: number; refused: number };
export type AblResult = { layer: number; coef: number; tried: number; refused_left: number; top_score: number; ranking: Rank[] };

type Abl = {
  v: number;
  harmful: string;
  harmless: string;
  dirty: boolean; // the probe lists were edited and not saved, so a refresh leaves them alone
  testProbes: string;
  name: string;
  nameTouched: boolean;
  coef: string; // the strength slider
  plan: {
    shown: boolean;
    goal: string;
    steps: string[] | null; // each step's class, once a run has set them
    widths: string[];
    scene: boolean;
    clock: string;
    live: Live;
    running: boolean; // Find it is disabled
  };
  examples: Pair[];
  rank: AblResult | null;
  rankN: number; // a new ranking is a new list
  test: { pairs: Pair[]; summary: { rb: number; ra: number; n: number; note: string } | null };
};

export const useAbl = create<Abl>(() => ({
  v: 0,
  harmful: "",
  harmless: "",
  dirty: false,
  testProbes: "",
  name: "",
  nameTouched: false,
  coef: "1",
  plan: { shown: false, goal: "", steps: null, widths: ["", "", "", ""], scene: false, clock: "0:00", live: null, running: false },
  examples: [],
  rank: null,
  rankN: 0,
  test: { pairs: [], summary: null },
}));

export const touchAbl = () => useAbl.setState((s) => ({ v: s.v + 1 }));
