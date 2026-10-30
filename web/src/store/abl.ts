import { create } from "zustand";
import type { Sink } from "../steer/sink";
import type { Live } from "./steer";

// Abliterate's own state. The saved direction itself (/api/abl) lives in the app store: the sidebar, Home and
// Chat read it too. As on Steer, the parts streams write to are mutable objects, redrawn when `v` changes.

/** An answer in the try-it thread: a question with refusals erased at a strength (null for the finder's examples,
 *  before it settles on one), and the normal answer made alongside it. `change` is the strength nudge that asked it. */
export type Entry = {
  prompt: string;
  coef: number | null;
  text: Sink;
  normal: Sink;
  refused: boolean | null;
  refusedNormal: boolean | null;
  worse: boolean; // writing got noticeably worse than the normal model's
  change: { from: number; to: number } | null;
  typing: boolean;
  open: boolean; // an older answer, unfolded
};

type Rank = { layer: number; score: number; tried: number; refused: number };
export type AblResult = {
  layer: number; coef: number; top_score: number; ranking: Rank[];
  tried: number; refused_before?: number; refused_left: number; // of the new bad requests, how many it refused before and after
};

type Abl = {
  v: number;
  harmful: string;
  harmless: string;
  dirty: boolean; // the probe lists were edited and not saved, so a refresh leaves them alone
  q: string; // the try-it box
  name: string;
  nameTouched: boolean;
  plan: {
    shown: boolean;
    goal: string;
    steps: string[] | null; // each step's class, once a run has set them
    widths: string[];
    scene: boolean;
    clock: string;
    live: Live;
    running: boolean; // Find it is disabled
    done: boolean; // finished: the steps fold away
  };
  rank: AblResult | null;
  rankN: number; // a new ranking is a new list
  at: Entry[] | null; // the try-it thread
};

export const useAbl = create<Abl>(() => ({
  v: 0,
  harmful: "",
  harmless: "",
  dirty: false,
  q: "",
  name: "",
  nameTouched: false,
  plan: { shown: false, goal: "", steps: null, widths: ["", "", "", ""], scene: false, clock: "0:00", live: null, running: false, done: false },
  rank: null,
  rankN: 0,
  at: null,
}));

export const touchAbl = () => useAbl.setState((s) => ({ v: s.v + 1 }));
