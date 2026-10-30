import { create } from "zustand";
import { load } from "../lib/storage";
import type { Msg } from "./chat";

// Neurons: a hand-picked set of neurons, each pushed by its own dose, and a conversation that runs with them on. As
// in Chat, the reply being streamed is a mutable message, redrawn when `v` changes.

/** One neuron and its push. `lens`: the words it pushes toward (and, pushed negative, away from), once fetched. */
export type Pin = { layer: number; neuron: number; dose: number; on: boolean; lens?: { up: string[]; down: string[] } };

type Neurons = {
  v: number;
  pins: Pin[];
  add: string; // the box new neurons are typed or pasted into
  err: string; // why the last thing added couldn't be
  messages: Msg[];
  streaming: boolean;
  input: string;
  compare: boolean; // each reply beside the plain model's
  set: (p: Partial<Omit<Neurons, "set">>) => void;
};

export const useNeurons = create<Neurons>((set) => ({
  v: 0,
  pins: [],
  add: "",
  err: "",
  messages: [],
  streaming: false,
  input: "",
  compare: load("neurons-compare") !== "0",
  set: (p) => set(p),
}));

export const touchNeurons = () => useNeurons.setState((s) => ({ v: s.v + 1 }));
