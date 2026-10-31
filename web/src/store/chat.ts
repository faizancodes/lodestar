import { create } from "zustand";
import { load } from "../lib/storage";
import type { ChatMessage, ChatSummary, WordSearch } from "../lib/types";

// The conversation. As on Steer, the reply being streamed is a mutable message, redrawn when `v` changes.
export type Msg = ChatMessage & { stopped?: boolean; cols?: Col[]; wps?: number; cut?: boolean }; // cut: wrote to the length limit

// Compare's side-by-side replies: the plain model, and each switch that was on, alone. Each column carries on its
// own thread, so a later turn's history takes that column's earlier replies.
export type ColKey = "plain" | "steer" | "abl";
export type Col = { key: ColKey; label: string; content: string; wps?: number; cut?: boolean }; // wps: words a second, once done

type Chat = {
  v: number;
  id: string | null; // the saved conversation this is, once its first reply is in
  list: ChatSummary[] | null; // every saved conversation, newest first
  messages: Msg[];
  streaming: boolean;
  steer: boolean; // the recipe that's on applies to replies
  compare: boolean; // replies come side by side: plain, and each switch that's on
  input: string;
  system: string;
  temp: string;
  max: string;
  jump: boolean; // scrolled up from the latest message
  neurons: boolean; // the panel for pushing neurons by hand is open
  add: string; // what's typed in its box
  addErr: string; // why the last thing typed there couldn't be added
  lens: Record<string, { up: string[]; down: string[] }>; // the words each neuron pushes toward, by "layer:neuron"
  pickBy: "type" | "map"; // neurons typed by number, or picked off a map of the ones behind a word
  find: string; // the word the map is for
  found: (WordSearch & { word: string }) | null;
  finding: boolean;
  set: (p: Partial<Omit<Chat, "set">>) => void;
};

export const useChat = create<Chat>((set) => ({
  v: 0,
  id: null,
  list: null,
  messages: [],
  streaming: false,
  steer: true,
  compare: load("chat-compare") === "1",
  input: "",
  system: "",
  temp: "0.7",
  max: "512",
  jump: false,
  neurons: load("chat-neurons") === "1",
  add: "",
  addErr: "",
  lens: {},
  pickBy: load("chat-pick-by") === "map" ? "map" : "type",
  find: "",
  found: null,
  finding: false,
  set: (p) => set(p),
}));

export const touchChat = () => useChat.setState((s) => ({ v: s.v + 1 }));
