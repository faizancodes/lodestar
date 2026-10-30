import { create } from "zustand";
import { load } from "../lib/storage";
import type { ChatMessage } from "../lib/types";

// The conversation. As on Steer, the reply being streamed is a mutable message, redrawn when `v` changes.
export type Msg = ChatMessage & { stopped?: boolean; cols?: Col[] };

// Compare's side-by-side replies: the plain model, and each switch that was on, alone. Each column carries on its
// own thread, so a later turn's history takes that column's earlier replies.
export type ColKey = "plain" | "steer" | "abl";
export type Col = { key: ColKey; label: string; content: string };

type Chat = {
  v: number;
  messages: Msg[];
  streaming: boolean;
  steer: boolean; // the recipe that's on applies to replies
  compare: boolean; // replies come side by side: plain, and each switch that's on
  input: string;
  system: string;
  temp: string;
  max: string;
  jump: boolean; // scrolled up from the latest message
  set: (p: Partial<Omit<Chat, "set">>) => void;
};

export const useChat = create<Chat>((set) => ({
  v: 0,
  messages: [],
  streaming: false,
  steer: true,
  compare: load("chat-compare") === "1",
  input: "",
  system: "",
  temp: "0.7",
  max: "512",
  jump: false,
  set: (p) => set(p),
}));

export const touchChat = () => useChat.setState((s) => ({ v: s.v + 1 }));
