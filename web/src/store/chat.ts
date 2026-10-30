import { create } from "zustand";
import type { ChatMessage } from "../lib/types";

// The conversation. As on Steer, the reply being streamed is a mutable message, redrawn when `v` changes.
export type Msg = ChatMessage & { stopped?: boolean };

type Chat = {
  v: number;
  messages: Msg[];
  streaming: boolean;
  steer: boolean; // the recipe that's on applies to replies
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
  input: "",
  system: "",
  temp: "0.7",
  max: "512",
  jump: false,
  set: (p) => set(p),
}));

export const touchChat = () => useChat.setState((s) => ({ v: s.v + 1 }));
