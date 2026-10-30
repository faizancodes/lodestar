import { create } from "zustand";
import type { ChatMessage } from "../lib/types";

type Chat = {
  messages: ChatMessage[];
  streaming: boolean;
  steer: boolean; // the recipe that's on applies to replies
  set: (p: Partial<Omit<Chat, "set">>) => void;
};

export const useChat = create<Chat>((set) => ({
  messages: [],
  streaming: false,
  steer: true,
  set: (p) => set(p),
}));
