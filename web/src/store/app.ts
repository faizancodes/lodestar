import { create } from "zustand";

export const TABS = ["abliterate", "steer", "chat"] as const;
export type Tab = (typeof TABS)[number] | "home";

function remembered(): Tab {
  let t: string | null = null;
  try { t = localStorage.getItem("tab"); } catch {}
  return TABS.includes(t as never) || t === "home" ? (t as Tab) : "steer";
}

type App = {
  tab: Tab;
  setTab: (t: Tab) => void;
};

export const useApp = create<App>((set) => ({
  tab: remembered(),
  setTab: (tab) => {
    try { localStorage.setItem("tab", tab); } catch {}
    set({ tab });
  },
}));
