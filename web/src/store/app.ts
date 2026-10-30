import { create } from "zustand";
import type { Abl, CopyJob, Direction, Example, Models, Probes, Recipe, Status } from "../lib/types";

export const TABS = ["abliterate", "steer", "chat"] as const;
export type Tab = (typeof TABS)[number] | "home";

function remembered(): Tab {
  let t: string | null = null;
  try { t = localStorage.getItem("tab"); } catch {}
  return TABS.includes(t as never) || t === "home" ? (t as Tab) : "steer";
}

export type Toast = { msg: string; err: boolean; n: number }; // n changes on every toast, so a repeat shows again

type App = {
  tab: Tab;
  info: Status | null;
  models: Models;
  copy: CopyJob | null;
  recipe: Recipe;
  saved: Recipe[];
  examples: Example[];
  dirs: Direction[] | null;
  dirLayers: number | null;
  inspect: { tokens: string[]; n_prompt: number } | null; // the reply last opened in Inspect
  abl: Abl | null;
  ablDefaults: Probes | null;
  /** One stream at a time: the engine has one model. Shown as body.busy. */
  busy: boolean;
  toast: Toast | null;

  setTab: (t: Tab) => void;
  setBusy: (b: boolean) => void;
  toastMsg: (msg: string, err?: boolean) => void;
};

export const useApp = create<App>((set) => ({
  tab: remembered(),
  info: null,
  models: { presets: [], cached: [], local: [] },
  copy: null,
  recipe: { id: null, name: "", items: [] },
  saved: [],
  examples: [],
  dirs: null,
  dirLayers: null,
  inspect: null,
  abl: null,
  ablDefaults: null,
  busy: false,
  toast: null,

  setTab: (tab) => {
    try { localStorage.setItem("tab", tab); } catch {}
    set({ tab });
  },
  setBusy: (busy) => set({ busy }),
  toastMsg: (msg, err = false) => set((s) => ({ toast: { msg, err, n: (s.toast?.n ?? 0) + 1 } })),
}));

/** The saved refusal direction is on. */
export const ablActive = (s: Pick<App, "abl">) => !!s.abl?.ready && !!s.abl.current?.active;
