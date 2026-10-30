// Inline panels: inspect a reply, or a neuron. A panel opens right under the thing that was clicked (a reply, a
// neuron row), named by an anchor key, and clicking it again closes it.
import type { ReactNode } from "react";
import { useSteer, type Panel } from "../store/steer";

const get = useSteer.getState;
const set = useSteer.setState;
let ids = 0;

/** Each panel's element, for scrolling to it and for Esc (which closes the last one on the page). */
export const panelEls = new Map<number, HTMLElement>();

export const panelAt = (anchor: string) => get().panels.find((p) => p.anchor === anchor);
export const alive = (id: number) => get().panels.some((p) => p.id === id);

/** Opens a panel under `anchor`, or reuses the one there; returns its id, or null when this closed it. */
export function openPanel(anchor: string, title: string, kind: Panel["kind"]): number | null {
  const p = panelAt(anchor);
  if (p && p.kind === kind && p.title === title) {
    closePanel(p.id); // second click on the same thing closes it
    return null;
  }

  let id: number;
  if (p) {
    id = p.id;
    set({ panels: get().panels.map((x) => (x === p ? { ...x, kind, title } : x)) });
  } else {
    id = ++ids;
    set({ panels: [...get().panels, { id, anchor, kind, title, body: { node: null, n: 0, reveal: false } }] });
  }

  requestAnimationFrame(() => panelEls.get(id)?.scrollIntoView({ block: "nearest", behavior: "smooth" }));
  return id;
}

/** Sets a panel's contents: straight away, or swapped in with the reveal transition. */
export function setBody(id: number, node: ReactNode, reveal = false) {
  set({ panels: get().panels.map((p) => (p.id === id ? { ...p, body: { node, n: p.body.n + 1, reveal } } : p)) });
}

export const closePanel = (id: number) => set({ panels: get().panels.filter((p) => p.id !== id) });
export const closeAt = (anchor: string) => {
  if (panelAt(anchor)) set({ panels: get().panels.filter((p) => p.anchor !== anchor) });
};
export const closeWhere = (keep: (p: Panel) => boolean) => set({ panels: get().panels.filter(keep) });

/** Esc: the last panel on the page. */
export function closeLast() {
  const els = [...document.querySelectorAll<HTMLElement>(".panel")];
  const last = els.pop();
  for (const [id, el] of panelEls) if (el === last) return closePanel(id);
}
