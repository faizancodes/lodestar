// The recipe: what's on now. Whenever the list is redrawn as a whole, any panel open under a row goes with it, so
// `redrawn` drops those too.
import { api } from "../lib/api";
import { isDir, itemKey, labelOf } from "../lib/format";
import type { Item, Recipe } from "../lib/types";
import { refreshSaved } from "../shell/engine";
import { useApp } from "../store/app";
import { useSteer } from "../store/steer";

const get = useApp.getState;
const set = useApp.setState;
const items = () => get().recipe.items;
const setItems = (items: Item[]) => set({ recipe: { ...get().recipe, items } });

export function redrawn() {
  useSteer.setState((s) => ({ recipeGen: s.recipeGen + 1, panels: s.panels.filter((p) => !p.anchor.startsWith("rc:")) }));
}

export function addItems(add: Item[], replace = false) {
  const next = replace ? [] : [...items()];
  const at = new Map(next.map((it, i) => [itemKey(it), i]));

  for (const it of add) {
    const ex = at.get(itemKey(it));
    if (ex !== undefined) next[ex] = { ...next[ex], dose: it.dose };
    else (at.set(itemKey(it), next.length), next.push({ ...it }));
  }

  setItems(next);
  redrawn();
}

export function removeAt(i: number) {
  setItems(items().filter((_, k) => k !== i));
  redrawn();
}

/** A change made on a row itself (its strength, its layers): the list isn't redrawn. */
export function patchAt(i: number, p: Partial<Item>) {
  setItems(items().map((it, k) => (k === i ? ({ ...it, ...p } as Item) : it)));
}

export function setOp(i: number, op: "push" | "erase") {
  const it = items()[i];
  if (!isDir(it) || it.op === op) return;

  patchAt(i, { op, dose: op === "erase" ? 1 : 4 });
  redrawn();
}

/** Start a recipe afresh, keeping nothing of the one on now. */
export function replaceRecipe(r: Recipe) {
  set({ recipe: r });
  redrawn();
}

export async function saveRecipe() {
  const { recipe, info } = get();
  const name = recipe.name || recipe.items.map(labelOf).slice(0, 3).join(" + ");
  const saved = await api<Recipe>("/api/recipes", { id: recipe.id || undefined, name, model: info?.model, items: recipe.items });

  set({ recipe: { ...get().recipe, id: saved.id, name: saved.name } });
  get().toastMsg(`Saved “${saved.name}”.`);
  refreshSaved();
}
