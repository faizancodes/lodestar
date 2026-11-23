// Experiment files (interp/experiment.py): a probe or a steering setup as one `.lodestar.json` anyone can open.
import { api } from "../lib/api";
import type { Item } from "../lib/types";
import { loadExperiment } from "../probe/engine";
import { addItems, replaceRecipe } from "../steer/recipe";
import { useApp } from "../store/app";
import { useChat } from "../store/chat";
import type { Experiment } from "../store/probe";
import { useSteer } from "../store/steer";
import { refreshDirs, setTab } from "./engine";

const app = useApp.getState;
const toast = (m: string, err = false) => app().toastMsg(m, err);

/** Turn a steering experiment's pushes on, as a new recipe. False when the model can't take them. */
export async function openSteer(exp: Experiment) {
  const r = await api<{ items: Item[]; mismatched: string[] }>("/api/experiments/steer", exp).catch((e) => (toast(e.message, true), null));
  if (!r) return false;

  replaceRecipe({ id: null, name: exp.name, items: [], model: app().info?.model });
  addItems(r.items, true);
  if (r.items.some((it) => it.kind === "dir")) refreshDirs();

  const m = r.mismatched;
  if (m.length) toast(`${m.join(", ")} ${m.length > 1 ? "have" : "has"} different weights in this model, so the setup won't act the same.`, true);
  return true;
}

export async function openFile(f: File) {
  let exp: Experiment;
  try {
    exp = JSON.parse(await f.text());
    if (exp?.lodestar !== "experiment/1") throw new Error();
  } catch {
    return toast(`${f.name} isn't a Lodestar experiment.`, true);
  }
  await openExperiment(exp);
}

/** Open an experiment someone shared a link to: a GitHub file, a gist, or any address of the file. */
export async function openLink(url: string) {
  const exp = await api<Experiment>("/api/experiments/fetch", { url }).catch((e) => (toast(e.message, true), null));
  if (exp) await openExperiment(exp);
}

/** Open the experiment whose link is on the clipboard, if one is (the server reads it; WebKit won't hand the page a
 *  paste outside a text field). */
export async function openClipboard() {
  const exp = await api<Experiment | null>("/api/experiments/fetch", { clipboard: true }).catch((e) => (toast(e.message, true), null));
  if (exp) await openExperiment(exp);
}

async function openExperiment(exp: Experiment) {
  if (!exp.steer) {
    loadExperiment(exp);
    return setTab("probe");
  }
  if ((await openSteer(exp)) && app().tab !== "chat") setTab("steer"); // in Chat it steers the replies, so stay
}

/** Write the setup on now as an experiment file, with the questions asked of it in this tab as its `try`. */
export async function saveSteer() {
  const { recipe, tab } = app();
  const asked = tab === "chat"
    ? useChat.getState().messages.filter((m) => m.role === "user").map((m) => m.content)
    : useSteer.getState().tl?.turns.map((t) => t.prompt) ?? [];
  const name = recipe.name || "Steering";

  const r = await api<{ path: string }>("/api/recipes/experiment", { name, items: recipe.items, try: [...new Set(asked)].slice(0, 6) })
    .catch((e) => (toast(e.message, true), null));
  if (r) toast(`Saved to ${r.path.replace(/^\/Users\/[^/]+/, "~")}`);
}
