// The shell's side of talking to the engine: the model list and copies, loading a
// model and following its status, and what to fetch once one is ready. Views read what this writes to the store.
import { flushSync } from "react-dom";
import { finishScene, setProgress, startScene, stopScene, wake } from "../art/loader";
import { api } from "../lib/api";
import { modelName, sleep } from "../lib/format";
import type { Abl, CopyJob, Directions, Example, ExportJob, Fit, Memory, Models, Recipe, Status } from "../lib/types";
import { useApp, type Tab } from "../store/app";
import { onChatShown, refreshChats } from "../chat/engine";
import { useChat } from "../store/chat";
import { resetAbl, syncAbl } from "../abl/engine";
import { useSteer } from "../store/steer";
import { pollExport } from "../steer/exporting";
import { redrawn } from "../steer/recipe";

const get = useApp.getState;
const set = useApp.setState;
const toast = (msg: string, err = false) => get().toastMsg(msg, err);
export const still = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = <T extends Element = HTMLElement>(sel: string, root: ParentNode = document) => root.querySelector<T>(sel);
const $$ = <T extends Element = HTMLElement>(sel: string, root: ParentNode = document) => [...root.querySelectorAll<T>(sel)];

// ------------------------------------------------------------------ models
export async function refreshModels() {
  set({ models: await api<Models>("/api/models") });
}

export type Choice = {
  id: string; name: string; sub: string; tag?: string;
  copy?: { path: string; name: string; size: number }; drive?: { size: number };
};

export function modelChoices(m: Models, job: CopyJob | null) {
  const copying = (x: { path: string }) => job?.status === "copying" && job.src === x.path;
  const cached = new Set(m.cached), notes = new Map(m.presets.map((p) => [p.id, p.note]));
  return {
    here: [
      ...(m.copies || []).map((x): Choice => ({ id: x.path, name: x.name, sub: "On this Mac", copy: x })),
      ...m.cached.map((id): Choice => ({ id, name: modelName(id), sub: notes.get(id) || id })),
    ],
    // a drive model that has a copy on this Mac is listed once, as the copy
    drives: (m.local || []).filter((x) => !x.copied).map((x): Choice => ({
      id: x.path, name: x.name, drive: x,
      sub: copying(x) ? `Copying to this Mac… ${Math.floor((100 * job!.done) / job!.total)}%` : `On ${x.drive}`,
    })),
    get: m.presets.filter((p) => !cached.has(p.id)).map((p): Choice => ({ id: p.id, name: modelName(p.id), sub: p.note, tag: p.size })),
  };
}

export async function startCopy(path: string) {
  try { set({ copy: await api<CopyJob>("/api/models/copy", { path }) }); } catch (e) { return toast((e as Error).message, true); }
  pollCopy();
}

let copyPolling = false;
async function pollCopy() {
  if (copyPolling) return;
  copyPolling = true;

  try {
    for (;;) {
      let copy: CopyJob;
      try { copy = await api<CopyJob>("/api/models/copy"); } catch { await sleep(1500); continue; }
      set({ copy });

      if (copy.status !== "copying") {
        if (copy.status === "done") toast(`Copied ${copy.name}. It loads from this Mac now.`);
        else if (copy.status.startsWith("error")) toast(copy.status.replace(/^error: /, "Couldn’t copy: "), true);
        await refreshModels();
        break;
      }

      await sleep(1000);
    }
  } finally {
    copyPolling = false;
  }
}

export const cancelCopy = () => api("/api/models/copy/cancel", {});

export async function removeCopy(path: string) {
  try { await api("/api/models/remove", { path }); } catch (err) { return toast((err as Error).message, true); }
  toast("Removed the copy. The model still loads from the drive.");
  refreshModels();
}

export function toggleModelMenu(open?: boolean) {
  open = open ?? !get().menu.open;
  if (!open) return set({ menu: { ...get().menu, open: false } });

  const r = $("#model-pick")!.getBoundingClientRect(), side = $(".side")!.getBoundingClientRect();
  set({ menu: { open: true, left: side.right + 10, bottom: Math.max(12, innerHeight - r.bottom - 12), maxHeight: Math.min(620, r.bottom - 12) } });
  refreshModels(); // a drive may have been plugged in since
}

// ------------------------------------------------------------------ tabs
export function setTab(t: Tab) {
  // with no model there's nothing to show yet: remember where they're going, and point at the models (before the
  // first status arrives, it's just the tab to show)
  const info = get().info;
  if (info && !info.loaded) return set({ want: t === "home" ? null : t, nudge: get().nudge + 1 });

  get().setTab(t); // Steer's status card redraws its scene when it's shown
  if (t === "chat" && get().info?.loaded) onChatShown();
}

// ------------------------------------------------------------------ loading
const setStart = (p: Partial<ReturnType<typeof get>["start"]>) => set({ start: { ...get().start, ...p } });
const startArt = () => $<HTMLCanvasElement>("#start-art")!;

export async function loadModel(id: string, anyway = false) {
  toggleModelMenu(false);

  // a model that won't fit beside what's running makes macOS swap, and everything crawls: say so first
  if (!anyway) {
    const fit = await api<Fit>("/api/memory/fit", { model: id }).catch(() => null);
    if (fit && !fit.fits) return set({ fitWarn: { id, fit } });
  }

  const entered = enterLoading(id);

  let s: Status;
  try {
    s = await api<Status>("/api/load", { model: id });
  } catch (e) {
    await entered;
    setStart({ loading: false, picked: null });
    return toast((e as Error).message, true);
  }

  await entered;
  applyStatus(s);
  pollStatus();
}

export async function unloadModel() {
  toggleModelMenu(false);

  try {
    applyStatus(await api<Status>("/api/unload", {}));
  } catch (e) {
    return toast((e as Error).message, true);
  }

  refreshMemory();
}

// ------------------------------------------------------------------ memory
export async function refreshMemory() {
  try { set({ memory: await api<Memory>("/api/memory") }); } catch {}
}

// every few seconds while the window is in front; it's what shows a reply slowing because the Mac is swapping
async function pollMemory() {
  for (;;) {
    if (!document.hidden) await refreshMemory();
    await sleep(4000);
  }
}

// Picking a model from the start screen: the chosen card lights up, the title and the rest of the list fall
// away, and the hero opens from its banner crop into the full scene as it glides to the middle, where the sun
// wakes with a ring of light and starts to rise with the load.
async function enterLoading(id: string) {
  const start = $("#start")!;
  if (start.hidden || get().start.loading) return;
  if (still()) return setStart({ loading: true });

  const art = startArt(), cards = $$(".start-model", start);
  const picked = cards.find((c) => c.dataset.id === id);
  if (picked) flushSync(() => setStart({ picked: id }));

  const rest = [$("h2", start)!, ...cards.filter((c) => c !== picked)];
  const fall = (el: Element, delay: number) => el.animate([{ opacity: 1, transform: "none" }, { opacity: 0, transform: "translateY(12px)" }],
    { duration: 280, delay, easing: "cubic-bezier(0.4, 0, 1, 1)", fill: "forwards" });
  const outs = rest.map((el, i) => fall(el, i * 40));
  if (picked) outs.push(fall(picked, 380));
  await Promise.all(outs.map((a) => a.finished));

  const first = art.getBoundingClientRect();
  flushSync(() => setStart({ loading: true }));
  const last = art.getBoundingClientRect();

  outs.forEach((a) => a.cancel()); // they're hidden now; leave them ready for next time
  setStart({ picked: null });

  // the banner was a band of the full scene (object-position 40%), so open it from that band
  const off = (last.height - first.height) * 0.4, below = last.height - first.height - off;
  art.animate([
    { transform: `translateY(${first.top - last.top - off}px)`, clipPath: `inset(${off}px 0 ${below}px 0)` },
    { transform: "none", clipPath: "inset(0 0 0 0)" },
  ], { duration: 950, easing: "cubic-bezier(0.65, 0, 0.2, 1)" });
  setTimeout(wake, 650);
}

let loadStart: number | null = null, loadTimer: ReturnType<typeof setInterval> | undefined, finishing = false;

/** What the sidebar's model card says for a status. */
export function statusLine(s: Status | null) {
  if (!s) return { cls: "", msg: "No model loaded" };
  const err = s.status.startsWith("error");

  return {
    cls: s.loaded ? "ready" : err ? "err" : s.status === "idle" ? "" : "busy",
    msg: s.loaded ? "Ready" : s.status === "idle" ? "No model loaded"
      : err ? s.status.replace(/^error: /, "Couldn’t load: ") : `Loading ${modelName(s.status.replace(/^loading /, ""))}…`,
  };
}

function applyStatus(s: Status) {
  // let the loading scene finish its sunrise and pulse before the workspace appears
  if (s.loaded && get().start.loading) {
    if (!finishing) {
      finishing = true;
      clearInterval(loadTimer);
      setStart({ status: "Ready" });
      finishScene().then(async () => {
        if (!still()) {
          setStart({ leaving: true });
          await sleep(240);
        }

        setStart({ loading: false, leaving: false });
        finishing = false;
        flushSync(() => applyStatus(s));
        setTab(get().want ?? "home"); // a fresh model opens on Home, or the tab picked while there was none
        set({ want: null });

        const els = $$(".view:not([hidden]) > *");
        els.forEach((el) => el.classList.remove("reveal"));
        void document.body.offsetWidth; // restart the animations, with one reflow for all of them
        els.forEach((el, i) => {
          el.style.animationDelay = i * 110 + "ms";
          el.classList.add("reveal");
        });
      });
    }
    return;
  }

  const was = get().info?.loaded ? get().info!.model : null;
  set({ info: s });

  const { msg } = statusLine(s);
  const err = s.status.startsWith("error");
  const busy = !s.loaded && !err && s.status !== "idle";

  // the server can be slow to answer while weights load, so the timer ticks on its own
  clearInterval(loadTimer);
  if (busy) {
    loadStart ??= Date.now();
    const show = () => {
      const secs = Math.floor((Date.now() - loadStart!) / 1000);
      setStart({ status: `${msg} ${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}` });
    };
    show();
    loadTimer = setInterval(show, 1000);
  } else {
    loadStart = null;
    setStart({ status: s.loaded || s.status === "idle" ? "" : msg });
  }

  setStart({ loading: busy });

  if (!s.loaded) { // the start screen's scene: at rest on the horizon, rising with the load
    startScene(startArt());
    if (busy) setProgress(s.progress ?? 0);
  } else stopScene();

  if (s.loaded && s.model !== was) onModelReady();
}

let polling = false;
async function pollStatus() {
  if (polling) return;
  polling = true;

  try {
    for (;;) {
      let s: Status;
      try { s = await api<Status>("/api/status"); } catch { await sleep(1000); continue; }
      applyStatus(s);

      if (s.loaded || s.status === "idle" || s.status.startsWith("error")) {
        if (s.status.startsWith("error")) toast(s.status, true);
        refreshModels();
        break;
      }

      await sleep(700);
    }
  } finally {
    polling = false;
  }
}

// ------------------------------------------------------------------ once a model is ready
export function newRecipe() {
  set({ recipe: { id: null, name: "", items: [], model: get().info?.model } });
  redrawn();
  refreshSaved();
}

async function onModelReady() {
  const { recipe, info } = get();
  if (recipe.items.length && recipe.model && modelName(recipe.model) !== modelName(info!.model)) newRecipe(); // neuron numbers are per model

  set({ recipe: { ...get().recipe, model: info!.model } });
  useChat.getState().set({ messages: [] });
  redrawn();
  resetAbl();
  if (get().tab === "chat") onChatShown();

  refreshSaved();
  refreshAbl();
  refreshDirs();
  set({ examples: await api<Example[]>("/api/examples").catch((): Example[] => []) });
}

export async function refreshSaved() {
  try { set({ saved: await api<Recipe[]>("/api/recipes") }); } catch { return; }
}

export async function deleteSaved(id: string) {
  await api(`/api/recipes/${id}`, null, "DELETE");
  if (get().recipe.id === id) set({ recipe: { ...get().recipe, id: null } });
  refreshSaved();
}

export function openSaved(r: Recipe) {
  set({ recipe: JSON.parse(JSON.stringify(r)) });
  refreshSaved();
  if (get().tab !== "chat") setTab("steer"); // in Chat it steers the replies, so stay
}

export async function refreshAbl() {
  let abl: Abl;
  try { abl = await api<Abl>("/api/abl"); } catch { return; }
  set({ abl, ablDefaults: abl.defaults });
  syncAbl(abl);
}

export async function refreshDirs() {
  try {
    const r = await api<Directions>("/api/directions");
    set({ dirs: r.directions, dirLayers: r.layers });
  } catch { set({ dirs: [] }); }

  redrawn();
}

// ------------------------------------------------------------------ launch
// At launch, open the model that was loaded last time, if it's still on this Mac or a plugged-in drive.
function resumeLastModel() {
  const s = get().info!, c = modelChoices(get().models, get().copy);
  if (s.loaded || s.status !== "idle" || !s.last) return;
  if ([...c.here, ...c.drives].some((x) => x.id === s.last)) loadModel(s.last);
}

let booted = false;
export async function boot() {
  if (booted) return; // StrictMode runs effects twice in development
  booted = true;

  try {
    const [defaults, status, copy, ex] = await Promise.all([
      api<{ prompts: string[] }>("/api/tune/defaults"),
      api<Status>("/api/status"),
      api<CopyJob>("/api/models/copy"),
      api<ExportJob>("/api/abl/export"),
      refreshModels(),
      refreshAbl(),
    ]);

    useSteer.setState({ prompts: defaults.prompts.join("\n") });
    applyStatus(status);
    pollMemory();
    refreshChats();
    resumeLastModel();

    if (copy.status === "copying") pollCopy();
    if (ex.status === "working") pollExport(ex.kind === "steer" ? "st" : "abl");
    if (!get().info!.loaded && get().info!.status !== "idle") pollStatus();
  } catch (e) {
    toast("Can’t reach the Lodestar engine: " + (e as Error).message, true);
  }
}
