// The Architecture page's state and moves: where the 3D view is focused, the text last run through the model, and
// what's picked. The scene itself lives in scene.ts (loaded with three.js only when the page first opens).
import { create } from "zustand";
import { api } from "../lib/api";
import { stream } from "../lib/stream";
import type { ArchActs, ArchDetail, ArchLive, ArchNeuron, ArchRun } from "../lib/types";
import { setTab } from "../shell/engine";
import { useApp } from "../store/app";
import type { Focus, Part, Pick, Scene } from "./scene";

export type Acts = { q: Int8Array; peak: number[]; top: ArchActs["top"] };

type State = {
  model: string | null; // whose detail this is
  detail: ArchDetail | null;
  norms: (number[] | undefined)[];
  focus: Focus;
  text: string;
  running: boolean;
  reply: string | null; // the reply as it's written, until the run that traces it arrives
  run: ArchRun | null;
  token: number;
  acts: Acts | null;
  neuron: ArchNeuron | null;
  head: { layer: number; head: number } | null;
  hover: { pick: Pick; x: number; y: number } | null;
};

export const useArch = create<State>(() => ({
  model: null, detail: null, norms: [], focus: { level: "model" },
  text: "What is the capital of Japan?",
  running: false, reply: null, run: null, token: 0, acts: null, neuron: null, head: null, hover: null,
}));

const get = useArch.getState, set = useArch.setState;
let scene: Scene | null = null;

export function attach(s: Scene | null) {
  scene = s;
  if (!s) return;
  const { norms, run, acts, token, focus } = get();
  norms.forEach((n, l) => n && s.setNorms(l, n));
  if (run) (s.setRun(run), s.setToken(token, acts?.q ?? null, acts?.peak ?? null));
  if (focus.level !== "model") s.setFocus(focus, 0);
  marks();
}

/** The loaded model's structure, fetched once per model, then each layer's neuron sizes in the background. */
export async function loadDetail() {
  const mid = useApp.getState().info?.model ?? null;
  if (!mid || (get().model === mid && get().detail)) return get().detail;
  set({ model: mid, detail: null, norms: [], focus: { level: "model" }, run: null, acts: null, neuron: null, head: null });
  const d = await api<ArchDetail>("/api/arch");
  if (useApp.getState().info?.model !== mid) return null;
  set({ detail: d });
  void (async () => {
    for (let l = 0; l < d.layers.length; l++) {
      if (get().model !== mid) return;
      const r = await api<{ norms: number[] }>(`/api/arch/neurons?layer=${l}`).catch(() => null);
      if (!r || get().model !== mid) return;
      const norms = [...get().norms];
      norms[l] = r.norms;
      set({ norms });
      scene?.setNorms(l, r.norms);
    }
  })();
  return d;
}

export function go(f: Focus) {
  const same = JSON.stringify(f) === JSON.stringify(get().focus);
  const keep = f.level === "part" && f.part === "mlp" && get().neuron?.layer === f.layer;
  set({ focus: f, neuron: keep ? get().neuron : null, head: f.level === "part" && f.part === "attn" && get().head?.layer === f.layer ? get().head : null });
  if (!same) scene?.setFocus(f);
  marks();
}

/** Up a level; from the whole model, back Home. */
export function back() {
  const { focus, neuron, head } = get();
  if (neuron || head) return (set({ neuron: null, head: null }), marks());
  if (focus.level === "part") return go({ level: "layer", layer: focus.layer });
  if (focus.level === "layer" || focus.level === "end") return go({ level: "model" });
  setTab("home");
}

/** The next or previous layer, keeping the block that's open. */
export function step(d: number) {
  const { focus, detail } = get();
  if (!detail || (focus.level !== "layer" && focus.level !== "part")) return;
  const layer = Math.max(0, Math.min(detail.layers.length - 1, focus.layer + d));
  go(focus.level === "part" ? { ...focus, layer } : { level: "layer", layer });
}

export function clicked(k: Pick) {
  if (k.kind === "layer") return go({ level: "layer", layer: k.layer });
  if (k.kind === "part") return go({ level: "part", layer: k.layer, part: k.part });
  if (k.kind === "end") return go({ level: "end", end: k.end });
  if (k.kind === "neuron") return openNeuron(k.layer, k.neuron);
  pickHead(k.layer, k.head);
}

export function pickHead(layer: number, head: number) {
  const cur = get().head;
  const f = get().focus;
  if (!(f.level === "part" && f.part === "attn" && f.layer === layer)) go({ level: "part", layer, part: "attn" });
  set({ head: cur && cur.layer === layer && cur.head === head ? null : { layer, head } });
  marks();
}

export async function openNeuron(layer: number, neuron: number) {
  const f = get().focus;
  if (!(f.level === "part" && f.part === "mlp" && f.layer === layer)) go({ level: "part", layer, part: "mlp" });
  set({ neuron: { layer, neuron, norm: get().norms[layer]?.[neuron] ?? 0, lens: { up: [], down: [] }, acts: null } });
  marks();
  const n = await api<ArchNeuron>(`/api/arch/neuron?layer=${layer}&n=${neuron}`).catch(() => null);
  if (n && get().neuron?.layer === layer && get().neuron?.neuron === neuron) set({ neuron: n });
}

function marks() {
  const { neuron, head } = get();
  scene?.select([
    ...(neuron ? [{ kind: "neuron" as const, layer: neuron.layer, neuron: neuron.neuron }] : []),
    ...(head ? [{ kind: "head" as const, layer: head.layer, head: head.head }] : []),
  ]);
}

/** Sends the text as a chat message: the reply streams in, then the prompt and reply come back traced. The model
 *  writes faster than anyone can follow, so its tokens queue up and play through the view at a steady pace, the reply
 *  growing as each one comes out. Opens on the prompt's last token, where the model chose the reply's first word. */
let stopped = false;
const HOLD = 450; // ms each token rests at the unembedding before the next one sets off

export async function runText() {
  const text = get().text.trim();
  if (!text || get().running) return;
  set({ running: true, reply: "" });
  stopped = false;
  const queue: { text: string; frame?: ArchLive }[] = [];
  let ended = false, wake: (() => void) | null = null;
  const nudge = () => (wake?.(), (wake = null));
  const play = (async () => {
    for (;;) {
      if (stopped) return;
      const next = queue.shift();
      if (!next) {
        if (ended) return;
        await new Promise<void>((r) => (wake = r));
        continue;
      }
      if (next.frame && scene) {
        await scene.liveFrame(next.frame, next.text);
        if (stopped) return;
        set({ reply: (get().reply ?? "") + next.text });
        await new Promise((r) => setTimeout(r, HOLD));
      } else set({ reply: (get().reply ?? "") + next.text });
    }
  })();
  let run: ArchRun | null = null;
  try {
    await stream("/api/arch/run", { text }, (ev) => {
      if (ev.type === "delta") (queue.push({ text: String(ev.text), frame: ev.frame as ArchLive | undefined }), nudge());
      if (ev.type === "run") run = ev as unknown as ArchRun;
    });
  } finally {
    ended = true;
    nudge();
    await play;
    set({ running: false, reply: null });
  }
  if (!run) { // stopped while writing: back to the last full run, or to none
    const prev = get().run;
    scene?.setRun(prev);
    if (prev) void pickToken(get().token);
    return;
  }
  const r: ArchRun = run;
  set({ run: r, acts: null });
  scene?.setRun(r);
  await pickToken(Math.max(0, r.n_prompt - 1));
  const n = get().neuron;
  if (n) void openNeuron(n.layer, n.neuron); // its activations over the new text
}

/** Stops the model writing, and the view playing what it already wrote. */
export function stopRun() {
  stopped = true;
  return api("/api/cancel", {});
}

export function clearRun() {
  set({ run: null, acts: null });
  scene?.setRun(null);
}

export async function pickToken(t: number) {
  set({ token: t });
  const r = await api<ArchActs>(`/api/arch/acts?t=${t}`).catch(() => null);
  if (get().token !== t) return;
  const acts = r ? { q: new Int8Array(Uint8Array.from(atob(r.q), (c) => c.charCodeAt(0)).buffer), peak: r.peak, top: r.top } : null;
  set({ acts });
  scene?.setToken(t, acts?.q ?? null, acts?.peak ?? null);
}

/** The strongest neurons of one layer at the chosen token: [neuron, impact relative to the layer's peak]. */
export function layerTop(acts: Acts, layer: number, N: number, k = 8): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i < N; i++) out.push([i, acts.q[layer * N + i] / 127]);
  return out.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, k);
}

export const PART_NAMES: Record<Part, string> = { attn: "Attention", mlp: "MLP", norm1: "Norm before attention", norm2: "Norm before the MLP" };
