// Pure helpers carried over from ui/app.js unchanged; the parity tests run both versions on recorded data.
import type { DirItem, Direction, Item, Push } from "./types";

export const esc = (s: unknown) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
export const showTok = (s: string) => (s === "" ? "∅" : s.replace(/\n/g, "↵").replace(/\t/g, "⇥"));
export const nid = (l: number, n: number) => `L${l} · N${n}`;
export const fmt = (x: number, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(Number(x)).toFixed(d);
export const modelName = <T extends string | null | undefined>(id: T) => (id ? id.split("/").pop()! : id);
export const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
export const gb = (bytes: number) => `${(bytes / 1e9).toFixed(bytes < 1e10 ? 1 : 0)} GB`;

// a recipe holds neurons ({layer, neuron, dose}) and directions ({kind: "dir", dir, name, layers, dose, op}), where
// a direction is pushed along (dose in units of its own size) or erased back to ordinary text (dose 1 = fully)
export const isDir = (it: Item): it is DirItem => it.kind === "dir";
export const itemKey = (it: Item) => (isDir(it) ? `d:${it.dir}` : `${it.layer}:${it.neuron}`);
export const dirReady = (it: DirItem, dirs: Direction[] | null | undefined) => !!dirs?.find((d) => d.id === it.dir && d.ready);
export const labelOf = (it: Item) => isDir(it) ? it.name || it.dir : it.label || (it.lens?.[0] || "").trim() || nid(it.layer, it.neuron);
export const bandText = (ls: number[] = []) => {
  const st = ls.length > 1 ? ls[1] - ls[0] : 1;
  return ls.length > 2 && ls.every((l, i) => !i || l - ls[i - 1] === st) ? `L${ls[0]}–${ls.at(-1)}${st > 1 ? ` /${st}` : ""}` : ls.map((l) => `L${l}`).join(" ");
};

/** The recipe as the engine applies it; sign -1 is the Opposite column, which can't erase. */
export function pushes(items: Item[], dirs: Direction[] | null | undefined, sign = 1): Push[] {
  return items.map((i): Push => isDir(i)
    ? { kind: "dir", dir: i.dir, layers: i.layers, op: i.op, dose: sign * i.dose, enabled: dirReady(i, dirs) && (sign > 0 || i.op !== "erase") }
    : { layer: i.layer, neuron: i.neuron, dose: sign * i.dose }).filter((p) => p.enabled !== false);
}

/** Where a word search starts a neuron: its direct push noticeable, not yet a loop. */
export const searchDose = (score: number) => Math.sign(score) * Math.min(1, +(0.15 / Math.abs(score)).toFixed(2));

// the words a neuron pushes toward, without repeats like "cookie · Cookie · cookies"
export function lensLabel(lens: string[] = []) {
  const seen = new Set<string>(), out: string[] = [];
  for (const t of lens) {
    const w = t.trim(), k = w.toLowerCase().replace(/s$/, "");
    if (w && !seen.has(k)) seen.add(k), out.push(w);
  }
  return out.slice(0, 4).join(" · ");
}

// one box: a word or two searches by word, a sentence searches by behaviour
export const isWord = (s: string) => s.split(/\s+/).length <= 2 && !/[.!?,:]/.test(s);
export const instruction = (t: string) => `Follow this in every reply, without mentioning it: ${t}`;

// how much the instruction actually changed the answers: mean fraction of words in the instructed answer that
// aren't in the plain one (0 = identical, so the instruction did nothing; ~1 = a completely different reply)
export function styleShift(pairs: { baseline?: string; target?: string }[]) {
  const words = (t?: string) => new Set((t || "").toLowerCase().match(/[a-z0-9']+/g) || []);
  const scores = pairs.map((p) => {
    const base = words(p.baseline), tgt = [...words(p.target)];
    if (!tgt.length) return 0;
    return tgt.filter((w) => !base.has(w)).length / tgt.length;
  });
  return scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
}

export const cleanReply = (t: string) => t.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").replace(/^\s+/, "");

export const ablDefaultName = (model: string | undefined, coef: number) => `${modelName(model || "model")}-abliterated` + (coef === 1 ? "" : `-${coef}`);
export const steerDefaultName = (model: string | undefined, recipeName: string) => {
  const base = modelName(model || "model").replace(/[^A-Za-z0-9._ -]+/g, "");
  const what = (recipeName || "steered").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40);
  return `${base}-${what || "steered"}`;
};

