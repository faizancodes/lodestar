import { describe, expect, it } from "vitest";
import * as now from "../src/lib/format";
import { md as nowMd, mdInline as nowMdInline } from "../src/lib/markdown";
import type { Direction, Item } from "../src/lib/types";
import { appSnippet, fromApp, replies, walk } from "./legacy";

// Inputs gathered from the recordings: every string, number, lens, layer band, recipe line and instructed pair
const strings = new Set<string>(), numbers = new Set<number>(), lenses: string[][] = [], bands: number[][] = [];
const items: Item[] = [], pairs: { baseline?: string; target?: string }[] = [], dirs: Direction[] = [];
walk((v, key) => {
  if (typeof v === "string") strings.add(v);
  else if (typeof v === "number") numbers.add(v);
  else if (Array.isArray(v) && v.length && v.every((x) => typeof x === "string") && (key === "lens" || key === "up" || key === "down")) lenses.push(v);
  else if (Array.isArray(v) && v.every((x) => Number.isInteger(x)) && (key === "band" || key === "layers")) bands.push(v);
  else if (v && typeof v === "object" && !Array.isArray(v)) {
    const o = v as Record<string, unknown>;
    if ((typeof o.layer === "number" && typeof o.neuron === "number") || (o.kind === "dir" && typeof o.dir === "string")) items.push(o as Item);
    if (typeof o.baseline === "string" || typeof o.target === "string") pairs.push(o as { baseline?: string; target?: string });
    if (typeof o.id === "string" && typeof o.ready === "boolean" && Array.isArray(o.band)) dirs.push(o as Direction);
  }
});
const texts = [...strings, ...replies(), "", " ", "a\n\nb", "**", "*x*", "```js\ncode", "1. a\n2) b\n- c\n# h\n#### h4"];
const models = [...strings].filter((s) => /^[\w.-]+\/[\w.-]+$/.test(s)).concat(["", "model", "Qwen3-0.6B"]);

const scope = { state: { recipe: { items, name: "" }, dirs, info: null }, ablCoef: () => 1 };
const old = fromApp<Record<string, (...a: never[]) => unknown>>(
  ["esc", "showTok", "nid", "fmt", "modelName", "gb", "isDir", "itemKey", "dirReady", "labelOf", "bandText", "lensLabel",
    "isWord", "instruction", "styleShift", "md", "mdInline", "mdBlocks", "cleanReply", "pushes"],
  scope,
);
const call = (f: unknown, ...a: unknown[]) => (f as (...a: unknown[]) => unknown)(...a);
const same = (name: string, a: unknown, b: unknown) => expect(b, name).toEqual(a);

describe("helpers give the same output as ui/app.js on recorded inputs", () => {
  it("has enough recorded data to mean something", () => {
    expect(texts.length).toBeGreaterThan(500);
    expect(lenses.length).toBeGreaterThan(20);
    expect(items.length).toBeGreaterThan(20);
    expect(bands.length).toBeGreaterThan(3);
    expect(pairs.length).toBeGreaterThan(5);
    expect(replies().length).toBeGreaterThan(10);
  });

  it("text: esc, showTok, isWord, instruction, cleanReply", () => {
    for (const t of texts) {
      same("esc", call(old.esc, t), now.esc(t));
      same("showTok", call(old.showTok, t), now.showTok(t));
      same("isWord", call(old.isWord, t), now.isWord(t));
      same("instruction", call(old.instruction, t), now.instruction(t));
      same("cleanReply", call(old.cleanReply, t), now.cleanReply(t));
    }
  });

  it("markdown: every recorded reply and string", () => {
    for (const t of texts) {
      same("md", call(old.md, t), nowMd(t));
      same("mdInline", call(old.mdInline, t), nowMdInline(t));
    }
  });

  it("numbers: fmt, gb, nid", () => {
    for (const x of [...numbers, 0, -0, 1e10, 9.99e9, -1.005]) {
      for (const d of [undefined, 0, 1, 2, 3]) same("fmt", call(old.fmt, x, d), now.fmt(x, d));
      same("gb", call(old.gb, Math.abs(x)), now.gb(Math.abs(x)));
      same("nid", call(old.nid, x, x), now.nid(x, x));
    }
  });

  it("models: modelName, steerDefaultName, ablDefaultName", () => {
    const steerName = (model: string | undefined, name: string) => fromApp<{ steerDefaultName: () => string }>(
      ["modelName", "steerDefaultName"], { state: { info: model ? { model } : null, recipe: { name } } }).steerDefaultName();
    const ablName = (model: string | undefined, coef: number) => fromApp<{ ablDefaultName: () => string }>(
      ["modelName", "ablDefaultName"], { state: { info: model ? { model } : null }, ablCoef: () => coef }).ablDefaultName();
    const names = [...strings].slice(0, 400).concat(["", "My Recipe!!", "--x--", "a".repeat(60)]);
    for (const m of [...models, undefined]) {
      same("modelName", call(old.modelName, m), now.modelName(m));
      for (const c of [1, 0.5, 1.5, 2]) same("ablDefaultName", ablName(m, c), now.ablDefaultName(m, c));
    }
    for (const n of names) same("steerDefaultName", steerName("Qwen/Qwen3-0.6B", n), now.steerDefaultName("Qwen/Qwen3-0.6B", n));
  });

  it("neurons: lensLabel, searchDose", () => {
    for (const l of [...lenses, [], ["cookie", " Cookie", "cookies", "", "x", "y", "z"]]) same("lensLabel", call(old.lensLabel, l), now.lensLabel(l));
    const oldDose = new Function("x", `return ${appSnippet("Math.sign(x.score) * Math.min(1, +(0.15 / Math.abs(x.score)).toFixed(2))")};`);
    for (const x of numbers) if (x) same("searchDose", oldDose({ score: x }), now.searchDose(x));
  });

  it("recipes: itemKey, labelOf, isDir, dirReady, bandText, pushes", () => {
    for (const it of items) {
      same("itemKey", call(old.itemKey, it), now.itemKey(it));
      same("labelOf", call(old.labelOf, it), now.labelOf(it));
      same("isDir", call(old.isDir, it), now.isDir(it));
      if (now.isDir(it)) same("dirReady", call(old.dirReady, it), now.dirReady(it, dirs));
    }
    for (const b of [...bands, [], [3], [2, 4], [1, 2, 3], [2, 4, 6, 8], [1, 2, 4]]) same("bandText", call(old.bandText, b), now.bandText(b));
    for (const sign of [1, -1]) same("pushes", call(old.pushes, sign), now.pushes(items, dirs, sign));
  });

  it("styleShift", () => {
    same("styleShift", call(old.styleShift, pairs), now.styleShift(pairs));
    for (const p of pairs) same("styleShift", call(old.styleShift, [p]), now.styleShift([p]));
    same("styleShift", call(old.styleShift, []), now.styleShift([]));
  });
});
