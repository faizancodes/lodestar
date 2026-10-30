import { describe, expect, it } from "vitest";
import * as now from "../src/lib/format";
import { md as nowMd, mdInline as nowMdInline } from "../src/lib/markdown";
import type { Direction, Item } from "../src/lib/types";
import { replies, same, walk } from "./recorded";

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

describe("helpers give the outputs ui/app.js gave on recorded inputs", () => {
  it("has enough recorded data to mean something", () => {
    expect(texts.length).toBeGreaterThan(500);
    expect(lenses.length).toBeGreaterThan(20);
    expect(items.length).toBeGreaterThan(20);
    expect(bands.length).toBeGreaterThan(3);
    expect(pairs.length).toBeGreaterThan(5);
    expect(replies().length).toBeGreaterThan(10);
  });

  it("text: esc, showTok, isWord, instruction, cleanReply", () => {
    same(texts.map((t) => [now.esc(t), now.showTok(t), now.isWord(t), now.instruction(t), now.cleanReply(t)]));
  });

  it("markdown: every recorded reply and string", () => {
    same(texts.map((t) => [nowMd(t), nowMdInline(t)]));
  });

  it("numbers: fmt, gb, nid", () => {
    same([...numbers, 0, -0, 1e10, 9.99e9, -1.005].map((x) => [
      [undefined, 0, 1, 2, 3].map((d) => now.fmt(x, d)), now.gb(Math.abs(x)), now.nid(x, x)]));
  });

  it("models: modelName, steerDefaultName, ablDefaultName", () => {
    const names = [...strings].slice(0, 400).concat(["", "My Recipe!!", "--x--", "a".repeat(60)]);
    same([...models, undefined].map((m) => [now.modelName(m), [1, 0.5, 1.5, 2].map((c) => now.ablDefaultName(m, c))]));
    same(names.map((n) => now.steerDefaultName("Qwen/Qwen3-0.6B", n)));
  });

  it("neurons: lensLabel, searchDose", () => {
    same([...lenses, [], ["cookie", " Cookie", "cookies", "", "x", "y", "z"]].map((l) => now.lensLabel(l)));
    same([...numbers].filter(Boolean).map((x) => now.searchDose(x)));
  });

  it("recipes: itemKey, labelOf, isDir, dirReady, bandText, pushes", () => {
    same(items.map((it) => [now.itemKey(it), now.labelOf(it), now.isDir(it), now.isDir(it) ? now.dirReady(it, dirs) : null]));
    same([...bands, [], [3], [2, 4], [1, 2, 3], [2, 4, 6, 8], [1, 2, 4]].map((b) => now.bandText(b)));
    same([1, -1].map((sign) => now.pushes(items, dirs, sign)));
  });

  it("styleShift", () => {
    same([now.styleShift(pairs), ...pairs.map((p) => now.styleShift([p])), now.styleShift([])]);
  });
});
