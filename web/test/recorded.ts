// The recorded /api sessions in ../e2e/fixtures, as inputs for the tests, and a digest for checking an output against
// the one stored when the port was proven equal to the original UI.
import { createHash } from "node:crypto";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { expect } from "vitest";

const FIXTURES = join(__dirname, "../../e2e/fixtures");

/** Checks a (large) output against its stored snapshot, by digest. `pnpm test -u` stores new ones after a deliberate
 * change. The snapshots were taken while ui/ still existed and these tests ran it beside the port on the same inputs,
 * with identical results. */
export function same(value: unknown) {
  const json = JSON.stringify(value);
  expect({ length: json.length, sha256: createHash("sha256").update(json).digest("hex") }).toMatchSnapshot();
}

export type Exchange = {
  method: string; path: string; query: string; body: unknown; status: number;
  type: "json" | "sse"; response?: unknown; events?: { dt: number; data: { type: string; [k: string]: unknown } }[];
};
export const fixtures: Record<string, Exchange[]> = Object.fromEntries(
  readdirSync(FIXTURES).filter((f) => f.endsWith(".json")).map((f) => [f.replace(/\.json$/, ""), JSON.parse(readFileSync(join(FIXTURES, f), "utf8"))]),
);
export const exchanges = Object.values(fixtures).flat();

/** Every value anywhere in the recordings (requests, responses, stream events), visited depth first. */
export function walk(visit: (v: unknown, key: string | null) => void) {
  const go = (v: unknown, key: string | null) => {
    visit(v, key);
    if (Array.isArray(v)) v.forEach((x) => go(x, key));
    else if (v && typeof v === "object") for (const [k, x] of Object.entries(v)) go(x, k);
  };
  for (const x of exchanges) go({ body: x.body, response: x.response, events: x.events?.map((e) => e.data) }, null);
}

/** Every reply a stream wrote, joined from its deltas, one per column / probe / dose. */
export function replies() {
  const out: string[] = [];
  for (const x of exchanges) {
    if (x.type !== "sse") continue;
    const by = new Map<string, string>();
    for (const { data: e } of x.events || [])
      if (e.type === "delta" && typeof e.delta === "string") {
        const k = `${e.col ?? ""}|${e.probe ?? ""}|${e.kind ?? ""}|${e.index ?? ""}`;
        by.set(k, (by.get(k) || "") + e.delta);
      }
    out.push(...by.values());
  }
  return out;
}
