// Loads pieces of the current UI (ui/) so the tests can run the old code beside the port on the same inputs.
import { parse } from "acorn";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

export const UI = join(__dirname, "../../ui");
const FIXTURES = join(__dirname, "../../e2e/fixtures");
const APP = readFileSync(join(UI, "app.js"), "utf8");

const AST = parse(APP, { ecmaVersion: "latest", sourceType: "module" }) as unknown as {
  body: { type: string; start: number; end: number; id?: { name: string }; declarations?: { id: { name: string } }[] }[];
};

/** The top-level declarations of ui/app.js with these names, evaluated together with `scope` in reach. */
export function fromApp<T>(names: string[], scope: Record<string, unknown> = {}): T {
  const ast = AST;
  const src = names.map((name) => {
    const node = ast.body.find((n) => n.id?.name === name || n.declarations?.some((d) => d.id.name === name));
    if (!node) throw new Error(`ui/app.js has no top-level ${name}`);
    return APP.slice(node.start, node.end);
  });
  return new Function(...Object.keys(scope), `${src.join("\n")}\nreturn { ${names.join(", ")} };`)(...Object.values(scope));
}

/** A snippet of ui/app.js, found by its text, so a test fails if the original changes. */
export function appSnippet(text: string) {
  if (!APP.includes(text)) throw new Error(`ui/app.js no longer contains: ${text}`);
  return text;
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
