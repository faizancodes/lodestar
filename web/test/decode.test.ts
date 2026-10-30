// @vitest-environment jsdom
// The streaming-text decoder leaves the same DOM as the original in ui/ did after every token and every animation
// frame, replaying recorded replies through the same markdown render the chat uses.
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanReply } from "../src/lib/format";
import { md } from "../src/lib/markdown";
import { exchanges, same, type Exchange } from "./recorded";

// each reply on its own: a steer run streams several columns, a test several probes
const streams = exchanges.filter((x) => x.type === "sse").flatMap((x) => {
  const by = new Map<string, NonNullable<typeof x.events>>();

  for (const e of x.events || []) {
    const d = e.data;
    if (d.type !== "delta" || typeof d.delta !== "string") continue;

    const k = `${d.col ?? ""}|${d.probe ?? ""}|${d.kind ?? ""}|${d.index ?? ""}`;
    by.set(k, [...(by.get(k) || []), e]);
  }

  return [...by.values()];
});

function replay(createDecoder: () => { update(root: HTMLElement | null): void }, events: NonNullable<Exchange["events"]>, still: boolean) {
  let clock = 0, queue: (() => void)[] = [];
  vi.stubGlobal("matchMedia", () => ({ matches: still }));
  vi.stubGlobal("requestAnimationFrame", (cb: () => void) => queue.push(cb));
  vi.spyOn(performance, "now").mockImplementation(() => clock);

  const root = document.createElement("div");
  document.body.append(root);

  const dec = createDecoder(), seen: string[] = [];
  let text = "", last = 0;
  const frame = (ms: number) => { clock += ms; const q = queue; queue = []; q.forEach((f) => f()); seen.push(root.innerHTML); };

  for (const e of events) {
    for (let t = last; t < e.dt * 1000 - 16; t += 16) frame(16); // frames between tokens, at their recorded pace

    clock = e.dt * 1000;
    last = clock;
    text += e.data.delta;

    root.innerHTML = md(cleanReply(text));
    dec.update(root);
    seen.push(root.innerHTML);
  }

  for (let i = 0; i < 40; i++) frame(16); // let the last letters settle
  root.remove();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();

  return seen;
}

afterEach(() => (vi.unstubAllGlobals(), vi.restoreAllMocks()));

describe.each([false, true])("decoder leaves the DOM ui/decode.js did (reduced motion: %s)", (still) => {
  it("on every recorded stream", async () => {
    const { createDecoder } = await import("../src/art/decode");
    expect(streams.length).toBeGreaterThan(10);

    const runs = streams.map((s) => replay(createDecoder, s, still));

    if (!still) expect(runs.some((r) => r.some((h) => h.includes('class="dc')))).toBe(true); // it really decoded
    same(runs);
  }, 60_000);
});
