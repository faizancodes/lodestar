import { afterEach, describe, expect, it, vi } from "vitest";
import { stream as nowStream } from "../src/lib/stream";
import { useApp } from "../src/store/app";
import { exchanges, fromApp, type Exchange } from "./legacy";

// A recorded stream as a fetch Response, its bytes cut at odd places (mid-event and mid-character) the way the
// network delivers them
function respond(x: Pick<Exchange, "status" | "events" | "response">, cut: number) {
  const ok = x.status < 400;
  if (!ok) return new Response(JSON.stringify(x.response ?? {}), { status: x.status });
  const bytes = new TextEncoder().encode((x.events || []).map((e) => `data: ${JSON.stringify(e.data)}\n\n`).join(""));
  return new Response(new ReadableStream({
    start(c) {
      for (let i = 0; i < bytes.length; i += cut) c.enqueue(bytes.slice(i, i + cut));
      c.close();
    },
  }), { status: x.status });
}

type Run = { events: unknown[]; toasts: [string, boolean][]; busy: boolean[]; ok: unknown };

async function runOld(x: Exchange, cut: number): Promise<Run> {
  const run: Run = { events: [], toasts: [], busy: [], ok: null };
  const classList = { add: () => run.busy.push(true), remove: () => run.busy.push(false) };
  const { stream } = fromApp<{ stream: (p: string, b: unknown, on: (e: unknown) => void) => Promise<boolean> }>(["busy", "stream"], {
    fetch: async () => respond(x, cut),
    toast: (m: string, err?: boolean) => run.toasts.push([m, !!err]),
    document: { body: { classList } },
  });
  run.ok = await stream(x.path, x.body, (e) => run.events.push(e));
  return run;
}

async function runNow(x: Exchange, cut: number): Promise<Run> {
  const run: Run = { events: [], toasts: [], busy: [], ok: null };
  vi.stubGlobal("fetch", async () => respond(x, cut));
  const off = useApp.subscribe((s, prev) => {
    if (s.busy !== prev.busy) run.busy.push(s.busy);
    if (s.toast !== prev.toast && s.toast) run.toasts.push([s.toast.msg, s.toast.err]);
  });
  run.ok = await nowStream(x.path, x.body, (e) => run.events.push(e));
  off();
  return run;
}

afterEach(() => vi.unstubAllGlobals());

const synthetic: Exchange[] = [
  { method: "POST", path: "/api/x", query: "", body: {}, status: 200, type: "sse", events: [{ dt: 0, data: { type: "delta", delta: "é ✓" } }, { dt: 0, data: { type: "error", message: "Out of memory" } }, { dt: 0, data: { type: "end" } }] },
  { method: "POST", path: "/api/x", query: "", body: {}, status: 200, type: "sse", events: [{ dt: 0, data: { type: "delta", delta: "a" } }, { dt: 0, data: { type: "cancelled" } }, { dt: 0, data: { type: "end" } }] },
  { method: "POST", path: "/api/x", query: "", body: {}, status: 409, type: "json", response: { error: "Load a model first (bottom left)." } },
  { method: "POST", path: "/api/x", query: "", body: {}, status: 500, type: "json", response: "not json" },
];

describe("stream() behaves like ui/app.js's", () => {
  const recorded = exchanges.filter((x) => x.type === "sse");
  it("has recorded streams to replay", () => expect(recorded.length).toBeGreaterThan(10));

  it.each([3, 7, 4096])("every recorded stream, delivered %i bytes at a time", async (cut) => {
    for (const x of recorded) expect(await runNow(x, cut), `${x.path}`).toEqual(await runOld(x, cut));
  }, 60_000);

  it("errors, cancellations and failed requests", async () => {
    for (const x of synthetic) expect(await runNow(x, 5)).toEqual(await runOld(x, 5));
  });

  it("refuses a second stream while one runs", async () => {
    const x = recorded[0];
    vi.stubGlobal("fetch", async () => respond(x, 64));
    const toasts: string[] = [];
    const off = useApp.subscribe((s, prev) => s.toast !== prev.toast && s.toast && toasts.push(s.toast.msg));
    const [a, b] = await Promise.all([nowStream(x.path, x.body, () => {}), nowStream(x.path, x.body, () => {})]);
    off();
    expect([a, b]).toEqual([true, false]);
    expect(toasts).toEqual(["Something is already running. Stop it first."]);
  });
});
