// The canvas modules draw exactly what the originals in ui/ drew: every call on the 2D context and every path,
// frame by frame, with the clock and animation frames driven by hand. Both with motion and with reduced motion.
import { afterEach, describe, expect, it, vi } from "vitest";
import { same } from "./recorded";

type Log = unknown[];

function setup(still: boolean) {
  const log: Log = [];

  class Path2D {
    ops: unknown[] = [];
    moveTo(...a: number[]) { this.ops.push(["m", ...a]); }
    lineTo(...a: number[]) { this.ops.push(["l", ...a]); }
  }

  const ctx = new Proxy({ fillStyle: "#000" } as Record<string, unknown>, {
    get: (t, k) => (k in t ? t[k as string] : (...a: unknown[]) => log.push([k, ...a.map((x) => (x instanceof Path2D ? x.ops : x))])),
    set: (t, k, v) => (log.push([`${String(k)}=`, v]), (t[k as string] = v), true),
  });

  const canvas = (w = 220, h = 140) => ({
    width: 0, height: 0, style: {}, clientWidth: w, clientHeight: h, isConnected: true, offsetParent: {}, offsetWidth: 0,
    dataset: {}, getContext: () => ctx, closest: () => null, getBoundingClientRect: () => ({ top: 10 }),
    classList: { add: (c: string) => log.push(["+class", c]), remove: (c: string) => log.push(["-class", c]) },
  });

  let queue: ((t: number) => void)[] = [], clock = 1000;
  const listeners: Record<string, (() => void)[]> = {};

  vi.stubGlobal("Path2D", Path2D);
  vi.stubGlobal("devicePixelRatio", 2);
  vi.stubGlobal("matchMedia", () => ({ matches: still }));
  vi.stubGlobal("getComputedStyle", () => ({ color: "rgb(195, 207, 245)" }));
  vi.stubGlobal("requestAnimationFrame", (cb: (t: number) => void) => queue.push(cb));
  vi.stubGlobal("cancelAnimationFrame", () => (queue = []));
  vi.stubGlobal("addEventListener", () => {});
  vi.stubGlobal("removeEventListener", () => {});
  vi.stubGlobal("document", { hidden: false, addEventListener: () => {}, removeEventListener: () => {}, querySelector: () => ({ getBoundingClientRect: () => ({ bottom: 190 }) }) });
  vi.spyOn(performance, "now").mockImplementation(() => clock);

  const frames = (n: number, ms = 70) => {
    for (let i = 0; i < n; i++) {
      clock += ms;
      const q = queue;
      queue = [];
      q.forEach((cb) => cb(clock));
    }
  };

  const card = () => {
    const el = { dataset: {}, addEventListener: (e: string, f: () => void) => (listeners[e] ||= []).push(f) };
    return el;
  };
  const fire = (e: string) => listeners[e]?.forEach((f) => f());

  return { log, canvas, frames, card, fire };
}

// Each module is loaded fresh per run (loader.ts keeps state at module level)
async function load<T>(name: string): Promise<T> {
  vi.resetModules();
  return import(/* @vite-ignore */ `../src/art/${name}.ts`);
}

type Scenario = (env: ReturnType<typeof setup>) => Promise<void>;

const scenarios: Record<string, Scenario> = {
  async drawMarks({ canvas }) {
    const { drawMarks } = await load<typeof import("../src/art/glyphs")>("glyphs");
    const grid = new Float32Array(40 * 30).map((_, i) => Math.abs(Math.sin(i * 12.9898) * 1.3));
    drawMarks(canvas().getContext() as unknown as CanvasRenderingContext2D, grid, 40, 30, 3.5);
  },
  async sideArt({ canvas, frames }) {
    const { startSideArt } = await load<typeof import("../src/art/glyphs")>("glyphs");
    startSideArt(canvas(248, 900) as unknown as HTMLCanvasElement);
    frames(6, 130);
  },
  async spinner({ canvas, frames }) {
    const { startSpinner } = await load<typeof import("../src/art/glyphs")>("glyphs");
    const again = startSpinner(canvas() as unknown as HTMLCanvasElement, () => true);
    frames(5, 60);
    again();
    frames(2, 60);
  },
  async loader({ canvas, frames }) {
    const m = await load<typeof import("../src/art/loader")>("loader");
    m.startScene(canvas() as unknown as HTMLCanvasElement);
    frames(3);

    m.wake();
    m.setProgress(0.4);
    frames(5);

    m.setLayers(32, Array.from({ length: 32 }, (_, i) => Math.max(0, Math.min(1, 14.5 - i))));
    frames(5);

    const done = m.finishScene();
    frames(45);
    await done;

    m.stopScene();
  },
  async scan({ canvas, frames }) {
    const { startScan } = await load<typeof import("../src/art/scan")>("scan");
    startScan(canvas() as unknown as HTMLCanvasElement);
    frames(60, 70);
  },
  async ablScene({ canvas, frames }) {
    const { startAblScene } = await load<typeof import("../src/art/ablscene")>("ablscene");
    const s = startAblScene(canvas() as unknown as HTMLCanvasElement);
    for (const i of [0, 1, 2, 3, 3, 9]) (s.set(i), frames(25, 70));
  },
  async home({ canvas, frames, card, fire }) {
    const { drawHomeArt } = await load<typeof import("../src/art/home")>("home");
    for (const kind of ["abliterate", "steer", "chat"] as const) {
      drawHomeArt(canvas(300, 120) as unknown as HTMLCanvasElement, kind, card() as unknown as HTMLElement);
      fire("mouseenter");
      frames(8, 60);
      fire("mouseleave");
    }
  },
};

afterEach(() => (vi.unstubAllGlobals(), vi.restoreAllMocks()));

describe.each([false, true])("canvas art draws what ui/ drew (reduced motion: %s)", (still) => {
  it.each(Object.keys(scenarios))("%s", async (name) => {
    const env = setup(still);
    await scenarios[name](env);

    expect(env.log.length).toBeGreaterThan(still ? 3 : 10);
    same(env.log);
  });
});
