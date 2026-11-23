import { afterEach, describe, expect, it, vi } from "vitest";
import { api, errorMessage } from "../src/lib/api";
import { searchDose } from "../src/lib/format";
import { md } from "../src/lib/markdown";

afterEach(() => vi.unstubAllGlobals());

describe("regression tests for web/src/lib helpers", () => {
  it("searchDose(0) returns 0 instead of NaN", () => {
    expect(searchDose(0)).toBe(0);
    expect(Number.isFinite(searchDose(0))).toBe(true);
    expect(searchDose(0.5)).toBe(0.3);
    expect(searchDose(-0.5)).toBe(-0.3);
  });

  it("errorMessage extracts strings from Error and non-Error values", () => {
    expect(errorMessage(new Error("boom"))).toBe("boom");
    expect(errorMessage("plain string")).toBe("plain string");
    expect(errorMessage(42)).toBe("42");
  });

  it("api() sends POST even for falsy JSON bodies and formats HTTP status fallback", async () => {
    const calls: { url: string; init?: RequestInit }[] = [];
    vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
      calls.push({ url, init });
      return new Response(JSON.stringify({ ok: true }), { status: 200 });
    });

    await api("/api/test", false);
    expect(calls[0].init?.method).toBe("POST");
    expect(calls[0].init?.body).toBe("false");

    await api("/api/get");
    expect(calls[1].init?.method).toBe("GET");
    expect(calls[1].init?.body).toBeUndefined();

    vi.stubGlobal("fetch", async () => new Response("not json", { status: 502 }));
    await expect(api("/api/fail")).rejects.toThrow("HTTP 502");
  });

  it("md() strips fenced code language and escapes code contents", () => {
    expect(md("```ts\nconst x = a < b;\n```")).toBe("<pre><code>const x = a &lt; b;</code></pre>");
  });
});
