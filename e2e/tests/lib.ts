import { expect, type Locator, type Page } from "@playwright/test";
import { audit, AUDIT } from "./audit";

export const RECORD = process.env.MODE === "record";
// how long to wait for the engine: minutes against a real model; against the mock, the longest replay (describe's
// ~1,400 events, 12 ms apart) with room for a busy machine
export const T = RECORD ? 30 * 60_000 : 60_000;
export const MODEL = "Qwen/Qwen3-0.6B";

// What moves on its own and isn't the UI's layout: canvases (the scenes, spinners, art), the plan clock and load
// timer, and toasts (checked by text instead). Everything else must match pixel for pixel.
const STILL = `
  canvas { visibility: hidden !important; }
  .toast { display: none !important; }
  .plan-clock, #start-status, .msg-rate { visibility: hidden !important; }
`;

/** Start recording a scenario (record mode) or load its recording into the mock (replay). */
export async function scenario(page: Page, name: string) {
  const r = await page.request.post(RECORD ? `/__fixture/start?name=${name}` : `/__fixture/load?name=${name}`);
  expect(r.ok(), `fixture ${name}`).toBeTruthy();
}

/** Compare against a golden; skipped while recording, where output and timing come from a live model. */
export async function snap(target: Page | Locator, name: string) {
  if (RECORD) return;

  const page = "page" in target ? target.page() : target;
  if (AUDIT) return audit(page, name);
  const tag = await page.addStyleTag({ content: STILL });

  // WebKit paints only what's inside the window, so a panel taller than it would come out blank at the bottom:
  // grow the window to fit it for the shot, then put it back
  const size = page.viewportSize()!;
  const box = "page" in target ? await target.boundingBox() : null;
  const tall = box && box.height + 120 > size.height;
  if (tall) await page.setViewportSize({ width: size.width, height: Math.ceil(box.height + 240) });

  try {
    await expect(target).toHaveScreenshot(`${name}.png`);
  } finally {
    if (tall) await page.setViewportSize(size);
    await tag.evaluate((el) => el.remove());
  }
}

/** Nothing streaming any more: the UI drops body.busy when a stream ends. */
export async function idle(page: Page, timeout = T) {
  await expect(page.locator("body")).not.toHaveClass(/\bbusy\b/, { timeout });
}

/** A live status line ran and has finished. */
export async function ran(line: Locator, timeout = T) {
  await expect(line).toHaveClass(/\blive\b/, { timeout: 15_000 });
  await expect(line).not.toHaveClass(/\blive\b/, { timeout });
}

/** Every request the UI made was recorded, with the same body. */
export async function clean(page: Page) {
  if (RECORD) return;

  const log = await (await page.request.get("/__fixture/log")).json();
  expect(log.unrecorded, "requests with no recording").toEqual([]);
  expect(log.mismatches, "request bodies that differ from the recording").toEqual([]);
}

export async function toast(page: Page, text: string | RegExp) {
  await expect(page.locator("#toast")).toHaveText(text);
}
