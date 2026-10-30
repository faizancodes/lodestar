// Behaviour the parity walk-through doesn't reach: one stream at a time, the Stop buttons, and the narrow window.
// Replay only: these hold a stream open or resize the window, which a live recording has no need to.
import { expect, test, type Route } from "@playwright/test";
import { clean, idle, RECORD, scenario, snap, toast } from "./lib";

test.skip(RECORD);
test.afterEach(async ({ page }) => clean(page));

test("rules: one stream at a time, and Stop cancels it", async ({ page }) => {
  await scenario(page, "chat");
  // hold the first reply open until the checks are done; cancelling isn't recorded, so answer it here
  let release!: () => void;
  const held = new Promise<void>((r) => (release = r));
  await page.route("**/api/chat", async (route: Route) => (await held, route.continue()), { times: 1 });
  const cancels: unknown[] = [];
  await page.route("**/api/cancel", (route) => (cancels.push(route.request().postDataJSON()), route.fulfill({ json: { ok: true } })));

  await page.goto("/");
  await page.keyboard.press("Meta+3");
  await expect(page.locator("#chat-input")).toBeFocused();
  await page.locator('#chat-mods .mod[data-mod="abl"]').click();
  await expect(page.locator('#chat-mods .mod[data-mod="abl"]')).not.toHaveClass(/\bon\b/);
  await page.locator("#chat-opts summary").click(); // the recorded reply's settings
  await page.locator("#chat-temp").fill("0");
  await page.locator("#chat-max").fill("192");
  await page.locator("#chat-model").click();
  await page.locator("#chat-starters .starter", { hasText: "Write a haiku about Mondays" }).click();
  await expect(page.locator("body")).toHaveClass(/\bbusy\b/);
  await expect(page.locator("#chat-stop")).toBeVisible();
  await expect(page.locator("#chat-send")).toBeHidden();
  await expect(page.locator("#chat-log .msg.bot.typing .dots")).toBeVisible();

  // every other page refuses to start a second stream, and says why
  await page.keyboard.press("Meta+1");
  await page.locator("#abl-test-go").click();
  await toast(page, "Something is already running. Stop it first.");
  await page.keyboard.press("Meta+2");
  await page.locator("#t-go").click();
  await toast(page, "Something is already running. Stop it first.");

  // each page's Stop asks the server to cancel
  await page.locator("#view-steer [data-cancel]").first().click();
  await page.keyboard.press("Meta+3");
  await page.locator("#chat-stop").click();
  await expect.poll(() => cancels.length).toBe(2);

  release();
  await idle(page);
  await expect(page.locator("#chat-send")).toBeVisible();
  await expect(page.locator("#chat-log .msg.bot .msg-acts")).toBeVisible();
});

test("rules: a narrow window", async ({ page }) => {
  await scenario(page, "shell");
  await page.setViewportSize({ width: 820, height: 760 });
  await page.goto("/");
  await expect(page.locator("#view-steer")).toBeVisible();
  await snap(page, "narrow-steer");
  await page.keyboard.press("Meta+0");
  await expect(page.locator("#view-home")).toBeVisible();
  await snap(page, "narrow-home");
  await page.keyboard.press("Meta+3");
  await expect(page.locator("#chat-input")).toBeFocused();
  await snap(page, "narrow-chat");
});
