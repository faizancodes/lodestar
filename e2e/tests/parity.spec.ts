// The parity baseline: every view and state of the current UI, driven the way a person would, with each request's
// body checked against the recording and each state checked against a golden screenshot. Scenarios run in order and
// build on each other on the server (the model loaded at the start, the recipe saved in Steer, the refusal direction
// found in Abliterate), so recording runs serially. Each replay loads its own recording, so one failure there doesn't
// stop the rest.
import { expect, test } from "@playwright/test";
import { clean, idle, MODEL, ran, RECORD, scenario, snap, T, toast } from "./lib";

test.describe.configure({ mode: RECORD ? "serial" : "default" });

test.afterEach(async ({ page }) => clean(page));

// Replays the start recording only (recording it again here would overwrite it)
test("sidebar: before a model is loaded", async ({ page }) => {
  test.skip(RECORD);
  await scenario(page, "start");
  await page.goto("/");
  await expect(page.locator("#brand-mark svg")).toBeVisible();
  await expect(page.locator('.tab[data-tab="steer"]')).toHaveAttribute("aria-selected", "true");
  await expect(page.locator("#model-status .txt")).toHaveText("No model loaded");
  await snap(page.locator("aside.side"), "sidebar-empty");
});

test("start: pick a model, load it, land on Home", async ({ page }) => {
  await scenario(page, "start");
  await page.goto("/");
  const card = page.locator(`#start-models .start-model[data-id="${MODEL}"]`);
  await expect(card).toBeVisible();
  await expect(page.locator("#model-status .txt")).toHaveText("No model loaded");
  await snap(page, "start");

  await page.locator("#model-pick").click();
  await expect(page.locator("#model-menu")).toBeVisible();
  await snap(page, "start-model-menu");
  await page.keyboard.press("Escape");
  await expect(page.locator("#model-menu")).toBeHidden();

  await card.click();
  await expect(page.locator("#view-home")).toBeVisible({ timeout: T });
  await expect(page.locator("#home-model")).toHaveText("Qwen3-0.6B");
  await expect(page.locator("#model-status .txt")).toHaveText("Ready");
  await expect(page.locator("#start")).toBeHidden();
  await snap(page, "home");
});

test("shell: tabs, shortcuts, brand, model menu", async ({ page }) => {
  await scenario(page, "shell");
  await page.goto("/");
  await expect(page.locator("#view-steer")).toBeVisible(); // a fresh window opens on Steer
  for (const [key, view] of [["Meta+1", "abliterate"], ["Meta+3", "chat"], ["Meta+2", "steer"], ["Meta+0", "home"]]) {
    await page.keyboard.press(key);
    await expect(page.locator(`#view-${view}`)).toBeVisible();
    await expect(page.locator("body")).toHaveAttribute("data-tab", view);
  }
  await page.locator('.home-card[data-go="chat"]').click();
  await expect(page.locator("#view-chat")).toBeVisible();
  await page.locator("#brand").click();
  await expect(page.locator("#view-home")).toBeVisible();
  await page.locator('.tab[data-tab="abliterate"]').click();
  await expect(page.locator('.tab[data-tab="abliterate"]')).toHaveAttribute("aria-selected", "true");
  await snap(page, "abliterate-empty");

  // the tab survives a reload
  await page.reload();
  await expect(page.locator("#view-abliterate")).toBeVisible();

  await page.locator("#model-pick").click();
  await expect(page.locator("#model-menu .mm-item.on")).toContainText("Qwen3-0.6B");
  await snap(page, "model-menu-loaded");
  await snap(page.locator("#model-menu"), "model-menu-loaded-menu");
  await page.mouse.click(900, 450); // a click outside closes it
  await expect(page.locator("#model-menu")).toBeHidden();
});

test("steer: search by word, turn neurons on, try it, inspect, fix, save", async ({ page }) => {
  await scenario(page, "steer-word");
  await page.goto("/");
  await expect(page.locator("#view-steer")).toBeVisible();
  await snap(page, "steer-empty");

  await page.locator("#c-input").fill("cookie");
  await expect(page.locator("#c-status")).toHaveText("");
  await page.locator("#c-input").press("Enter");
  const found = page.locator("#c-results .found-row");
  await expect(found.first()).toBeVisible({ timeout: T });
  await snap(page.locator("#st-s1"), "steer-word-results");

  await found.nth(0).locator(".btn").click();
  await found.nth(1).locator(".btn").click();
  const rows = page.locator("#rc-list .rc-row");
  await expect(rows).toHaveCount(2);
  await expect(found.nth(0).locator(".btn")).toHaveText("On");
  await expect(page.locator("#tab-steer-count")).toHaveText("2");
  await rows.first().locator("input[type=range]").fill("0.5");
  await expect(rows.first().locator(".dose")).toHaveText("+0.50");
  await snap(page.locator("#st-s1"), "steer-recipe");
  await snap(page.locator("#steer-status"), "steer-status-on");

  // ask it something: the answer with the neurons on is v1, and the box now takes a fix
  const answers = page.locator("#tl .tl-a"), cur = page.locator("#tl .tl-a.cur");
  await page.locator("#t-chips .chip", { hasText: "Describe the way you look." }).click();
  await expect(answers).toHaveCount(1);
  await idle(page);
  await expect(cur.locator(".tl-a-b").first()).not.toBeEmpty();
  await expect(page.locator("#t-prompt")).toHaveAttribute("placeholder", "What should change?");
  await snap(page.locator("#st-s2"), "steer-answer");
  await cur.locator(".tl-act", { hasText: "Compare with normal" }).click();
  await expect(cur.locator(".tl-normal")).toBeVisible();
  await snap(page.locator("#st-s2"), "steer-compare");
  await cur.locator(".tl-act", { hasText: "Compare with normal" }).click();
  await expect(cur.locator(".tl-normal")).toHaveCount(0);

  // inspect the answer, a word in it, and a neuron from that word
  await cur.locator(".tl-act", { hasText: "Inspect" }).click();
  const words = page.locator('.panel[data-kind="inspect"] .in-words .tok');
  await expect(words.first()).toBeVisible({ timeout: T });
  await snap(page.locator('.panel[data-kind="inspect"]'), "inspect");
  // the word's details replace the first word's: wait until the old rows are gone before clicking one
  const old = await page.locator('.panel[data-kind="inspect"] .in-detail .found-row').first().elementHandle();
  await words.nth(2).click();
  await expect(words.nth(2)).toHaveClass(/\bsel\b/);
  await page.waitForFunction((el) => !el?.isConnected, old, { timeout: T });
  await page.locator('.panel[data-kind="inspect"] .in-detail .found-row .who').first().click();
  const neuron = page.locator('.panel[data-kind="neuron"]');
  await expect(neuron.locator(".nd-add")).toBeVisible({ timeout: T });
  await snap(neuron, "neuron-panel");
  await page.keyboard.press("Escape");
  await expect(neuron).toHaveCount(0);
  await page.keyboard.press("Escape");
  await expect(page.locator(".panel")).toHaveCount(0);

  // say what should change: a fix retunes the neurons and answers again as v2, which Undo takes back
  await page.locator("#t-chips .chip", { hasText: "It's too strong" }).click();
  await expect(page.locator("#t-prompt")).toHaveValue("It's too strong.");
  await page.locator("#t-go").click();
  await expect(page.locator("#tl .tl-now")).toBeVisible();
  await expect(page.locator("#tl .tl-now")).toHaveCount(0, { timeout: T });
  await expect(page.locator("#tl .typing")).toHaveCount(0, { timeout: T });
  await idle(page);
  await snap(page.locator("#st-s2"), "fix-result");
  if ((await answers.count()) === 2) {
    await cur.locator(".tl-act", { hasText: "Undo" }).click();
    await expect(answers).toHaveCount(1);
    await expect(rows).toHaveCount(2);
  }
  await page.locator("#tl-newq").click();
  await expect(page.locator("#t-prompt")).toHaveAttribute("placeholder", "Ask it something");
  await expect(page.locator("#tl-newq")).toHaveText("Cancel");

  // save, clear, and bring it back from the sidebar
  await page.locator("#rc-save").click();
  await toast(page, /^Saved “.+”\.$/);
  const saved = page.locator("#saved-list .saved-item");
  await expect(saved).toHaveCount(1);
  await expect(saved.first()).toHaveClass(/\bon\b/);
  await snap(page.locator(".side"), "sidebar-saved");
  await page.locator("#rc-clear").click();
  await expect(page.locator("#rc")).toBeHidden();
  await saved.first().click();
  await expect(rows).toHaveCount(2);
  await snap(page.locator("#steer-status"), "steer-status-saved");
});

test("steer: find a direction, push and erase it, import, delete", async ({ page }) => {
  await scenario(page, "steer-directions");
  page.on("dialog", (d) => d.accept());
  await page.goto("/");
  await expect(page.locator("#dir-line")).toBeVisible();
  await page.locator("#dir-new").click();
  await expect(page.locator("#dn")).toBeVisible();
  await page.locator("#dn-kind [data-k=topic]").click();
  await expect(page.locator("#dn-pos-opt")).toHaveText("one prompt per line");
  await page.locator("#dn-ex").click();
  await expect(page.locator("#dn-kind [data-k=persona]")).toHaveClass(/\bon\b/);
  await page.locator("#dn-name").fill("Nationalist test");
  await snap(page.locator("#dn"), "direction-form");

  await page.locator("#dn-go").click();
  await expect(page.locator("#dn-result")).toBeVisible({ timeout: T });
  await idle(page);
  await expect(page.locator("#dn-facts dd")).toHaveCount(3);
  await snap(page.locator("#dn"), "direction-result");

  await page.locator("#dn-on").click();
  const row = page.locator("#rc-list .rc-row.dir");
  await expect(row).toHaveCount(1);
  await expect(row.locator(".dose")).toHaveText("+4.0");
  await row.locator("[data-op=erase]").click();
  await expect(row.locator(".dose")).toHaveText("+1.00");
  await snap(page.locator("#rc"), "direction-erase-row");
  await row.locator("[data-op=push]").click();
  await row.locator(".band").fill("12 14 99");
  await row.locator(".band").press("Tab");
  await expect(row.locator(".band")).toHaveValue("12 14");
  await expect(page.locator("#steer-facts")).toContainText("Directions on");
  await snap(page.locator("#rc"), "direction-push-row");

  await page.locator("#t-prompt").fill("Should the country close its borders?");
  await page.locator("#t-go").click();
  await idle(page);
  await expect(page.locator("#tl .tl-a.cur .tl-n")).toHaveText("1 direction");
  await snap(page.locator("#tl"), "direction-answer");

  // a research file made for another model is refused with a reason
  await page.locator("#dn-path").fill("../research/political_steering/out/directions.npz");
  await page.locator("#dn-import").click();
  await expect(page.locator("#dn-status")).not.toHaveClass(/\blive\b/);
  await expect(page.locator("#dn-status")).not.toHaveText(/Importing/);
  await snap(page.locator("#dn-status"), "direction-import-error");

  await page.locator('#dir-chips .dir-chip:has-text("Nationalist test") .del').click();
  await expect(page.locator('#dir-chips .dir-chip:has-text("Nationalist test")')).toHaveCount(0);
  await expect(row).toHaveClass(/\bgone\b/);
  await snap(page.locator("#st-s1"), "direction-deleted");
});

test("steer: describe a behaviour and let it find the neurons", async ({ page }) => {
  await scenario(page, "steer-describe");
  await page.goto("/");
  await page.locator("#c-input").fill("talk like a pirate");
  await expect(page.locator("#c-status")).toContainText("Takes a few minutes");
  await page.locator("#c-go").click();
  await expect(page.locator("#c-results .plan")).toBeVisible();
  await ran(page.locator("#c-results .plan-now"));
  await idle(page);
  await snap(page.locator("#st-s1"), "describe-done");
  const apply = page.locator("#c-results .plan-apply");
  if (await apply.isVisible()) {
    await apply.click();
    await expect(page.locator("#tl .tl-a")).toHaveCount(1);
    await idle(page);
    await snap(page.locator("#c-results .plan-foot"), "describe-applied");
    await snap(page.locator("#tl"), "describe-answer");
  }
});

test("steer: save the recipe as a new model", async ({ page }) => {
  await scenario(page, "steer-export");
  await page.goto("/");
  await page.locator("#saved-list .saved-item").first().click();
  await expect(page.locator("#st-s3")).not.toHaveClass(/\blocked\b/);
  await expect(page.locator("#st-abl-row")).toBeHidden();
  await page.locator("#st-name").fill("parity-steered");
  await page.locator("#st-bake").click();
  await expect(page.locator("#st-saved")).toBeVisible({ timeout: T });
  await expect(page.locator("#st-saved b")).toHaveText("parity-steered");
  await snap(page.locator("#st-s3"), "steer-exported");
});

test("abliterate: find the refusal direction, switch it, try it, save", async ({ page }) => {
  await scenario(page, "abliterate");
  await page.goto("/");
  await page.keyboard.press("Meta+1");
  await expect(page.locator("#view-abliterate")).toBeVisible();
  await expect(page.locator("#abl-go")).toHaveText("Find it");
  await page.locator("#abl-go").click();
  await expect(page.locator("#abl-plan")).toBeVisible();
  await ran(page.locator("#abl-plan .plan-now"));
  await idle(page);
  await expect(page.locator("#abl-toggle")).toBeVisible();
  await expect(page.locator("#abl-toggle")).toHaveAttribute("aria-checked", "true");
  await expect(page.locator("#tab-abl-dot")).toBeVisible();
  await snap(page.locator("#abl-s1"), "abl-found");
  await page.locator("#abl-rank summary").click();
  await snap(page.locator("#abl-rank"), "abl-rank");
  await snap(page.locator("#abl-status"), "abl-status-on");

  await page.locator("#abl-toggle").click();
  await expect(page.locator("#abl-toggle")).toHaveAttribute("aria-checked", "false");
  await expect(page.locator("#tab-abl-dot")).toBeHidden();
  await snap(page.locator("#abl-status"), "abl-status-off");
  await page.locator("#abl-toggle").click();
  await expect(page.locator("#abl-toggle")).toHaveAttribute("aria-checked", "true");

  // try it: the finder's two examples open the thread; ask one of the suggestions, then make it stronger
  const answers = page.locator("#at .tl-a"), cur = page.locator("#at .tl-a.cur");
  await expect(answers).toHaveCount(2);
  await snap(page.locator("#abl-s2"), "abl-examples");
  await page.locator("#abl-sugg .chip").first().click();
  await expect(answers).toHaveCount(3);
  await idle(page);
  await expect(cur.locator(".tl-n").first()).toHaveText(/^(Refused|Answered)/);
  await cur.locator(".tl-act", { hasText: "Stronger" }).click();
  await expect(page.locator("#at .tl-note")).toContainText("Stronger");
  await expect(answers).toHaveCount(4);
  await idle(page);
  await expect(page.locator("#abl-facts")).toContainText((await cur.locator(".tl-tag").first().textContent())!);
  await expect(page.locator("#abl-name")).toHaveValue(/-abliterated/);
  await snap(page.locator("#abl-s2"), "abl-tried");

  await page.locator("#abl-name").fill("parity-abliterated");
  await page.locator("#abl-bake").click();
  await expect(page.locator("#abl-saved")).toBeVisible({ timeout: T });
  await expect(page.locator("#abl-saved b")).toHaveText("parity-abliterated");
  await snap(page.locator("#abl-s3"), "abl-exported");

  await page.keyboard.press("Meta+0");
  await expect(page.locator("#hc-abliterate")).toContainText("Refusals erased at layer");
  await snap(page, "home-after");
});

test("chat: starters, markdown, switches, retry, settings, new chat", async ({ page }) => {
  await scenario(page, "chat");
  await page.goto("/");
  await page.keyboard.press("Meta+3");
  await expect(page.locator("#view-chat")).toBeVisible();
  await expect(page.locator("#chat-input")).toBeFocused();
  await expect(page.locator('#chat-mods .mod[data-mod="abl"]')).toHaveClass(/\bon\b/);
  await snap(page, "chat-empty");
  // erased at strength 1.5, the 0.6B model answers with nothing at all, so chat with the switch off
  await page.locator('#chat-mods .mod[data-mod="abl"]').click();
  await expect(page.locator('#chat-mods .mod[data-mod="abl"]')).not.toHaveClass(/\bon\b/);

  await page.locator("#chat-opts summary").click();
  await page.locator("#chat-temp").fill("0");
  await expect(page.locator("#chat-temp-v")).toHaveText("0.00");
  await page.locator("#chat-max").fill("192");
  await expect(page.locator("#chat-max-v")).toHaveText("192");
  await snap(page, "chat-settings");
  await page.locator("#chat-model").click(); // a click elsewhere closes the settings
  await expect(page.locator("#chat-opts")).not.toHaveAttribute("open");

  await page.locator("#chat-starters .starter", { hasText: "Write a haiku about Mondays" }).click();
  await expect(page.locator("#chat-log .msg.user")).toHaveCount(1);
  await idle(page);
  await expect(page.locator("#chat-send")).toBeVisible();
  await snap(page, "chat-reply");

  await page.locator("#chat-input").fill("List three fruits as a markdown bulleted list, each name in bold. /no_think");
  await expect(page.locator("#chat-send")).toBeEnabled();
  await page.locator("#chat-input").press("Enter");
  await expect(page.locator("#chat-log .msg.user")).toHaveCount(2);
  await idle(page);
  await snap(page, "chat-markdown");

  await page.locator("#chat-log .msg.bot").last().locator('[data-a="retry"]').click({ timeout: 15_000 });
  await idle(page);
  await expect(page.locator("#chat-log .msg.bot")).toHaveCount(2);
  await snap(page, "chat-retry");
  await page.locator('#chat-mods .mod[data-mod="abl"]').click();
  await expect(page.locator('#chat-mods .mod[data-mod="abl"]')).toHaveClass(/\bon\b/);

  await page.keyboard.press("Meta+k");
  await expect(page.locator("#chat-log .msg")).toHaveCount(0);
  await expect(page.locator("#chat-empty")).toBeVisible();
  if (!RECORD) await expect(page.locator("#chat-log")).toBeEmpty();
});
