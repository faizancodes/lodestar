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
  await expect(page.locator('.tab[aria-selected="true"]')).toHaveCount(0); // nothing to show yet
  await expect(page.locator("#model-status .txt")).toHaveText("No model loaded");
  await snap(page.locator("aside.side"), "sidebar-empty");

  // a tab picked now is kept for when a model is ready, and the start screen points at the models
  await page.locator('.tab[data-tab="chat"]').click();
  await expect(page.locator('.tab[data-tab="chat"]')).toHaveClass(/\bwant\b/);
  await expect(page.locator("#start h2")).toHaveText("Pick a model to chat with.");
  await expect(page.locator("#start-models")).toHaveClass(/\bnudge\b/);
  await page.keyboard.press("Meta+1");
  await expect(page.locator("#start h2")).toHaveText("Pick a model to abliterate.");
  await expect(page.locator(".tab.want")).toHaveCount(1);
  await snap(page, "start-want");
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

  // other apps reach the model at /v1, under the names it lists
  await expect(page.locator("#api-url")).toHaveText(/\/v1$/);
  await expect(page.locator("#api-models dt")).toHaveText(["Qwen3-0.6B", "Qwen3-0.6B-plain"]);
  await expect(page.locator('#api-models [data-model="Qwen3-0.6B"] dd')).toHaveText("As set in Lodestar: Plain right now");
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

  await page.locator("#mm-custom").fill("org/some-model");
  await page.mouse.click(900, 450); // a click outside closes it
  await expect(page.locator("#model-menu")).toBeHidden();

  await page.locator("#model-pick").click(); // and it opens again with nothing typed
  await expect(page.locator("#mm-custom")).toHaveValue("");
  await page.mouse.click(900, 450);
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

  // ask it something: the reply comes with the neurons on, and the box takes the next message
  const answers = page.locator("#tl .tl-a"), cur = page.locator("#tl .tl-a.cur");
  await page.locator("#t-chips .chip", { hasText: "Describe the way you look." }).click();
  await expect(answers).toHaveCount(1);
  await idle(page);
  await expect(cur.locator(".tl-a-b").first()).not.toBeEmpty();
  await expect(page.locator("#t-prompt")).toHaveAttribute("placeholder", "Reply");
  await expect(page.locator("#tl-fix")).toBeVisible();
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

  // say what's wrong under the reply: the neurons are retuned and it's written again as a second version
  await page.locator("#tl-fix .chip", { hasText: "It's too strong" }).click();
  await expect(page.locator("#tl .tl-now")).toBeVisible();
  await expect(page.locator("#tl .tl-now")).toHaveCount(0, { timeout: T });
  await expect(page.locator("#tl .typing")).toHaveCount(0, { timeout: T });
  await idle(page);
  await expect(answers).toHaveCount(1);
  await snap(page.locator("#st-s2"), "fix-result");

  // the arrows go back to the first version, and its neurons
  if (await cur.locator(".tl-ver").isVisible()) {
    await expect(cur.locator(".tl-ver")).toContainText("2 of 2");
    await cur.locator('[aria-label="Previous version"]').click();
    await expect(cur.locator(".tl-ver")).toContainText("1 of 2");
    await expect(rows).toHaveCount(2);
  }

  // New chat starts over
  await page.locator("#tl-new").click();
  await expect(page.locator("#tl")).toBeHidden();
  await expect(page.locator("#t-prompt")).toHaveAttribute("placeholder", "Ask it something");

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
  await expect(page.locator("#dn-name")).toBeFocused();

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

  // Import with no path asks for one; a research file made for another model is refused with a reason
  await page.locator("#dn-import").click();
  await expect(page.locator("#dn-path")).toBeFocused();

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
  await snap(page.locator("#abl-s1"), "abl-find");

  await page.locator("#abl-edit").click();
  await expect(page.locator("#abl-probes-harmful")).toBeVisible();
  await snap(page.locator("#abl-s1"), "abl-probes");
  await page.locator("#abl-edit").click();
  await expect(page.locator("#abl-probes")).toHaveCount(0);

  await page.locator("#abl-go").click();
  await expect(page.locator("#abl-plan")).toBeVisible();
  await ran(page.locator("#abl-plan .plan-now"));
  await idle(page);

  await expect(page.locator("#abl-toggle")).toBeVisible();
  await expect(page.locator("#abl-toggle")).toHaveAttribute("aria-checked", "true");
  await expect(page.locator("#tab-abl-dot")).toBeVisible();
  await snap(page.locator("#abl-s1"), "abl-found");

  await expect(page.locator("#abl-result")).toContainText("16 new requests it should refuse");
  await expect(page.locator('#abl-result [data-row="before"] .abl-res-n')).toHaveText("14 refused");
  await expect(page.locator('#abl-result [data-row="now"] .abl-res-n')).toHaveText("0 refused");
  await expect(page.locator("#abl-s1 [data-cancel]")).toHaveCount(0);
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

// The recording is assembled from chat's (the plain column is its haiku; erased at strength 1.5, the 0.6B model
// answers with nothing), so it's replayed only
test("chat: compare replies side by side", async ({ page }) => {
  test.skip(RECORD);

  await scenario(page, "chat-compare");
  await page.goto("/");
  await page.keyboard.press("Meta+3");
  await expect(page.locator("#chat-input")).toBeFocused();

  const sw = page.locator("#chat-compare");
  await expect(sw).toHaveAttribute("aria-checked", "false");
  await sw.click();
  await expect(sw).toHaveAttribute("aria-checked", "true");

  await page.locator("#chat-opts summary").click();
  await page.locator("#chat-temp").fill("0");
  await page.locator("#chat-max").fill("192");
  await page.locator("#chat-model").click();

  await page.locator("#chat-starters .starter", { hasText: "Write a haiku about Mondays" }).click();
  await idle(page);
  await expect(page.locator("#chat-send")).toBeVisible();

  const cols = page.locator("#chat-log .msg.bot.cmp .cmp-col");
  await expect(cols).toHaveCount(2);
  await expect(cols.nth(0).locator(".cmp-h")).toHaveText("Plain");
  await expect(cols.nth(0).locator(".md")).toContainText("Mondays are quiet");
  await expect(cols.nth(1).locator(".cmp-h")).toHaveText("Refusals erased");
  await expect(cols.nth(1).locator(".msg-none")).toHaveText("No reply");
  await snap(page, "chat-compare");

  // the choice is remembered
  await page.reload();
  await page.keyboard.press("Meta+3");
  await expect(page.locator("#chat-compare")).toHaveAttribute("aria-checked", "true");
});

// Assembled by hand: a Mac that's swapping, with another Python process holding a model (the machine isn't part of
// what the recordings capture), so it's replayed only
test("memory: the meter, a model that won't fit, unload", async ({ page }) => {
  test.skip(RECORD);

  await scenario(page, "memory");
  await page.goto("/");
  await expect(page.locator("#mem")).toHaveClass(/\btight\b/);
  await expect(page.locator("#mem .mem-l")).toHaveText("Swapping");
  await expect(page.locator("#mem .mem-n")).toHaveText("31 / 32 GB");

  await page.locator("#model-pick").click();
  await page.locator('#model-menu .mm-row[data-id="Qwen/Qwen3-1.7B"] .mm-item').click();
  await expect(page.locator("#fit-warn")).toBeVisible();
  await expect(page.locator("#fit-warn h3")).toHaveText("Qwen3-1.7B won’t fit in memory");
  await expect(page.locator("#fit-warn .fit-proc")).toHaveText(/Python \(repl\.py\)\s*19 GB/);
  await snap(page, "memory-fit");

  await page.keyboard.press("Escape");
  await expect(page.locator("#fit-warn")).toHaveCount(0);

  await page.locator("#model-pick").click();
  await page.locator('#model-menu .mm-row .mact[data-act="unload"]').click();
  await expect(page.locator("#model-status .txt")).toHaveText("No model loaded");
  await expect(page.locator("#start")).toBeVisible();
});

// Saving and reopening go through the mock's stand-in for saved chats; the reply is chat's haiku
test("chat: saved chats, reopened", async ({ page }) => {
  test.skip(RECORD);

  await scenario(page, "chats");
  await page.goto("/");
  await page.keyboard.press("Meta+3");
  await expect(page.locator("#chat-list")).toHaveCount(0);

  await page.locator('#chat-mods .mod[data-mod="abl"]').click();
  await page.locator("#chat-opts summary").click();
  await page.locator("#chat-temp").fill("0");
  await page.locator("#chat-max").fill("192");
  await page.locator("#chat-model").click();

  await page.locator("#chat-starters .starter", { hasText: "Write a haiku about Mondays" }).click();
  await idle(page);
  await expect(page.locator("#chat-list .saved-item")).toHaveCount(1);
  await expect(page.locator("#chat-list .saved-item.on .nm")).toHaveText("Write a haiku about Mondays");

  await page.keyboard.press("Meta+k");
  await expect(page.locator("#chat-log .msg")).toHaveCount(0);
  await expect(page.locator("#chat-list .saved-item.on")).toHaveCount(0);

  // reopened as it ran: the reply, and the settings it had
  await page.locator("#chat-list .saved-item").click();
  await expect(page.locator("#chat-log .msg.bot .md")).toContainText("Mondays are quiet");
  await expect(page.locator("#chat-list .saved-item.on")).toHaveCount(1);
  await expect(page.locator("#chat-temp-v")).toHaveText("0.00");
  await expect(page.locator("#chat-max-v")).toHaveText("192");
  await snap(page, "chat-saved");

  await page.locator("#chat-list .saved-item .del").click({ force: true });
  await expect(page.locator("#chat-list")).toHaveCount(0);
});

// Assembled by hand from the chat recording, with made-up neurons (their words and replies aren't the model's), so
// it's replayed only
test("chat: neurons pushed by hand", async ({ page }) => {
  test.skip(RECORD);

  await scenario(page, "neurons");
  await page.goto("/");
  await page.keyboard.press("Meta+3");
  await page.locator("#chat-neurons-btn").click();
  await expect(page.locator("#chat-add")).toBeFocused();

  // one this model doesn't have, then two pasted the way they're shared
  const add = page.locator("#chat-add");
  await add.fill("L14 N8949 +1");
  await add.press("Enter");
  await expect(page.locator("#chat-add-err")).toHaveText("There’s no neuron 8949: each layer has neurons 0–3071.");

  await add.fill("L14 N2949 +1 and L13 N1956 −1");
  await add.press("Enter");
  await expect(page.locator("#chat-add-err")).toBeHidden();
  await expect(page.locator(".nr-pin")).toHaveCount(2);
  await expect(page.locator('.nr-pin[data-pin="14:2949"] .nr-lens')).toHaveText("villain · sinister · cruel · wicked");
  await expect(page.locator('.nr-pin[data-pin="13:1956"] .nr-lens')).toHaveText("deceit · scheme · malice · lie"); // pushed down
  await expect(page.locator('#chat-mods .mod[data-mod="steer"]')).toHaveText("Steered · 2 neurons");
  await expect(page.locator("#tab-steer-count")).toHaveText("2"); // they're the recipe that's on

  await page.locator("#chat-compare").click();
  await page.locator("#chat-starters .starter", { hasText: "free afternoon" }).click();
  await idle(page);
  await expect(page.locator("#chat-send")).toBeVisible();

  const cols = page.locator("#chat-log .msg.bot.cmp .cmp-col");
  await expect(cols.nth(0).locator(".cmp-h")).toHaveText("Plain");
  await expect(cols.nth(0).locator(".md")).toContainText("long walk");
  await expect(cols.nth(1).locator(".cmp-h")).toHaveText("Steered");
  await expect(cols.nth(1).locator(".md")).toContainText("plotting");
  await expect(cols.nth(0).locator(".msg-mod")).toHaveCount(0);
  await expect(cols.nth(1).locator(".msg-mod")).toHaveText(["L14 N2949 +1", "L13 N1956 −1"]);
  await snap(page, "chat-neurons");

  // a dose nudged, a neuron taken out
  await page.locator('.nr-pin[data-pin="14:2949"] [aria-label="More"]').click();
  await expect(page.locator('.nr-pin[data-pin="14:2949"] input')).toHaveValue("+1.25");
  await page.locator('.nr-pin[data-pin="13:1956"]').hover();
  await page.locator('.nr-pin[data-pin="13:1956"] .nr-del').click();
  await expect(page.locator(".nr-pin")).toHaveCount(1);
  await expect(page.locator('#chat-mods .mod[data-mod="steer"]')).toHaveText("Steered · 1 neuron");

  // or switched on from the neurons behind a word
  await page.locator('#chat-pick-by [data-by="map"]').click();
  await expect(page.locator("#chat-find")).toBeFocused();
  await page.locator("#chat-find").fill("evil");
  await page.locator("#chat-find").press("Enter");
  await expect(page.locator("#chat-found .nr-card")).toHaveCount(14);
  const card = page.locator('#chat-found [data-card="18:2207"]');
  await expect(card).toContainText("L18N2207up");
  await expect(card.locator(".nr-lens")).toHaveText("villain · evil · wicked");
  await card.click();
  await expect(card).toHaveAttribute("aria-checked", "true");
  await expect(page.locator('.nr-pin[data-pin="18:2207"] input')).toHaveValue("+0.36");
  await snap(page.locator("#chat-neurons"), "chat-neuron-map");

  // the panel stays open, on the way it was used last
  await page.reload();
  await page.keyboard.press("Meta+3");
  await expect(page.locator("#chat-neurons")).toBeVisible();
  await expect(page.locator('#chat-pick-by [data-by="map"]')).toHaveClass(/\bon\b/);
  await page.locator("#chat-neurons .nr-x").click();
  await expect(page.locator("#chat-neurons")).toHaveCount(0);
});

// Assembled from chat's haiku, cut off partway and then carried on
test("chat: continue a reply that ran out of length", async ({ page }) => {
  test.skip(RECORD);

  await scenario(page, "chat-continue");
  await page.goto("/");
  await page.keyboard.press("Meta+3");
  await page.locator('#chat-mods .mod[data-mod="abl"]').click();

  await page.locator("#chat-opts summary").click();
  await page.locator("#chat-temp").fill("0");
  await page.locator("#chat-max").fill("192");
  await page.locator("#chat-model").click();

  await page.locator("#chat-starters .starter", { hasText: "Write a haiku about Mondays" }).click();
  await idle(page);
  const bot = page.locator("#chat-log .msg.bot");
  await expect(bot.locator(".md")).toHaveText(/The world$/);
  await snap(page, "chat-cut");

  await bot.locator('[data-a="continue"]').click();
  await idle(page);
  await expect(bot).toHaveCount(1);
  await expect(bot.locator(".md")).toContainText("A quiet day.");
  await expect(bot.locator('[data-a="continue"]')).toHaveCount(0);
  await snap(page, "chat-continued");
});

// Assembled from chat's haiku, with a direction to read and made-up readings
test("chat: gauges read a direction as the reply is written", async ({ page }) => {
  test.skip(RECORD);

  await scenario(page, "chat-gauges");
  await page.goto("/");
  await page.keyboard.press("Meta+3");
  await page.locator('#chat-mods .mod[data-mod="abl"]').click();

  await page.locator("#chat-opts summary").click();
  await page.locator("#chat-temp").fill("0");
  await page.locator("#chat-max").fill("192");
  await page.locator("#chat-model").click();

  await page.locator("#chat-gauges summary").click();
  const sw = page.locator('#chat-gauges .switch[data-dir="cheerful"]');
  await sw.click();
  await expect(sw).toHaveAttribute("aria-checked", "true");
  await expect(page.locator("#chat-gauges summary")).toHaveText("Gauges · 1");
  await snap(page, "chat-gauges-pick");
  await page.locator("#chat-model").click();
  await expect(page.locator("#chat-gauges")).not.toHaveAttribute("open");

  await page.locator("#chat-starters .starter", { hasText: "Write a haiku about Mondays" }).click();
  await idle(page);
  const g = page.locator('#chat-log .msg.bot .gauge[data-gauge="cheerful"]');
  await expect(g.locator(".gauge-n")).toHaveText("Cheerful");
  await expect(g.locator(".gauge-strip i")).toHaveCount(17);
  await expect(g.locator(".gauge-v")).toHaveText(/^[+−]\d\.\d\d$/);

  await g.locator(".gauge-strip i").nth(2).hover();
  await expect(page.locator(".gauge-tip b")).toHaveText("ays");
  await expect(g.locator(".gauge-v")).toHaveText("−0.24");
  await snap(page, "chat-gauges");
});
