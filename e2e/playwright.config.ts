import { defineConfig, devices } from "@playwright/test";

// MODE=record drives the real server (record.sh starts it) and writes fixtures; the default replays them through
// the mock server and compares screenshots against goldens/. UI_DIR picks the UI the mock serves (default ../ui).
const RECORD = process.env.MODE === "record";
const PORT = RECORD ? 8795 : 8796;
const UI_DIR = process.env.UI_DIR || "../ui";

export default defineConfig({
  testDir: "tests",
  snapshotPathTemplate: "goldens/{arg}{ext}",
  fullyParallel: false,
  workers: 1, // one server, one scenario at a time
  retries: 0,
  timeout: RECORD ? 60 * 60_000 : 120_000,
  reporter: [["list"]],
  expect: {
    timeout: 10_000,
    toHaveScreenshot: { animations: "disabled", caret: "hide", scale: "css", maxDiffPixels: 0 },
  },
  use: {
    ...devices["Desktop Safari"], // pywebview renders with the system WebKit
    baseURL: `http://127.0.0.1:${PORT}`,
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 1,
    reducedMotion: "reduce",
    colorScheme: "dark",
    locale: "en-US",
    timezoneId: "UTC",
    trace: "retain-on-failure",
  },
  projects: [{ name: "webkit", use: { browserName: "webkit" } }],
  webServer: RECORD ? undefined : {
    command: `uv run --project .. python harness/mock.py ${PORT} ${UI_DIR}`,
    url: `http://127.0.0.1:${PORT}/__fixture/log`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
