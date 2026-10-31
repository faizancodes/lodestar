// Renders the site's ASCII animations to GIFs for the README, in docs/media/. With the dev server running
// (pnpm dev): node scripts/readme-gifs.mjs [name ...]. Each one is stepped frame by frame on capture.html's (or
// og.html's) stepped clock in headless Chrome, so the timing is exact however slow the capture, and ffmpeg makes
// the GIF from a palette of its own frames.
import { spawn, execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const SITE = "http://localhost:5180";
const OUT = resolve(import.meta.dirname, "../../docs/media");
const CHROME = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9333;

// name → page, size, frames per second, seconds, and how a frame is made
const scene = (name, seconds = 8) => ({
  name, url: `capture.html?scene=${name}&w=800&h=420`, w: 800, h: 420, fps: 12, seconds,
  // it opens on the finished picture (what a still preview shows), then plays the story over five seconds, as
  // if scrolled through, and holds
  frame: (i, fps) => { const s = i / fps; return `__frame(${2 + s}, ${s < 1.5 ? 1 : Math.min(1, (s - 1.5) / 5)})`; },
});
const clock = (name, url, w, h, fps, seconds, scale) => ({
  name, url, w, h, fps, seconds, scale, warm: 1000, frame: (_, fps) => `__step(${1000 / fps})`,
});
const JOBS = [
  clock("banner", "og.html?step", 1200, 630, 10, 6, 880),
  scene("erase"), scene("layers"), scene("steer"), scene("chat"), scene("api"),
  clock("assemble", "capture.html?scene=get&w=800&h=480", 800, 480, 10, 13.5),
  clock("footer", "capture.html?scene=foot&w=960&h=520", 960, 520, 12, 6),
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function cdp() {
  const profile = mkdtempSync(join(tmpdir(), "lodestar-chrome-"));
  const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
    "--hide-scrollbars", "--force-device-scale-factor=1", "about:blank"], { stdio: "ignore" });
  let page;
  for (let k = 0; k < 50 && !page; k++) {
    await sleep(200);
    page = await fetch(`http://127.0.0.1:${PORT}/json/list`).then((r) => r.json()).then((l) => l.find((t) => t.type === "page")).catch(() => null);
  }
  if (!page) throw new Error("Chrome didn't start");
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r, { once: true }));
  let id = 0;
  const waiting = new Map();
  ws.addEventListener("message", (e) => {
    const m = JSON.parse(e.data);
    if (m.id && waiting.has(m.id)) (waiting.get(m.id)(m), waiting.delete(m.id));
  });
  const send = (method, params = {}) => new Promise((r) => { waiting.set(++id, r); ws.send(JSON.stringify({ id, method, params })); })
    .then((m) => { if (m.error) throw new Error(`${method}: ${m.error.message}`); return m.result; });
  const close = async () => {
    ws.close();
    const gone = new Promise((r) => chrome.once("exit", r));
    chrome.kill();
    await gone; // before its profile goes, or it is still writing there
    rmSync(profile, { recursive: true, force: true });
  };
  return { send, close };
}

const only = process.argv.slice(2);
mkdirSync(OUT, { recursive: true });
const { send, close } = await cdp();
try {
  for (const job of JOBS.filter((j) => !only.length || only.includes(j.name))) {
    const dir = mkdtempSync(join(tmpdir(), `lodestar-${job.name}-`));
    await send("Emulation.setDeviceMetricsOverride", { width: job.w, height: job.h, deviceScaleFactor: 1, mobile: false });
    await send("Page.navigate", { url: `${SITE}/${job.url}` });
    await sleep(2500); // modules, fonts, the first observer callbacks
    if (job.warm) for (let t = 0; t < job.warm; t += 1000 / job.fps) await send("Runtime.evaluate", { expression: job.frame(0, job.fps) });
    const n = Math.round(job.fps * job.seconds);
    for (let i = 0; i < n; i++) {
      await send("Runtime.evaluate", { expression: job.frame(i, job.fps) });
      const { data } = await send("Page.captureScreenshot", { format: "png" });
      writeFileSync(join(dir, `${String(i).padStart(4, "0")}.png`), Buffer.from(data, "base64"));
    }
    const scale = job.scale ? `scale=${job.scale}:-1:flags=lanczos,` : "";
    const out = join(OUT, `${job.name}.gif`);
    execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-framerate", String(job.fps), "-i", join(dir, "%04d.png"), "-filter_complex",
      `[0:v]${scale}split[a][b];[a]palettegen=max_colors=32:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle`, out]);
    rmSync(dir, { recursive: true, force: true });
    console.log(`${job.name}.gif`, `${n} frames`);
  }
} finally {
  await close();
}
