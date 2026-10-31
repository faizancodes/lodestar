// The app itself, acted out: a prompt is typed and sent, the reply streams in with each word tinted by a gauge, then
// a pointer opens one word and flips its layer table from the logit lens to the J-lens. Drawn in HTML at the app's
// own size (1180×720) and scaled to fit, so it reads as a screenshot that moves. Plays only while on screen.
import { markSVG } from "../../web/src/art/mark";

const W = 1180, H = 720;
const PROMPT = "Write a haiku about Mondays";

// the reply, a line at a time; each word with its reading on a "Cheerful" gauge, −1..1
const REPLY: [string, number][][] = [
  [["Mondays", -0.55], [" are", -0.2], [" quiet", -0.62], [",", -0.1]],
  [["The", 0.05], [" world", 0.12], [" feels", -0.3], [" still", -0.78], [",", -0.15]],
  [["A", 0.1], [" quiet", -0.4], [" day", 0.25], [" begins", 0.58], [".", 0.3]],
];
const PICK = [1, 3]; // the word the pointer opens: "still"

// what each layer makes of that word: the logit lens (what it expects next) and the J-lens (what it holds ready to
// say, with the runners-up)
const LOGIT: [number, string, number][] = [
  [0, " always", 0.04], [3, "ấy", 0.03], [6, "ra", 0.05], [9, "mit", 0.06], [12, " and", 0.1],
  [15, " the", 0.12], [18, ",", 0.3], [21, ",", 0.48], [24, ",", 0.66], [27, ",", 0.84],
];
const JLENS: [number, string, number, string][] = [
  [0, " still", 0.08, " yet  even"], [3, " still", 0.12, " calm  yet"], [6, " calm", 0.19, " quiet  still"],
  [9, " quiet", 0.27, " calm  hush"], [12, " quiet", 0.35, " silent  calm"], [15, " silent", 0.42, " quiet  hush"],
  [18, " hush", 0.48, " silent  calm"], [21, " quiet", 0.56, " still  slow"], [24, " still", 0.65, " quiet  calm"],
  [27, " still", 0.74, " quiet  slow"],
];

/** A gauge reading as the app colours it: blue below zero, amber above, faint near it. */
const tint = (v: number, a = 1) => (v < 0 ? `rgba(122, 162, 247, ${a * Math.min(1, 0.15 + Math.abs(v))})` : `rgba(240, 180, 90, ${a * Math.min(1, 0.15 + v)})`);
const show = (t: string) => t.replace(/^ /, "");

const ICONS = {
  abl: '<path d="M3 3l10 10M8 1.8l5 2v4.1c0 2.6-2.1 4.7-5 6.3-2.9-1.6-5-3.7-5-6.3V3.8"/>',
  steer: '<circle cx="8" cy="8" r="6.2"/><path d="M10.6 5.4 9.2 9.2 5.4 10.6 6.8 6.8z"/>',
  chat: '<path d="M2.5 7.6c0-2.7 2.5-4.6 5.5-4.6s5.5 1.9 5.5 4.6-2.5 4.6-5.5 4.6c-.8 0-1.6-.1-2.3-.4L2.8 13l.7-2.4c-.6-.8-1-1.8-1-3z"/>',
};
const icon = (d: string) => `<svg viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round" stroke-linecap="round">${d}</svg>`;

function build(root: HTMLElement) {
  root.innerHTML = `
  <div class="ad-stage" style="width:${W}px;height:${H}px">
    <div class="ad-bar"><i></i><i></i><i></i></div>
    <aside class="ad-side">
      <div class="ad-brand">${markSVG(26, 21)}<b>Lodestar</b></div>
      <nav><span>${icon(ICONS.abl)}Abliterate</span><span>${icon(ICONS.steer)}Steer</span><span class="on">${icon(ICONS.chat)}Chat</span></nav>
      <div class="ad-sec">Saved</div><div class="ad-item">calm + slow <em>2</em></div>
      <div class="ad-sec">Chats</div><div class="ad-item on ad-title">New chat</div>
      <div class="ad-model">
        <div class="ad-m1"><i></i><b>Qwen3-8B</b></div><div class="ad-m2">Ready</div>
        <div class="ad-mem"><span>Memory</span><span class="ad-membar">${Array.from({ length: 16 }, (_, i) => `<i${i < 8 ? ' class="on"' : ""}></i>`).join("")}</span><span>17 / 32 GB</span></div>
      </div>
    </aside>
    <section class="ad-main">
      <header><b>Qwen3-8B</b><span class="ad-pill">Refusals erased</span><span class="ad-sp"></span><span class="ad-b">Gauges · 1</span><span class="ad-b">Neurons</span><span class="ad-b">New chat</span></header>
      <div class="ad-chat">
        <div class="ad-user"></div>
        <div class="ad-reply"></div>
        <div class="ad-gauge"><span>Cheerful</span><div class="ad-cells"></div><code>+0.00</code></div>
      </div>
      <div class="ad-comp"><span class="ad-typed"></span><span class="ad-caret"></span><span class="ad-ph">Message the model</span><i class="ad-send"></i></div>
    </section>
    <aside class="ad-panel">
      <div class="ad-panel-h"><b>Inside the reply</b><span>×</span></div>
      <div class="ad-word"><span>still</span></div>
      <div class="ad-lh"><h4>What each layer expects next</h4><div class="ad-seg"><span class="on">Logit lens</span><span>J-lens</span></div></div>
      <table class="ad-guess"><tbody></tbody></table>
    </aside>
    <div class="ad-ptr"><svg viewBox="0 0 16 22" width="16" height="22"><path d="M1 1v17l4.6-4.3 3 7 2.6-1.1-3-6.9H14z" fill="#fff" stroke="#0a0a0a" stroke-width="1.2" stroke-linejoin="round"/></svg></div>
  </div>`;
}

type Run = { on: boolean };
const sleep = (ms: number, run: Run) => new Promise<void>((ok, stop) => setTimeout(() => (run.on ? ok() : stop(new Error("stopped"))), ms));

/** What the frame's corner readouts say as it plays, like the hero's layer and activation. */
type Hud = { state: (t: string) => void; read: (t: string) => void };

export function startAppDemo(root: HTMLElement, hud: Hud) {
  build(root);
  const stage = root.querySelector<HTMLElement>(".ad-stage")!;
  const q = (s: string) => stage.querySelector<HTMLElement>(s)!;
  const user = q(".ad-user"), reply = q(".ad-reply"), cells = q(".ad-cells"), read = q(".ad-gauge code"), gauge = q(".ad-gauge");
  const typed = q(".ad-typed"), comp = q(".ad-comp"), title = q(".ad-title"), panel = q(".ad-panel"), ptr = q(".ad-ptr");
  const body = q(".ad-guess tbody"), seg = stage.querySelectorAll<HTMLElement>(".ad-seg span"), lh = q(".ad-lh h4");

  // the app's own size, scaled to the width it's given
  const fit = () => {
    const k = root.clientWidth / W;
    stage.style.transform = `scale(${k})`;
    root.style.height = `${H * k}px`;
  };
  new ResizeObserver(fit).observe(root);

  // where an element sits, in the stage's own pixels
  const at = (el: HTMLElement, fx = 0.5, fy = 0.5) => {
    const s = stage.getBoundingClientRect(), r = el.getBoundingClientRect(), k = s.width / W;
    return [(r.left - s.left + r.width * fx) / k, (r.top - s.top + r.height * fy) / k];
  };
  const point = (el: HTMLElement, ms: number, fx?: number, fy?: number) => {
    const [x, y] = at(el, fx, fy);
    ptr.style.transitionDuration = `${ms}ms`;
    ptr.style.transform = `translate(${x}px, ${y}px)`;
  };
  const click = async (run: Run) => { ptr.classList.add("down"); await sleep(140, run); ptr.classList.remove("down"); };

  const rows = (mode: "logit" | "j") => {
    lh.textContent = mode === "j" ? "What each layer holds" : "What each layer expects next";
    seg.forEach((s, i) => s.classList.toggle("on", (i === 1) === (mode === "j")));
    const data = mode === "j" ? JLENS : LOGIT.map(([l, t, p]) => [l, t, p, ""] as const);
    body.innerHTML = data.map(([l, t, p, m], i) =>
      `<tr style="--i:${i}"><td class="l">${l}</td><td class="t">${show(t)}</td><td class="m">${m.trim()}</td><td class="b"><div class="bar"><i style="--w:${Math.round(p * 100)}%"></i></div></td></tr>`).join("");
  };

  // the reply's words, laid out a line at a time, each hidden until it streams in
  const words: HTMLElement[] = [];
  const lay = () => {
    words.length = 0;
    for (const line of REPLY) {
      const p = document.createElement("p");
      for (const [w, v] of line) {
        const s = document.createElement("span");
        s.className = "ad-tok";
        s.textContent = w;
        s.style.setProperty("--tint", tint(v, 0.32));
        p.append(s);
        words.push(s);
        const c = document.createElement("i");
        c.style.background = tint(v);
        cells.append(c);
      }
      reply.append(p);
    }
  };
  const picked = () => reply.children[PICK[0]].children[PICK[1]] as HTMLElement;

  const signed = (v: number) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(3);
  const reset = () => {
    hud.state("Qwen3-8B · ready"); hud.read("Cheerful +0.000");
    user.textContent = ""; user.classList.remove("in");
    reply.innerHTML = ""; cells.innerHTML = ""; read.textContent = "+0.00"; gauge.classList.remove("in");
    typed.textContent = ""; comp.classList.remove("typing");
    title.textContent = "New chat";
    panel.classList.remove("in"); stage.classList.remove("open"); ptr.classList.remove("in");
    rows("logit");
  };

  async function play(run: Run) {
    reset();
    await sleep(700, run);

    comp.classList.add("typing");
    hud.state("Qwen3-8B · prompt");
    for (let i = 1; i <= PROMPT.length; i++) { typed.textContent = PROMPT.slice(0, i); await sleep(34 + Math.random() * 40, run); }
    await sleep(350, run);
    typed.textContent = ""; comp.classList.remove("typing");
    user.textContent = PROMPT; user.classList.add("in");
    title.textContent = PROMPT;
    await sleep(500, run);

    lay();
    gauge.classList.add("in");
    const flat = REPLY.flat();
    for (let i = 0; i < words.length; i++) {
      words[i].classList.add("in");
      cells.children[i].classList.add("in");
      const v = flat[i][1];
      read.textContent = (v < 0 ? "−" : "+") + Math.abs(v).toFixed(2);
      hud.state(`Qwen3-8B · token ${i + 1} / ${words.length}`); hud.read(`Cheerful ${signed(v)}`);
      await sleep(i % 5 === 4 ? 210 : 120, run);
    }
    await sleep(900, run);

    // the pointer opens "still"
    const word = picked();
    const [sx, sy] = at(comp, 0.7, 0.3);
    ptr.style.transitionDuration = "0ms";
    ptr.style.transform = `translate(${sx}px, ${sy}px)`;
    ptr.classList.add("in");
    await sleep(60, run);
    point(word, 900, 0.5, 0.75);
    await sleep(1000, run);
    await click(run);
    word.classList.add("picked");
    panel.classList.add("in"); stage.classList.add("open");
    hud.state("“still” · logit lens · L0–27");
    await sleep(2600, run);

    // and flips the table to the J-lens
    point(seg[1], 800, 0.5, 0.7);
    await sleep(950, run);
    await click(run);
    rows("j");
    hud.state("“still” · J-lens · L0–27");
    await sleep(900, run);
    point(body.children[5] as HTMLElement, 900, 0.22, 0.6);
    await sleep(900, run);
    for (const [l, t, p] of JLENS) { hud.state(`“still” · J-lens · L${l} ${t.trim()}`); hud.read(`p ${p.toFixed(3)}`); await sleep(340, run); }
    await sleep(800, run);

    panel.classList.remove("in"); stage.classList.remove("open"); ptr.classList.remove("in");
    word.classList.remove("picked");
    await sleep(900, run);
  }

  // for reduced motion, the end state: the reply in and the J-lens open on its word
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
    reset();
    user.textContent = PROMPT; user.classList.add("in"); title.textContent = PROMPT;
    lay(); gauge.classList.add("in");
    words.forEach((w) => w.classList.add("in")); [...cells.children].forEach((c) => c.classList.add("in"));
    picked().classList.add("picked");
    rows("j"); panel.classList.add("in"); stage.classList.add("open");
    hud.state("“still” · J-lens · L0–27");
    return;
  }
  reset();

  let run: Run = { on: false };
  const loop = async (me: Run) => { while (me.on) await play(me).catch(() => {}); };
  new IntersectionObserver(([e]) => {
    if (e.isIntersecting && !run.on) loop((run = { on: true }));
    else if (!e.isIntersecting) run.on = false;
  }, { threshold: 0.25 }).observe(root);
}
