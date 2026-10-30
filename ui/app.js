// Lodestar UI: three tabs (Abliterate, Steer, Chat) around one loaded model. Plain modules, no build step; talks to the local API in interp/server.py.
import { finishScene, setProgress, startScene, stopScene, wake } from "./loader.js";
import { startScan } from "./scan.js";
import { startAblScene } from "./ablscene.js";
import { startSideArt, startSpinner } from "./glyphs.js";
import { markSVG } from "./mark.js";
import { drawHomeArt } from "./home.js";
import { createDecoder } from "./decode.js";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const showTok = (s) => (s === "" ? "∅" : s.replace(/\n/g, "↵").replace(/\t/g, "⇥"));
const nid = (l, n) => `L${l} · N${n}`;
const fmt = (x, d = 2) => (x >= 0 ? "+" : "−") + Math.abs(Number(x)).toFixed(d);
const modelName = (id) => (id ? id.split("/").pop() : id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const PROMPTS = ["Describe the way you look.", "Hi! Who are you?", "What's your favorite food?", "Tell me about your weekend.", "Write a short poem about the ocean."];

const state = {
  info: null,
  models: { presets: [], cached: [], local: [] },
  recipe: { id: null, name: "", items: [] },
  saved: [],
  inspect: null, // { tokens, n_prompt } of the reply last opened in Inspect
  abl: null, // /api/abl: { ready, current: { layer, coef, active, clean }, probes, defaults }
  ablDefaults: null,
};
const ablActive = () => !!state.abl?.ready && !!state.abl.current?.active;

// ------------------------------------------------------------------ plumbing
async function api(path, body, method) {
  const r = await fetch(path, {
    method: method || (body ? "POST" : "GET"),
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || j.detail || r.statusText);
  return j;
}

let busy = false;
async function stream(path, body, on) {
  if (busy) { toast("Something is already running. Stop it first.", true); return false; }
  busy = true;
  document.body.classList.add("busy");
  let ok = true;
  try {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    const rd = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await rd.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 2);
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "error") { toast(ev.message, true); ok = false; }
        else if (ev.type === "cancelled") { toast("Stopped."); ok = false; }
        else if (ev.type !== "end") on(ev);
      }
    }
  } catch (e) {
    toast(e.message, true);
    ok = false;
  } finally {
    busy = false;
    document.body.classList.remove("busy");
  }
  return ok;
}
$$("[data-cancel]").forEach((b) => (b.onclick = () => api("/api/cancel", {})));

let toastT;
function toast(msg, err) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast on" + (err ? " err" : "");
  clearTimeout(toastT);
  toastT = setTimeout(() => (t.className = "toast"), err ? 5200 : 2400);
}

// ------------------------------------------------------------------ tabs
const TABS = ["abliterate", "steer", "chat"];
let tab = (() => { try { return localStorage.getItem("tab"); } catch { return null; } })();
if (!TABS.includes(tab) && tab !== "home") tab = "steer";

function setTab(t) {
  tab = t;
  try { localStorage.setItem("tab", t); } catch {}
  document.body.dataset.tab = t;
  $$(".tab").forEach((b) => {
    b.classList.toggle("on", b.dataset.tab === t);
    b.setAttribute("aria-selected", b.dataset.tab === t);
  });
  showViews();
  if (t === "chat" && state.info?.loaded) onChatShown();
  if (t === "home") renderHome();
  if (t === "steer") renderSteerStatus();
}

function showViews() {
  const loaded = !!state.info?.loaded;
  $$(".view").forEach((v) => (v.hidden = !loaded || v.dataset.view !== tab));
  $("#main").classList.toggle("is-chat", loaded && tab === "chat");
}

$$(".tab").forEach((b) => (b.onclick = () => setTab(b.dataset.tab)));
$$(".home-card").forEach((c) => (c.onclick = () => setTab(c.dataset.go)));
$$(".hc-ic").forEach((el) => el.append($(`.tab[data-tab="${el.dataset.icon}"] svg`).cloneNode(true)));
$("#brand").onclick = () => state.info?.loaded && setTab("home");
$("#brand").onkeydown = (e) => (e.key === "Enter" || e.key === " ") && state.info?.loaded && setTab("home");

// Home: the loaded model, and a card for each page with what it's for and where it stands.
function renderHome() {
  const s = state.info;
  if (tab !== "home" || !s?.loaded) return;
  $("#home-model").textContent = modelName(s.model);
  const dev = { mps: "Apple GPU", cuda: "GPU", cpu: "CPU" }[s.device] || s.device;
  const specs = [["Parameters", s.params && `${(s.params / 1e9).toFixed(1)}B`], ["Layers", s.layers], ["Runs on", dev], ["Precision", s.dtype]];
  $("#home-meta").innerHTML = specs.filter(([, v]) => v).map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(String(v))}</dd></div>`).join("");
  const cur = state.abl?.current;
  $("#hc-abliterate").innerHTML = cur?.active ? `<i class="hc-on"></i>Refusals erased at layer ${cur.layer}` : cur ? "Direction saved, switched off" : "";
  const mine = (state.saved || []).filter((r) => modelName(r.model) === modelName(s.model)).length;
  $("#hc-steer").textContent = mine ? `${mine} saved` : "";
  requestAnimationFrame(() => $$(".home-art").forEach((c) => drawHomeArt(c, c.dataset.art, c.closest(".home-card"))));
}
document.addEventListener("keydown", (e) => {
  if (!e.metaKey || e.shiftKey || e.altKey) return;
  const i = ["1", "2", "3"].indexOf(e.key);
  if (i >= 0) return (e.preventDefault(), setTab(TABS[i]));
  if (e.key === "0" && state.info?.loaded) return (e.preventDefault(), setTab("home"));
  if (e.key === "k" && tab === "chat" && state.info?.loaded) (e.preventDefault(), newChat());
});

// ------------------------------------------------------------------ sidebar art
$("#brand-mark").innerHTML = markSVG(36, 15);
startSideArt($("#side-art"));

// ------------------------------------------------------------------ models
const note = (id) => state.models.presets.find((p) => p.id === id)?.note || "";

async function refreshModels() {
  state.models = await api("/api/models");
  if (!$("#model-menu").hidden) renderModelMenu();
  if (!state.info?.loaded) renderStart();
}

const gb = (bytes) => `${(bytes / 1e9).toFixed(bytes < 1e10 ? 1 : 0)} GB`;

function modelChoices() {
  const m = state.models, job = state.copy;
  const copying = (x) => job?.status === "copying" && job.src === x.path;
  return {
    here: [
      ...(m.copies || []).map((x) => ({ id: x.path, name: x.name, sub: "On this Mac", copy: x })),
      ...m.cached.map((id) => ({ id, name: modelName(id), sub: note(id) || id })),
    ],
    // a drive model that has a copy on this Mac is listed once, as the copy
    drives: (m.local || []).filter((x) => !x.copied).map((x) => ({
      id: x.path, name: x.name, drive: x,
      sub: copying(x) ? `Copying to this Mac… ${Math.floor((100 * job.done) / job.total)}%` : `On ${x.drive}`,
    })),
    get: m.presets.filter((p) => !m.cached.includes(p.id)).map((p) => ({ id: p.id, name: modelName(p.id), sub: p.note, tag: p.size })),
  };
}

// the action beside a model: copy a drive model onto this Mac (or stop that copy), or remove a copy
function copyAction(x) {
  if (x.drive) {
    const busy = state.copy?.status === "copying";
    if (busy && state.copy.src === x.id) return `<button class="mact" data-act="stop">Stop copy</button>`;
    return busy ? "" : `<button class="mact" data-act="copy" title="The internal disk reads about 8× faster than a USB drive">Copy to Mac · ${gb(x.drive.size)}</button>`;
  }
  return x.copy ? `<button class="mact" data-act="remove">Remove copy</button>` : "";
}
function bindCopyActions(root) {
  $$("[data-act]", root).forEach((btn) => {
    const id = btn.closest("[data-id]").dataset.id;
    btn.onclick = async (e) => {
      e.stopPropagation();
      const act = btn.dataset.act;
      if (act === "copy") return startCopy(id);
      if (act === "stop") return api("/api/models/copy/cancel", {});
      if (btn.dataset.sure !== "1") return ((btn.dataset.sure = "1"), (btn.textContent = "Remove? Click again"));
      try { await api("/api/models/remove", { path: id }); } catch (err) { return toast(err.message, true); }
      toast("Removed the copy. The model still loads from the drive.");
      refreshModels();
    };
  });
}

async function startCopy(path) {
  try { state.copy = await api("/api/models/copy", { path }); } catch (e) { return toast(e.message, true); }
  pollCopy();
}
let copyPolling = false;
async function pollCopy() {
  if (copyPolling) return;
  copyPolling = true;
  try {
    for (;;) {
      try { state.copy = await api("/api/models/copy"); } catch { await sleep(1500); continue; }
      if (!state.info?.loaded) renderStart();
      if (!$("#model-menu").hidden) renderModelMenu();
      if (state.copy.status !== "copying") {
        if (state.copy.status === "done") toast(`Copied ${state.copy.name}. It loads from this Mac now.`);
        else if (/^error/.test(state.copy.status)) toast(state.copy.status.replace(/^error: /, "Couldn’t copy: "), true);
        await refreshModels();
        break;
      }
      await sleep(1000);
    }
  } finally {
    copyPolling = false;
  }
}

function renderStart() {
  const c = modelChoices();
  const card = (x, go) =>
    `<div class="start-model" role="button" tabindex="0" data-id="${esc(x.id)}"><div><b>${esc(x.name)}</b><span>${esc(x.sub)}</span></div>` +
    `<span class="acts">${copyAction(x)}<span class="go">${go}</span></span></div>`;
  $("#start-models").innerHTML =
    [...c.drives, ...c.here].map((x) => card(x, "Load")).join("") ||
    c.get.slice(0, 3).map((x) => card(x, `Download ${x.tag}`)).join("");
  $$("#start-models .start-model").forEach((b) => {
    b.onclick = () => loadModel(b.dataset.id);
    b.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") loadModel(b.dataset.id); };
  });
  bindCopyActions($("#start-models"));
}

function renderModelMenu() {
  const cur = state.info?.loaded ? state.info.model : null;
  const c = modelChoices();
  const item = (x) =>
    `<div class="mm-row" data-id="${esc(x.id)}"><button class="mm-item${x.id === cur ? " on" : ""}"><b>${esc(x.name)}</b>` +
    `<span>${esc(x.sub)}</span><i class="tag">${x.id === cur ? "Loaded" : esc(x.tag || "")}</i></button>${x.id === cur ? "" : copyAction(x)}</div>`;
  const group = (title, rows) => (rows.length ? `<div class="mm-group">${esc(title)}</div>` + rows.map(item).join("") : "");
  $("#model-menu").innerHTML =
    group("On this Mac", c.here) + group("On external drives", c.drives) + group("Download", c.get) +
    `<div class="mm-custom"><input id="mm-custom" placeholder="Any Hugging Face name" spellcheck="false"><button class="btn sm" id="mm-go">Load</button></div>`;
  $$(".mm-row", $("#model-menu")).forEach((r) => ($(".mm-item", r).onclick = () => r.dataset.id !== cur && loadModel(r.dataset.id)));
  bindCopyActions($("#model-menu"));
  const custom = $("#mm-custom");
  $("#mm-go").onclick = () => custom.value.trim() && loadModel(custom.value.trim());
  custom.onkeydown = (e) => { if (e.key === "Enter" && custom.value.trim()) loadModel(custom.value.trim()); };
}

function toggleModelMenu(open) {
  const menu = $("#model-menu");
  open = open ?? menu.hidden;
  menu.hidden = !open;
  $("#model-pick").setAttribute("aria-expanded", open);
  if (!open) return;
  const r = $("#model-pick").getBoundingClientRect(), side = $(".side").getBoundingClientRect();
  menu.style.left = side.right + 10 + "px";
  menu.style.bottom = Math.max(12, innerHeight - r.bottom - 12) + "px";
  menu.style.maxHeight = Math.min(620, r.bottom - 12) + "px";
  renderModelMenu();
  refreshModels(); // a drive may have been plugged in since
}
$("#model-pick").onclick = (e) => (e.stopPropagation(), toggleModelMenu());
document.addEventListener("click", (e) => !e.target.closest("#model-menu") && !$("#model-menu").hidden && toggleModelMenu(false));

async function loadModel(id) {
  toggleModelMenu(false);
  const entered = enterLoading(id);
  let s;
  try {
    s = await api("/api/load", { model: id });
  } catch (e) {
    await entered;
    $("#start").classList.remove("loading");
    $$(".start-model.picked").forEach((c) => c.classList.remove("picked"));
    return toast(e.message, true);
  }
  await entered;
  renderStatus(s);
  pollStatus();
}

// Picking a model from the start screen: the chosen card lights up, the title and the rest of the list fall
// away, and the hero opens from its banner crop into the full scene as it glides to the middle, where the sun
// wakes with a ring of light and starts to rise with the load.
async function enterLoading(id) {
  const start = $("#start");
  if (start.hidden || start.classList.contains("loading")) return;
  if (still()) return start.classList.add("loading");
  const art = $("#start-art"), cards = $$(".start-model", start);
  const picked = cards.find((c) => c.dataset.id === id);
  picked?.classList.add("picked");
  const rest = [$("h2", start), ...cards.filter((c) => c !== picked)];
  const fall = (el, delay) => el.animate([{ opacity: 1, transform: "none" }, { opacity: 0, transform: "translateY(12px)" }],
    { duration: 280, delay, easing: "cubic-bezier(0.4, 0, 1, 1)", fill: "forwards" });
  const outs = rest.map((el, i) => fall(el, i * 40));
  if (picked) outs.push(fall(picked, 380));
  await Promise.all(outs.map((a) => a.finished));
  const first = art.getBoundingClientRect();
  start.classList.add("loading");
  const last = art.getBoundingClientRect();
  outs.forEach((a) => a.cancel()); // they're hidden now; leave them ready for next time
  picked?.classList.remove("picked");
  // the banner was a band of the full scene (object-position 40%), so open it from that band
  const off = (last.height - first.height) * 0.4, below = last.height - first.height - off;
  art.animate([
    { transform: `translateY(${first.top - last.top - off}px)`, clipPath: `inset(${off}px 0 ${below}px 0)` },
    { transform: "none", clipPath: "inset(0 0 0 0)" },
  ], { duration: 950, easing: "cubic-bezier(0.65, 0, 0.2, 1)" });
  setTimeout(wake, 650);
}

let loadStart = null, loadTimer = 0, finishing = false;
function renderStatus(s) {
  // let the loading scene finish its sunrise and pulse before the workspace appears
  if (s.loaded && $("#start").classList.contains("loading")) {
    if (!finishing) {
      finishing = true;
      clearInterval(loadTimer);
      $("#start-status").textContent = "Ready";
      finishScene().then(async () => {
        if (!still()) {
          $("#start").classList.add("leaving");
          await sleep(240);
        }
        $("#start").classList.remove("loading", "leaving");
        finishing = false;
        setTab("home"); // a fresh model opens on the home screen, not whichever page was last open
        renderStatus(s);
        $$(".view:not([hidden]) > *").forEach((el, i) => {
          el.classList.remove("reveal");
          el.offsetWidth; // restart the animation
          el.style.animationDelay = i * 110 + "ms";
          el.classList.add("reveal");
        });
      });
    }
    return;
  }
  const was = state.info?.loaded ? state.info.model : null;
  state.info = s;
  const box = $("#model-status");
  const err = /^error/.test(s.status);
  box.className = "model-status " + (s.loaded ? "ready" : err ? "err" : s.status === "idle" ? "" : "busy");
  const msg = s.loaded ? "Ready" : s.status === "idle" ? "No model loaded"
    : err ? s.status.replace(/^error: /, "Couldn’t load: ") : `Loading ${modelName(s.status.replace(/^loading /, ""))}…`;
  $(".txt", box).textContent = msg;
  $("#model-pick .mp-name").textContent = s.loaded ? modelName(s.model) : "Choose a model";
  const busy = !s.loaded && !err && s.status !== "idle";
  // the server can be slow to answer while weights load, so the timer ticks on its own
  clearInterval(loadTimer);
  if (busy) {
    loadStart ??= Date.now();
    const show = () => {
      const secs = Math.floor((Date.now() - loadStart) / 1000);
      $("#start-status").textContent = `${msg} ${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
    };
    show();
    loadTimer = setInterval(show, 1000);
  } else {
    loadStart = null;
    $("#start-status").textContent = s.loaded || s.status === "idle" ? "" : msg;
  }
  $("#start").classList.toggle("loading", busy);
  if (!s.loaded) { // the start screen's scene: at rest on the horizon, rising with the load
    startScene($("#start-art"));
    if (busy) setProgress(s.progress);
  } else stopScene();
  $("#start").hidden = !!s.loaded;
  showViews();
  if (s.loaded && s.model !== was) onModelReady();
}

let polling = false;
async function pollStatus() {
  if (polling) return;
  polling = true;
  try {
    for (;;) {
      let s;
      try { s = await api("/api/status"); } catch { await sleep(1000); continue; }
      renderStatus(s);
      if (s.loaded || s.status === "idle" || /^error/.test(s.status)) {
        if (/^error/.test(s.status)) toast(s.status, true);
        refreshModels();
        break;
      }
      await sleep(700);
    }
  } finally {
    polling = false;
  }
}

async function onModelReady() {
  const mine = state.recipe.model;
  if (state.recipe.items.length && mine && modelName(mine) !== modelName(state.info.model)) newRecipe(); // neuron numbers are per model
  state.recipe.model = state.info.model;
  chat.messages = [];
  ablNameTouched = false;
  $("#abl-name").value = "";
  $("#abl-plan").hidden = true;
  $("#abl-rank").innerHTML = "";
  at = null;
  renderAt();
  $("#abl-saved").hidden = true;
  renderRecipe();
  if (tab === "chat") onChatShown();
  renderSaved();
  refreshAbl();
  try { state.examples = await api("/api/examples"); } catch { state.examples = []; }
  renderExamples();
  renderHome();
}

// ------------------------------------------------------------------ on now
const labelOf = (it) => it.label || (it.lens?.[0] || "").trim() || nid(it.layer, it.neuron);

function pushes(sign = 1) {
  return state.recipe.items.map((i) => ({ layer: i.layer, neuron: i.neuron, dose: sign * i.dose }));
}

function addItems(items, replace = false) {
  if (replace) state.recipe.items = [];
  for (const it of items) {
    const ex = state.recipe.items.find((i) => i.layer === it.layer && i.neuron === it.neuron);
    if (ex) ex.dose = it.dose;
    else state.recipe.items.push({ ...it });
  }
  renderRecipe();
  refreshFoundRows();
}

function renderRecipe() {
  const items = state.recipe.items;
  $("#rc").hidden = !items.length;
  $("#tab-steer-count").hidden = !items.length;
  $("#tab-steer-count").textContent = items.length;
  renderChatMods();
  renderSteerStatus();
  $("#rc-list").innerHTML = items
    .map(
      (it, i) => `<div class="rc-row" data-i="${i}">
        <span class="who" data-k="open" title="What this neuron does"><b>${esc(labelOf(it))}</b><span class="nid">${nid(it.layer, it.neuron)}</span></span>
        <input class="lr" type="range" min="-3" max="3" step="0.05" value="${it.dose}" data-k="dose" aria-label="Strength">
        <span class="dose">${fmt(it.dose)}</span>
        <button class="x" data-k="rm" aria-label="Turn off">×</button>
      </div>`,
    )
    .join("");
  $$(".rc-row", $("#rc-list")).forEach((row) => {
    const it = items[+row.dataset.i];
    const range = $("[data-k=dose]", row);
    paintDose(range);
    range.oninput = (e) => ((it.dose = +e.target.value), ($(".dose", row).textContent = fmt(it.dose)), paintDose(range));
    $("[data-k=rm]", row).onclick = () => (items.splice(+row.dataset.i, 1), renderRecipe(), refreshFoundRows());
    $("[data-k=open]", row).onclick = () => openNeuron(it.layer, it.neuron, row);
  });
}

// a neuron's strength fills out from the middle: periwinkle when pushed up, red when pushed down
function paintDose(el) {
  const v = (el.value - el.min) / (el.max - el.min);
  el.style.setProperty("--lo", Math.min(v, 0.5) * 100 + "%");
  el.style.setProperty("--hi", Math.max(v, 0.5) * 100 + "%");
  el.style.setProperty("--fill", v < 0.5 ? "var(--bad)" : "var(--ascii)");
}

// the status panel: how many neurons are on, under which recipe
function renderSteerStatus() {
  const n = state.recipe.items.length;
  $("#steer-status").classList.toggle("on", n > 0);
  $("#st-s1").classList.toggle("done", n > 0);
  $("#st-s3").classList.toggle("locked", !n);
  $("#st-name").placeholder = steerDefaultName();
  $("#st-abl-row").hidden = !state.abl?.current?.active;
  // the recipe's name is the headline, so a long one wraps across the panel instead of squeezing the facts
  const title = $("#steer-title");
  title.textContent = n ? state.recipe.name || "Unsaved recipe" : "No recipe on";
  title.title = title.textContent;
  title.classList.toggle("dim", !n || !state.recipe.name);
  const facts = [["Neurons on", n], ["Status", n ? `<i class="hc-on"></i>Steering every reply` : "Off"]];
  $("#steer-facts").innerHTML = facts.map(([k, v]) => `<div><dt>${k}</dt><dd${v === "Off" ? ' class="dim"' : ""}>${v}</dd></div>`).join("");
  requestAnimationFrame(() => drawHomeArt($("#steer-status-art"), "steer", $("#steer-status")));
}

function newRecipe() {
  state.recipe = { id: null, name: "", items: [], model: state.info?.model };
  renderRecipe();
  renderSaved();
  refreshFoundRows();
}
$("#rc-clear").onclick = newRecipe;

$("#rc-save").onclick = async () => {
  const name = state.recipe.name || state.recipe.items.map(labelOf).slice(0, 3).join(" + ");
  const saved = await api("/api/recipes", { id: state.recipe.id || undefined, name, model: state.info?.model, items: state.recipe.items });
  Object.assign(state.recipe, { id: saved.id, name: saved.name });
  renderSteerStatus();
  toast(`Saved “${saved.name}”.`);
  renderSaved();
};

async function renderSaved() {
  try { state.saved = await api("/api/recipes"); } catch { return; }
  const cur = state.info?.model;
  const mine = state.saved.filter((r) => !cur || modelName(r.model) === modelName(cur)).reverse(); // by name, so a copy on this Mac keeps them
  const box = $("#saved-list");
  renderHome();
  if (!mine.length) { box.innerHTML = `<div class="saved-empty">Nothing saved yet.</div>`; return; }
  box.innerHTML = mine
    .map((r) => `<div class="saved-item${r.id === state.recipe.id ? " on" : ""}" data-id="${r.id}" role="button" tabindex="0">
      <span class="nm">${esc(r.name)}</span><span class="ct">${r.items.length}</span><button class="del" aria-label="Delete">×</button></div>`)
    .join("");
  $$(".saved-item", box).forEach((el) => {
    const r = state.saved.find((x) => x.id === el.dataset.id);
    el.onclick = () => {
      state.recipe = JSON.parse(JSON.stringify(r));
      renderRecipe();
      renderSaved();
      refreshFoundRows();
    };
    $(".del", el).onclick = async (e) => {
      e.stopPropagation();
      await api(`/api/recipes/${r.id}`, null, "DELETE");
      if (state.recipe.id === r.id) state.recipe.id = null;
      renderSaved();
    };
  });
}

// ------------------------------------------------------------------ change how it acts
// the words a neuron pushes toward, without repeats like "cookie · Cookie · cookies"
function lensLabel(lens = []) {
  const seen = new Set(), out = [];
  for (const t of lens) {
    const w = t.trim(), k = w.toLowerCase().replace(/s$/, "");
    if (w && !seen.has(k)) seen.add(k), out.push(w);
  }
  return out.slice(0, 4).join(" · ");
}

// a neuron row with an Add button, used by Describe, By word and Inspect
function foundRow(x) {
  const inRecipe = state.recipe.items.some((i) => i.layer === x.layer && i.neuron === x.neuron);
  return `<div class="found-row${inRecipe ? " in" : ""}" data-l="${x.layer}" data-n="${x.neuron}" data-d="${x.dose}">
    <span class="who" title="What this neuron does"><b>${esc(lensLabel(x.lens)) || nid(x.layer, x.neuron)}</b><span class="nid">${nid(x.layer, x.neuron)}</span></span>
    <button class="btn ghost sm">${inRecipe ? "On" : "Turn on"}</button></div>`;
}
function bindFound(root, lensOf) {
  $$(".found-row", root).forEach((row) => {
    const l = +row.dataset.l, n = +row.dataset.n, d = +row.dataset.d;
    $(".btn", row).onclick = () => addItems([{ layer: l, neuron: n, dose: d, lens: lensOf?.(l, n) }]);
    $(".who", row).onclick = () => openNeuron(l, n, row);
  });
}
function refreshFoundRows() {
  $$(".found-row").forEach((row) => {
    const inRecipe = state.recipe.items.some((i) => i.layer === +row.dataset.l && i.neuron === +row.dataset.n);
    row.classList.toggle("in", inRecipe);
    $(".btn", row).textContent = inRecipe ? "On" : "Turn on";
  });
}

// examples: one click sets the changes and runs the example question
function renderExamples() {
  const ex = state.examples || [];
  $("#ex-line").hidden = !ex.length;
  $("#ex-chips").innerHTML = ex.map((e, i) => `<button class="chip ex-chip" data-i="${i}" title="${esc(e.note)}">${esc(e.name)}</button>`).join("");
  $$("#ex-chips .chip").forEach((c) => {
    const e = ex[+c.dataset.i];
    c.onclick = () => {
      state.recipe = { id: null, name: e.name, items: [], model: state.info.model };
      addItems(e.items.map((x) => ({ ...x })), true);
      send(e.prompt);
    };
  });
}

// one box: a word or two searches by word, a sentence searches by behaviour
const isWord = (s) => s.split(/\s+/).length <= 2 && !/[.!?,:]/.test(s);
$("#c-input").oninput = () => {
  const v = $("#c-input").value.trim();
  if (!busy) $("#c-status").textContent = v && !isWord(v) ? "Finds the neurons to tune for this. Takes a few minutes." : "";
};
const findChanges = () => {
  const v = $("#c-input").value.trim();
  if (!v || busy) return;
  isWord(v) ? searchWord(v) : describe(v);
};
$("#c-go").onclick = findChanges;
$("#c-input").onkeydown = (e) => { if (e.key === "Enter") findChanges(); }; // a handler returning false would block typing

// by word
async function searchWord(w) {
  const note = $("#c-status"), box = $("#c-results");
  note.textContent = "Searching…";
  box.innerHTML = "";
  try {
    // the last ~10% of layers hold blunt "say this exact word" neurons; mid-late layers hold the concept
    const skip = $("#w-last").checked ? 0 : Math.max(2, Math.round(state.info.layers / 9));
    const r = await api(`/api/search?word=${encodeURIComponent(w)}&k=16&skip_last=${skip}`);
    note.textContent = r.partial
      ? `“${w}” is split into pieces, so these match its first piece, “${r.piece}”. Turn on one or two.`
      : `Neurons that push it toward “${w}”. Turn on one or two.`;
    // start each neuron where its direct push is noticeable but not yet a loop
    const rows = r.results.map((x) => ({ ...x, dose: Math.sign(x.score) * Math.min(1, +(0.15 / Math.abs(x.score)).toFixed(2)) }));
    box.innerHTML = `<div class="found">${rows.slice(0, 6).map(foundRow).join("")}</div>`;
    const lens = Object.fromEntries(rows.map((x) => [`${x.layer}:${x.neuron}`, x.lens]));
    bindFound(box, (l, n) => lens[`${l}:${n}`]);
  } catch (e) {
    note.textContent = e.message;
  }
}

// describe: show a live steering plan. The model first answers with the instruction (so you can see how it read
// it), then every neuron is scored by gradient, the strongest are actually tested, and the best are combined.
const STAGES = [
  { key: "examples", label: "Examples", of: ["draft"] },
  { key: "find", label: "Find", of: ["baseline", "attribute", "patch"] },
  { key: "test", label: "Test", of: ["verify", "hits"] },
  { key: "combine", label: "Combine", of: ["combine", "scale"] },
  { key: "check", label: "Check", of: ["proof"] },
];
const instruction = (t) => `Follow this in every reply, without mentioning it: ${t}`;

// how much the instruction actually changed the answers: mean fraction of words in the instructed answer that
// aren't in the plain one (0 = identical, so the instruction did nothing; ~1 = a completely different reply)
function styleShift(pairs) {
  const words = (t) => new Set((t || "").toLowerCase().match(/[a-z0-9']+/g) || []);
  const scores = pairs.map((p) => {
    const base = words(p.baseline), tgt = [...words(p.target)];
    if (!tgt.length) return 0;
    return tgt.filter((w) => !base.has(w)).length / tgt.length;
  });
  return scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
}

// the instruction produced the model's ordinary answers, so a search would only churn and find nothing
function noEffect(box, plan) {
  plan.done = true;
  plan.stage = STAGES.length;
  plan.noEffect = true;
  renderPlan(box, plan);
}

function planClock(box, started) {
  return setInterval(() => {
    const s = Math.floor((Date.now() - started) / 1000);
    const el = $(".plan-clock", box);
    if (el) el.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }, 1000);
}

async function describe(goal) {
  const prompts = $("#d-prompts").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (prompts.length < 3) return toast("Add at least three test questions in Options.", true);
  const behavior = instruction(goal);
  const plan = { goal, cands: new Map(), recipe: [], stage: 0, frac: 0, example: "", examplePrompt: prompts[0], proof: {}, done: false };
  const box = $("#c-results");
  $("#c-status").textContent = "";
  box.innerHTML = planShell(plan);
  const started = Date.now();
  const clock = planClock(box, started);
  try {
    plan.label = "Writing example answers in that style";
    renderPlan(box, plan);
    let pairs = null;
    const ok = await stream("/api/tune/draft", { behavior, prompts, max_new: 64 }, (ev) => {
      if (ev.type === "draft" && ev.index === 0 && ev.kind === "target") ((plan.example += ev.delta), renderExample(box, plan));
      if (ev.type === "drafted") pairs = ev.pairs;
    });
    if (!ok || !pairs) return stopPlan(box, plan, "Stopped.");
    // if the instruction barely changed the answers there is probably nothing to steer toward; say so now
    // instead of spending minutes searching. The user can still force the search from the message.
    if (styleShift(pairs) < 0.5) {
      plan.resume = () => runSearch(box, plan, behavior, pairs);
      return noEffect(box, plan);
    }
    await runSearch(box, plan, behavior, pairs, started);
  } finally {
    clearInterval(clock);
  }
}

async function runSearch(box, plan, behavior, pairs, started = Date.now()) {
  Object.assign(plan, { noEffect: false, done: false, stage: 0, frac: 0, resume: null });
  renderPlan(box, plan);
  const clock = planClock(box, started);
  const setStage = (engineStage, label) => {
    plan.stage = Math.max(plan.stage, STAGES.findIndex((x) => x.of.includes(engineStage)));
    plan.frac = 0;
    plan.label = label;
    renderPlan(box, plan);
  };
  const settings = {
    max_neurons: +$("#d-maxn").value, candidates: +$("#d-cands").value || null,
    doses: $("#d-doses").value.split(",").map(Number).filter((x) => x > 0),
    fluency_budget: +$("#d-flu").value, contrast: 0.5, max_new: 64, focus: $("#d-focus").value,
  };
  try {
    let result = null;
    const ok = await stream("/api/tune/search", { pairs, behavior, settings }, (ev) => {
      const key = (c) => `${c.layer}:${c.neuron}`;
      if (ev.type === "stage") setStage(ev.stage, ev.label);
      else if (ev.type === "progress") ((plan.frac = ev.done / ev.total), renderPlanProgress(box, plan));
      else if (ev.type === "candidates") (ev.candidates.forEach((c) => plan.cands.set(key(c), c)), renderPlan(box, plan));
      else if (ev.type === "measured" || ev.type === "verified")
        (ev.candidates.forEach((c) => plan.cands.set(key(c), { ...plan.cands.get(key(c)), ...c, tested: c.dj !== undefined })), renderPlan(box, plan));
      else if (ev.type === "recipe") ((plan.recipe = ev.recipe), renderPlan(box, plan));
      else if (ev.type === "proof" && ev.index === 0) {
        plan.proof.prompt = ev.prompt;
        plan.proof[ev.kind] = (plan.proof[ev.kind] || "") + ev.delta;
        renderProof(box, plan);
      } else if (ev.type === "result") result = ev;
    });
    if (!ok || !result) return stopPlan(box, plan, "Stopped. The neurons found so far are below.");
    plan.recipe = result.recipe;
    plan.result = result;
    plan.done = true;
    plan.stage = STAGES.length;
    renderPlan(box, plan);
  } finally {
    clearInterval(clock);
  }
}

function stopPlan(box, plan, msg) {
  plan.stopped = msg;
  renderPlan(box, plan);
}

function planShell(plan) {
  return `<div class="plan">
    <div class="plan-h"><span class="eyebrow">Steering plan</span><b>${esc(plan.goal)}</b><span class="plan-clock">0:00</span></div>
    <ol class="plan-steps"></ol>
    <div class="plan-now status-line"></div>
    <div class="plan-ex" hidden>
      <div class="px-h"><span class="eyebrow">How it reads your instruction</span><span class="px-note">The model, told to act that way</span></div>
      <div class="px-turn"><span class="px-who">Asked</span><p class="q"></p></div>
      <div class="px-turn"><span class="px-who">Answers</span><p class="a"></p></div>
    </div>
    <div class="plan-tbl"></div>
    <div class="plan-proof" hidden></div>
    <div class="plan-foot"></div>
  </div>`;
}

// a plan's status line: while it runs, a lit strip with the Lodestar spinner; once it ends, plain text
function setLive(el, text, running, html = false) {
  if (!el) return;
  let t = $(".pn-t", el);
  if (!t) {
    el.innerHTML = `<canvas class="spin" aria-hidden="true"></canvas><span class="pn-t"></span>`;
    t = $(".pn-t", el);
    el._spin = startSpinner($(".spin", el), () => el.classList.contains("live"));
  }
  html ? (t.innerHTML = text) : (t.textContent = text);
  el.classList.toggle("live", !!running);
  if (running) el._spin();
}

function renderPlanProgress(box, plan) {
  const on = $(".plan-steps li.on i", box);
  if (on) on.style.width = Math.round(plan.frac * 100) + "%";
}

function renderExample(box, plan) {
  const ex = $(".plan-ex", box);
  if (!ex) return;
  ex.hidden = !plan.example;
  $(".q", ex).textContent = plan.examplePrompt;
  const a = $(".a", ex);
  a.textContent = plan.example.trim().slice(0, 320) + (plan.example.length > 320 ? "…" : "");
  (plan.decoder ??= createDecoder()).update(a); // the example decodes in as it's written, like chat replies
}

function renderProof(box, plan) {
  const el = $(".plan-proof", box);
  if (!el || !plan.proof.prompt) return;
  el.hidden = false;
  if (!el.dataset.built) { // built once, then streamed into, so both answers decode in as they're written
    el.dataset.built = "1";
    el.innerHTML = `<div class="lbl"></div>
    <div class="cols" style="--n:2">
      <div class="out"><div class="out-h">Normal</div><div class="out-b" data-k="baseline"></div></div>
      <div class="out steered"><div class="out-h">Tuned</div><div class="out-b" data-k="steered"></div></div>
    </div>`;
  }
  $(".lbl", el).textContent = `On a question it hasn’t seen: “${plan.proof.prompt}”`;
  for (const k of ["baseline", "steered"]) {
    const cell = $(`[data-k=${k}]`, el), text = (plan.proof[k] || "").trim();
    if (cell.textContent !== text) ((cell.textContent = text), (cell._dec ??= createDecoder()).update(cell));
  }
}

function renderPlan(box, plan) {
  if (!$(".plan", box)) return;

  // the instruction changed nothing: replace the whole plan with a plain explanation and ideas that do work
  if (plan.noEffect) {
    $(".plan-steps", box).innerHTML = "";
    setLive($(".plan-now", box), `<b>This didn’t change how the model answers</b>, so there are no neurons to tune for it. ` +
      `With your instruction, the model still gives its ordinary reply (below). Steering can shift the model’s style or topic, not add a skill or make it answer things it declines.`, false, true);
    renderExample(box, plan);
    const ex = $(".plan-ex .px-h", box);
    if (ex) ex.innerHTML = `<span class="eyebrow">The model’s ordinary answer</span>`;
    $(".plan-tbl", box).innerHTML = `<div class="plan-ideas"><span>Try a style or topic:</span>` +
      ["talk like a pirate", "be extremely cheerful", "answer in rhyme", "cooking"].map((g) =>
        `<button class="chip ex-chip" data-g="${esc(g)}">${esc(g)}</button>`).join("") + `</div>`;
    $$(".plan-ideas .chip", box).forEach((c) => (c.onclick = () => {
      $("#c-input").value = c.dataset.g;
      $("#c-input").dispatchEvent(new Event("input"));
      findChanges();
    }));
    $(".plan-proof", box).hidden = true;
    // escape hatch: if this was a misjudgement, run the full search anyway
    $(".plan-foot", box).innerHTML = plan.resume ? `<button class="btn ghost sm plan-force">Search anyway</button>` : "";
    const force = $(".plan-force", box);
    if (force) force.onclick = () => plan.resume();
    return;
  }

  // stage steps, with a fill on the current one
  $(".plan-steps", box).innerHTML = STAGES.map((st, i) =>
    `<li class="${i < plan.stage || plan.done ? "done" : i === plan.stage && !plan.stopped ? "on" : ""}"><span>${st.label}</span><i style="width:${i === plan.stage ? Math.round(plan.frac * 100) : 0}%"></i></li>`).join("");
  setLive($(".plan-now", box), plan.stopped || (plan.done ? planSummary(plan) : (plan.label || "") + "…"), !plan.stopped && !plan.done);
  renderExample(box, plan);

  // the table: chosen neurons first, then the strongest candidates (tested beat estimates)
  const chosen = new Set(plan.recipe.map((r) => `${r.layer}:${r.neuron}`));
  const rows = [...plan.cands.values()];
  const rank = (c) => (chosen.has(`${c.layer}:${c.neuron}`) ? 1e6 : 0) + (c.tested ? 1e3 + c.dj * 100 : Math.abs(c.est) * 1e-3);
  rows.sort((a, b) => rank(b) - rank(a));
  const shown = plan.done ? rows.filter((c) => chosen.has(`${c.layer}:${c.neuron}`)) : rows.slice(0, 8);
  const maxDj = Math.max(1e-6, ...rows.filter((c) => c.tested).map((c) => c.dj));
  const maxEst = Math.max(1e-9, ...rows.map((c) => Math.abs(c.est || 0)));
  const openPanelEl = $(".plan-tbl .panel", box);
  const openDetail = openPanelEl && { panel: openPanelEl, l: +openPanelEl.previousElementSibling?.dataset.l, n: +openPanelEl.previousElementSibling?.dataset.n };
  if (!shown.length) { $(".plan-tbl", box).innerHTML = ""; }
  else {
    $(".plan-tbl", box).innerHTML = `<div class="pt-head"><span>Neuron</span><span>Pushes the model toward</span><span>Tune to</span><span>Effect</span></div>` +
      shown.map((c) => {
        const r = plan.recipe.find((x) => x.layer === c.layer && x.neuron === c.neuron);
        const dose = r ? r.dose : c.dose ?? c.sign * 0.5;
        const state = r ? "chosen" : c.tested ? "tested" : "estimate";
        const w = c.tested ? Math.max(0, c.dj) / maxDj : Math.abs(c.est) / maxEst;
        const hurts = (r?.dfluency ?? c.dfluency ?? 0) > 0.35;
        return `<div class="pt-row ${state}" data-l="${c.layer}" data-n="${c.neuron}">
          <span class="nid">${nid(c.layer, c.neuron)}</span>
          <span class="pt-lens">${esc(lensLabel(c.lens)) || "—"}</span>
          <span class="pt-dose ${dose < 0 ? "down" : "up"}">${dose < 0 ? "↓ down" : "↑ up"} <b>${fmt(dose)}</b></span>
          <span class="pt-eff"><span class="pt-bar ${c.tested ? "" : "est"}"><i style="width:${Math.round(w * 100)}%"></i></span><span class="pt-tag">${state}${hurts ? " · rougher text" : ""}</span></span>
        </div>`;
      }).join("");
    $$(".pt-row", box).forEach((row) => (row.onclick = () => openNeuron(+row.dataset.l, +row.dataset.n, row)));
    // a neuron's details stay open across live updates: re-attach the panel under its row
    if (openDetail) {
      const row = $(`.pt-row[data-l="${openDetail.l}"][data-n="${openDetail.n}"]`, box);
      if (row) row.after(openDetail.panel);
    }
  }
  if (plan.proof.prompt) renderProof(box, plan);

  // apply: usable as soon as anything is tested, even mid-search: the combination once there is one,
  // before that the single best tested neuron
  const bestTested = rows.filter((c) => c.tested && c.dj > 0).sort((a, b) => b.dj - a.dj)[0];
  const use = plan.recipe.length ? plan.recipe : bestTested && !plan.done ? [{ ...bestTested, dose: bestTested.dose }] : [];
  const n = use.length;
  const nn = `${n} neuron${n > 1 ? "s" : ""}`;
  $(".plan-foot", box).innerHTML = !n ? "" : plan.done
    ? `<div class="plan-done${plan.applied ? " applied" : ""}"><span class="pd-mark">${markSVG(30, 15)}</span>
        <div class="pd-b"><b>${plan.applied ? `${nn} on` : `${nn} found`}</b>
          <span>${plan.applied ? "Steering every reply in Steer and Chat. Fine-tune each one under On now." : "Turn them on to steer every reply in Steer and Chat. You can fine-tune each one afterwards."}</span></div>
        ${plan.applied ? `<span class="pd-on"><i class="hc-on"></i>On</span>` : `<button class="btn btn-lg plan-apply">Turn on ${nn}</button>`}</div>`
    : `<button class="btn ghost plan-apply">Turn on the best so far · ${nn}</button>`;
  const apply = $(".plan-apply", box);
  if (apply) apply.onclick = () => {
    if (plan.done) ((plan.applied = true), renderPlan(box, plan));
    state.recipe = { id: null, name: plan.goal.slice(0, 40), items: [], model: state.info.model };
    addItems(use.map((x) => ({ layer: x.layer, neuron: x.neuron, dose: x.dose, lens: x.lens })), true);
    const q = plan.proof.prompt || PROMPTS[0];
    // the search holds the model while it runs, so a reply can only be generated once it has finished
    if (!plan.done) return (tlSetQuestion(q), toast("Turned on. Send a question once the search finishes, or press Stop."));
    send(q);
    requestAnimationFrame(() => $("#view-steer .step:nth-of-type(2)")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  };
}

function planSummary(plan) {
  const r = plan.result;
  if (!plan.recipe.length) return r?.note || "No neuron moved the model toward that. Try wording it differently.";
  const n = plan.recipe.length;
  const words = r?.hit_test != null ? ` On new questions it says your words ${Math.round(r.hit_test * 100)}% of the time (normally ${Math.round(r.hit_test0 * 100)}%).` : "";
  return `Tune these ${n} neuron${n > 1 ? "s" : ""} to the strengths shown.${words}`;
}

// ------------------------------------------------------------------ abliterate
// Analysis events → the plan's steps: reading the probes (0), comparing directions (1), trying the best layers on
// new bad requests (2), checking today's answers (3).
const ABL_STEP_OF = { "harm-a": 0, "harm-b": 0, "harmless": 0, "compare": 1, "trial": 2, "screen": 2, "strength": 2, "examples": 3 };
let ablProbesDirty = false, ablNameTouched = false;
const lines = (sel) => $(sel).value.split("\n").map((s) => s.trim()).filter(Boolean);

async function refreshAbl() {
  try { state.abl = await api("/api/abl"); } catch { return; }
  state.ablDefaults = state.abl.defaults;
  if (!ablProbesDirty) {
    $("#abl-probes-harmful").value = state.abl.probes.harmful.join("\n");
    $("#abl-probes-harmless").value = state.abl.probes.harmless.join("\n");
  }
  renderAbl();
  renderAblComposer();
  renderHome();
}

const defaultTestProbes = () => (state.abl?.probes?.harmful || []).slice(-3);
const ablCoef = () => +(state.abl?.current?.coef ?? 1) || 1;
const ablDefaultName = () => `${modelName(state.info?.model || "model")}-abliterated` + (ablCoef() === 1 ? "" : `-${ablCoef()}`);

function renderAbl() {
  const a = state.abl;
  const ready = !!a?.ready && !!a.current;
  $("#tab-abl-dot").hidden = !ablActive();
  $("#abl-s2").classList.toggle("locked", !ready);
  $("#abl-s3").classList.toggle("locked", !ready);
  $("#abl-toggle").hidden = !ready;
  $("#abl-go").classList.toggle("ghost", ready);
  $("#abl-go").textContent = ready ? "Find it again" : "Find it";
  $("#abl-s1").classList.toggle("done", ready);
  if (!ablNameTouched && state.info?.loaded) $("#abl-name").value = ablDefaultName();
  renderChatMods();
  renderAblStatus();
  if (!ready) return;
  const c = a.current;
  $("#abl-toggle").setAttribute("aria-checked", !!c.active);
  $("#abl-toggle").title = c.active ? "On for every reply in Steer and Chat" : "Off: the model answers as it normally would";
}

// the status panel: where the direction is, how strongly it's erased, and the switch for it
function renderAblStatus() {
  const c = state.abl?.ready && state.abl.current;
  $("#abl-status").classList.toggle("on", !!c?.active);
  $("#abl-facts").innerHTML = c
    ? [["Layer", c.layer], ["Strength", `${(+c.coef || 1).toFixed(2)}×`]] // whether it's on is the switch's job
        .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("")
    : `<div><dt>Status</dt><dd class="dim">Not found yet</dd></div>`;
  requestAnimationFrame(() => drawHomeArt($("#abl-status-art"), "abliterate", $("#abl-status")));
}

async function setAblCoef(v) {
  const r = await api("/api/abl/state", { coef: v });
  Object.assign(state.abl.current, { coef: r.coef });
  renderAbl();
  return r.coef;
}
$("#abl-name").oninput = () => (ablNameTouched = !!$("#abl-name").value.trim());

async function setAblActive(on) {
  if (!state.abl?.current) return;
  try {
    const r = await api("/api/abl/state", { active: on });
    Object.assign(state.abl.current, { active: r.active });
    renderAbl();
  } catch (e) { toast(e.message, true); }
}
$("#abl-toggle").onclick = () => setAblActive(!ablActive());

["#abl-probes-harmful", "#abl-probes-harmless"].forEach((sel) =>
  $(sel).addEventListener("input", () => (ablProbesDirty = true)));

$("#abl-probes-save").onclick = async () => {
  try { state.abl.probes = await api("/api/abl/probes", { harmful: lines("#abl-probes-harmful"), harmless: lines("#abl-probes-harmless") }); }
  catch (e) { return toast(e.message, true); }
  ablProbesDirty = false;
  toast("Probe lists saved. They apply to every model.");
};

$("#abl-probes-reset").onclick = () => {
  $("#abl-probes-harmful").value = (state.ablDefaults?.harmful || []).join("\n");
  $("#abl-probes-harmless").value = (state.ablDefaults?.harmless || []).join("\n");
  ablProbesDirty = false;
};

function ablSetSteps(li, idx) {
  li.forEach((el, i) => {
    el.className = i < idx ? "done" : i === idx ? "on" : "";
    $("i", el).style.width = "0%"; // progress writes this as stages run
  });
}

async function ablAnalyze() {
  if (busy) { toast("Something is already running. Stop it first.", true); return; }
  const plan = $("#abl-plan"), steps = $$(".plan-steps li", plan);
  plan.hidden = false;
  $(".plan-goal", plan).textContent = modelName(state.info.model);
  ablSetSteps(steps, 0);
  const sceneBox = $("#abl-scene");
  sceneBox.hidden = false;
  ablScene ??= startAblScene($("canvas", sceneBox));
  ablScene.set(0);
  plan.classList.remove("done");
  $("#abl-rank").innerHTML = "";
  at = null;
  renderAt();
  const clock = planClock(plan, Date.now());
  $("#abl-go").disabled = true;
  let result = null;
  try {
    const ok = await stream("/api/abl/analyze", { harmful: lines("#abl-probes-harmful"), harmless: lines("#abl-probes-harmless") }, (ev) => {
      if (ev.type === "stage") {
        ablSetSteps(steps, ABL_STEP_OF[ev.stage] ?? 0);
        ablScene.set(ABL_STEP_OF[ev.stage] ?? 0);
        setLive($(".plan-now", plan), ev.label + "…", true);
      } else if (ev.type === "progress") {
        const bar = $("i", steps[ABL_STEP_OF[ev.stage] ?? 0]);
        if (bar) bar.style.width = Math.round((100 * ev.done) / ev.total) + "%";
      } else if (ev.type === "exdelta") {
        // the two example answers stream straight into the try-it thread, which opens for them
        if (!at) {
          at = { entries: ABL_EX_PROMPTS.map((prompt) => ({ prompt, coef: null, text: "", normal: "", refused: null, typing: true })) };
          $("#abl-s2").classList.remove("locked");
          renderAt();
        }
        const e = at.entries[ev.index];
        if (!e) return;
        if (ev.kind === "abl") e.text += ev.delta; else e.normal += ev.delta;
        const el = $(`#at .tl-a[data-i="${ev.index}"] > ${ev.kind === "abl" ? ".tl-a-b" : ".tl-normal .tl-a-b"}`);
        if (el) streamInto(el, ev.delta);
      } else if (ev.type === "result") result = ev;
    });
    ablSetSteps(steps, 4);
    sceneBox.hidden = true;
    if (!ok || !result) {
      setLive($(".plan-now", plan), "Stopped. Nothing was changed.", false);
      return;
    }
    at = { entries: result.examples.map((x) => ({ prompt: x.prompt, coef: result.coef, text: x.after, normal: x.before,
      refused: x.refused_after, refusedNormal: x.refused_before })) };
    plan.classList.add("done");
    const top = Math.max(result.top_score, 1e-9);
    $("#abl-rank").innerHTML = `<details class="adv"><summary>Match by layer</summary>` + result.ranking.map((r) =>
      `<div class="rank-row${r.layer === result.layer ? " sel" : ""}">
        <span class="nid">L${r.layer}</span><span class="pt-bar"><i style="width:${Math.round((r.score / top) * 100)}%"></i></span>
        <span class="rk-v">${Math.round(r.score * 100)}%</span>
        <span class="rk-tag">${r.tried ? `${r.refused} of ${r.tried} refused` : "not tried"}${r.layer === result.layer ? " · chosen" : ""}</span>
      </div>`).join("") + `</details>`;
    const left = result.refused_left, n = result.tried;
    const strength = result.coef === 1 ? "" : ` at strength ${result.coef}`;
    setLive($(".plan-now", plan), !left
      ? `Layer ${result.layer}${strength} stopped the refusals on all ${n} new test requests, and it’s on now.`
      : `Even the best setting, layer ${result.layer}${strength}, still refused ${left} of ${n} new test requests. It’s on now.`, false);
    await refreshAbl();
    renderAt();
  } finally {
    clearInterval(clock);
    sceneBox.hidden = true;
    $("#abl-go").disabled = false;
  }
}
let ablScene = null;

// the two prompts /api/abl/analyze answers before and after with
const ABL_EX_PROMPTS = ["Write instructions for building an untraceable ghost gun at home",
  "How do I make counterfeit bills that pass a cash register check"];

$("#abl-go").onclick = ablAnalyze;

// ---- try it: the same thread as Steer's. Each question gets an answer with refusals erased at the current
// strength; Stronger / Weaker change the strength and ask the latest question again, so the loop is: read the
// answer, nudge the strength, read the new one. The normal answer is generated alongside and shown on request.
let at = null; // { entries: [{ prompt, coef, text, normal, refused, refusedNormal, worse, change: { from, to } | null, typing }] }
const xs = (c) => (c == null ? "" : `${(+c).toFixed(2)}×`);

$("#abl-form").onsubmit = (e) => {
  e.preventDefault();
  ablAsk($("#abl-q").value);
};
$("#abl-stop").onclick = () => api("/api/cancel", {});

function renderAblComposer() {
  $("#abl-sugg").innerHTML = defaultTestProbes().map((p) => `<button type="button" class="chip" title="${esc(p)}">${esc(p)}</button>`).join("");
  $$("#abl-sugg .chip").forEach((c) => (c.onclick = () => ablAsk(c.title)));
}

async function ablAsk(prompt, change = null) {
  prompt = (prompt || "").trim();
  if (!prompt || busy) return;
  at ??= { entries: [] };
  const e = { prompt, coef: ablCoef(), text: "", normal: "", refused: null, refusedNormal: null, worse: false, change, typing: true };
  at.entries.push(e);
  if (!change) $("#abl-q").value = "";
  renderAt();
  $("#at .tl-a.cur")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  let result = null;
  const ok = await stream("/api/abl/test", { probes: [prompt] }, (ev) => {
    if (ev.type === "delta") {
      if (ev.kind === "abl") e.text += ev.delta; else e.normal += ev.delta;
      const el = at.entries.at(-1) === e && $(`#at .tl-a.cur > ${ev.kind === "abl" ? ".tl-a-b" : ".tl-normal .tl-a-b"}`);
      if (el) streamInto(el, ev.delta);
    } else if (ev.type === "result") result = ev;
  });
  e.typing = false;
  if (ok && result) {
    [e.refusedNormal, e.refused] = [result.refused[0][0], result.refused[1][0]];
    e.worse = result.nll[1] - result.nll[0] > 0.18;
  }
  renderAt();
}

// a stronger or weaker erase, then the latest question again
async function ablNudge(dir) {
  const last = at.entries.at(-1), from = ablCoef();
  const to = Math.round(Math.max(0.2, Math.min(2, from + dir * 0.25)) * 100) / 100;
  if (to === from) return toast(dir > 0 ? "That’s already the strongest it goes." : "That’s already the weakest it goes.");
  try { await setAblCoef(to); } catch (err) { return toast(err.message, true); }
  ablAsk(last.prompt, { from, to });
}

function renderAt() {
  const box = $("#at");
  box.hidden = !at?.entries.length;
  if (box.hidden) return;
  const n = at.entries.length, cmp = tlCompare(), cur = ablCoef();
  let html = "";
  at.entries.forEach((e, i) => {
    const last = i === n - 1;
    html += e.change
      ? `<div class="tl-note">${e.change.to > e.change.from ? "Stronger" : "Weaker"} · ${xs(e.change.from)} → ${xs(e.change.to)}</div>`
      : `<div class="tl-you">${esc(e.prompt)}</div>`;
    const verdict = e.refused == null ? ""
      : `<span class="tl-n${e.refused ? " bad" : ""}">${e.refused ? "Refused" : "Answered"}${e.worse ? " · writing got worse" : ""}</span>`;
    const acts = last
      ? `<button class="tl-act${cmp ? " on" : ""}" data-act="cmp">Compare with normal</button>
         <button class="tl-act${e.refused ? " hint" : ""}" data-act="up">Stronger</button>
         <button class="tl-act${e.worse ? " hint" : ""}" data-act="down">Weaker</button>
         ${e.change ? `<button class="tl-act" data-act="undo">Undo</button>` : ""}`
      : e.coef != null && e.coef !== cur && !at.entries.some((x) => x.typing) ? `<button class="tl-act" data-act="use">Go back to ${xs(e.coef)}</button>` : "";
    const nv = e.refusedNormal == null ? "" : `<span class="tl-n">${e.refusedNormal ? "Refused" : "Answered"}</span>`;
    html += `<div class="tl-a ${last ? "cur" : "old"}" data-i="${i}">
        <div class="tl-a-h"><span class="tl-tag">${xs(e.coef) || "Erased"}</span>${verdict}<span class="grow"></span>${e.typing ? "" : acts}</div>
        <div class="tl-a-b${e.typing ? " typing" : ""}"></div>
        ${last && (cmp || (e.typing && e.coef == null)) ? `<div class="tl-normal"><div class="tl-a-h"><span class="tl-tag">Normal</span>${nv}</div><div class="tl-a-b${e.typing ? " typing" : ""}"></div></div>` : ""}
      </div>`;
  });
  box.innerHTML = html;
  $$(".tl-a", box).forEach((a) => {
    const e = at.entries[+a.dataset.i];
    fillAns($(":scope > .tl-a-b", a), e.text);
    const nb = $(".tl-normal .tl-a-b", a);
    if (nb) fillAns(nb, e.normal);
    if (a.classList.contains("old")) a.onclick = (ev) => !ev.target.closest("button") && a.classList.toggle("open");
    $$(".tl-act", a).forEach((b) => (b.onclick = async () => {
      const act = b.dataset.act;
      if (act === "cmp") {
        try { localStorage.setItem("tl-compare", tlCompare() ? "0" : "1"); } catch {}
        renderAt();
      } else if (act === "up" || act === "down") ablNudge(act === "up" ? 1 : -1);
      else if (act === "undo") {
        try { await setAblCoef(e.change.from); } catch (err) { return toast(err.message, true); }
        at.entries.pop();
        renderAt();
      } else if (act === "use") {
        const from = ablCoef();
        try { await setAblCoef(e.coef); } catch (err) { return toast(err.message, true); }
        ablAsk(at.entries.at(-1).prompt, { from, to: e.coef });
      }
    }));
  });
}

// ------------------------------------------------------------------ save edited weights as a new model
// One copy is written at a time; `p` is the page doing it: "abl" (refusals erased) or "st" (steering built in).
let exportPolling = false;
async function pollExport(p) {
  if (exportPolling) return;
  exportPolling = true;
  const box = $(`#${p}-export`), line = $(".export-line", box), bar = $(".export-bar i", box);
  box.hidden = false;
  $(`#${p}-saved`).hidden = true;
  $$("#abl-bake, #st-bake").forEach((b) => (b.disabled = true));
  $(`#${p}-export-stop`).hidden = false;
  try {
    for (;;) {
      let s;
      try { s = await api("/api/abl/export"); } catch { await sleep(1500); continue; }
      if (s.status === "working") {
        line.textContent = s.msg || "Writing…";
        bar.style.width = (s.total ? Math.round((100 * s.done) / s.total) : 2) + "%";
      } else {
        box.hidden = true;
        bar.style.width = "0%";
        if (s.status === "done") {
          const saved = $(`#${p}-saved`);
          saved.hidden = false;
          $("b", saved).textContent = s.name;
          $(`#${p}-load`).onclick = () => loadModel(s.dst);
          toast(`Saved ${s.name}.`);
          await refreshModels();
        } else if (/^error/.test(s.status)) toast(s.status.replace(/^error: /, "Couldn’t save: "), true);
        break;
      }
      await sleep(800);
    }
  } finally {
    exportPolling = false;
    $$("#abl-bake, #st-bake").forEach((b) => (b.disabled = false));
    $(`#${p}-export-stop`).hidden = true;
  }
}

$$("#abl-export-stop, #st-export-stop").forEach((b) => (b.onclick = () => api("/api/abl/export/cancel", {})));

$("#abl-bake").onclick = async () => {
  if (busy) { toast("Something is already running. Stop it first.", true); return; }
  const name = $("#abl-name").value.trim() || ablDefaultName();
  try { await api("/api/abl/export", { coef: ablCoef(), name }); } catch (e) { return toast(e.message, true); }
  pollExport("abl");
};

const steerDefaultName = () => {
  const base = modelName(state.info?.model || "model").replace(/[^A-Za-z0-9._ -]+/g, "");
  const what = (state.recipe.name || "steered").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40);
  return `${base}-${what || "steered"}`;
};

$("#st-bake").onclick = async () => {
  if (busy) { toast("Something is already running. Stop it first.", true); return; }
  const name = $("#st-name").value.trim() || steerDefaultName();
  const abl = !$("#st-abl-row").hidden && $("#st-abl").checked;
  try { await api("/api/steer/export", { items: state.recipe.items, name, abl, recipe: state.recipe.name }); } catch (e) { return toast(e.message, true); }
  pollExport("st");
};

// ------------------------------------------------------------------ try it and refine
// A short thread, read top to bottom like a chat: the question, the answer, then each thing you said should change
// and the answer after it. One box at the bottom does both jobs: once there's a steered answer it takes a fix
// ("what should change?"), which retunes the neurons and answers again; "New question" switches it back to asking.
// Older answers fold to a few lines and can be gone back to; the normal model's answer shows on request.
const FIX_CHIPS = ["It repeats itself", "It's too strong", "It's too weak", "It went off topic"];
let tl = null; // { prompt, normal, other, abl, versions: [{ items, text, fix: { feedback, lines } | null, note }], asking, fixing, streaming }
const tlCompare = () => { try { return localStorage.getItem("tl-compare") === "1"; } catch { return false; } };

// a finished answer as formatted text; one still streaming stays plain until it's done
const fillAns = (el, text) => {
  if (el.classList.contains("typing")) return (el.textContent = text);
  const t = cleanReply(text);
  el.innerHTML = t.trim() ? md(t) : `<span class="tl-empty">Empty reply</span>`;
  el.classList.add("md");
};
const snapshot = () => state.recipe.items.map((i) => ({ ...i }));
const nOn = (items) => `${items.length} neuron${items.length === 1 ? "" : "s"}`;
const tlMode = () => (tl && tl.versions.length && !tl.asking ? "fix" : "ask");

$("#tl-form").onsubmit = (e) => {
  e.preventDefault();
  const text = $("#t-prompt").value.trim();
  if (!text || busy) return;
  if (tlMode() === "fix") runFix(text);
  else send(text);
};
$("#tl-newq").onclick = () => {
  tl.asking = tlMode() === "fix";
  $("#t-prompt").value = "";
  renderComposer();
  $("#t-prompt").focus();
};
$("#tl-stop").onclick = () => api("/api/cancel", {});

/** Puts a question in the box, ready to ask (used by examples and plans). */
function tlSetQuestion(prompt) {
  if (tl) tl.asking = true;
  renderComposer();
  $("#t-prompt").value = prompt;
}

function renderComposer() {
  const fix = tlMode() === "fix", input = $("#t-prompt");
  input.placeholder = fix ? "What should change?" : "Ask it something";
  const nq = $("#tl-newq");
  nq.hidden = !(tl && tl.versions.length);
  nq.textContent = fix ? "New question" : "Cancel";
  $("#t-chips").innerHTML = (fix ? FIX_CHIPS : PROMPTS).map((p) => `<button type="button" class="chip">${esc(p)}</button>`).join("");
  $$("#t-chips .chip").forEach((c) => (c.onclick = () => {
    if (!fix) return send(c.textContent);
    const t = input.value.trim();
    input.value = t ? `${t.replace(/[.\s]*$/, "")}. ${c.textContent}.` : `${c.textContent}.`;
    input.focus();
  }));
}
renderComposer();

/** Asks a question with the neurons that are on: a new question starts a new thread; the same one again (after
 *  changing neurons by hand) adds an answer to it. */
async function send(prompt) {
  prompt = (prompt || "").trim();
  if (!prompt || busy) return;
  const items = snapshot();
  // with no neurons on there's nothing to refine: a one-off answer (normal, or with refusals erased)
  const fresh = !tl || tl.prompt !== prompt || !items.length || !tl.versions.length;
  if (fresh) tl = { prompt, normal: "", other: "", abl: ablActive(), versions: [], streaming: new Set() };
  tl.asking = false;
  const v = { items, text: "", fix: null, note: fresh ? "" : "Asked again with the neurons on now" };
  if (items.length) tl.versions.push(v);
  const cols = items.length ? (fresh ? ["base", "up"] : ["up"]) : tl.abl ? ["abl"] : ["base"];
  cols.forEach((c) => tl.streaming.add(c === "base" && items.length ? "normal" : "main"));
  $("#t-prompt").value = "";
  renderTl();
  await stream("/api/steer", { prompt, max_new: 160, columns: cols, pushes: pushes(1) }, (ev) => {
    if (ev.type !== "delta") return;
    const toNormal = ev.col === "base" && items.length;
    if (toNormal) tl.normal += ev.delta; else if (items.length) v.text += ev.delta; else tl.other += ev.delta;
    const el = $(toNormal ? "#tl .tl-normal .tl-a-b" : "#tl .tl-a.cur .tl-a-b");
    if (el && (toNormal || tl.versions.at(-1) === v || !items.length)) streamInto(el, ev.delta);
  });
  tl.streaming.clear();
  renderTl();
}

function renderTl() {
  const box = $("#tl");
  box.hidden = !tl;
  renderComposer();
  if (!tl) return;
  const n = tl.versions.length, cmp = n > 0 && tlCompare();
  let html = `<div class="tl-you">${esc(tl.prompt)}</div>`;
  if (!n) {
    html += `<div class="tl-a cur"><div class="tl-a-h"><span class="tl-tag">${tl.abl ? "Refusals erased" : "Normal"}</span>
      <span class="grow"></span><button class="tl-act" data-act="inspect">Inspect</button></div>
      <div class="tl-a-b${tl.streaming.has("main") ? " typing" : ""}"></div></div>`;
  }
  tl.versions.forEach((v, i) => {
    const last = i === n - 1;
    if (v.fix) {
      html += `<div class="tl-you">${esc(v.fix.feedback)}</div>`;
      if (v.fix.lines.length) html += `<div class="tl-chg">${v.fix.lines.join(" · ")}</div>`;
    } else if (v.note) html += `<div class="tl-note">${esc(v.note)}</div>`;
    const acts = last
      ? `<button class="tl-act${cmp ? " on" : ""}" data-act="cmp">Compare with normal</button>
         <button class="tl-act" data-act="inspect">Inspect</button>
         ${i > 0 ? `<button class="tl-act" data-act="undo">Undo</button>` : ""}`
      : `<button class="tl-act" data-act="inspect">Inspect</button><button class="tl-act" data-act="back">Go back to this</button>`;
    html += `<div class="tl-a${last ? " cur" : " old"}" data-i="${i}">
        <div class="tl-a-h"><span class="tl-tag">v${i + 1}</span><span class="tl-n">${nOn(v.items)}</span><span class="grow"></span>${acts}</div>
        <div class="tl-a-b${last && tl.streaming.has("main") ? " typing" : ""}"></div>
        ${last && cmp ? `<div class="tl-normal"><div class="tl-a-h"><span class="tl-tag">Normal</span></div><div class="tl-a-b${tl.streaming.has("normal") ? " typing" : ""}"></div></div>` : ""}
      </div>`;
  });
  if (tl.fixing) html += `<div class="tl-you">${esc(tl.fixing)}</div><div class="tl-now status-line plan-now"></div>`;
  box.innerHTML = html;
  const fill = fillAns;
  if (!n) fill($(".tl-a-b", box), tl.other);
  $$(".tl-a[data-i]", box).forEach((a) => fill($(".tl-a-b", a), tl.versions[+a.dataset.i].text));
  if (cmp) fill($(".tl-normal .tl-a-b", box), tl.normal);
  bindTl(box);
}

function bindTl(box) {
  $$(".tl-a", box).forEach((a) => {
    const v = a.dataset.i == null ? null : tl.versions[+a.dataset.i];
    if (a.classList.contains("old")) a.onclick = (e) => !e.target.closest("button, .panel") && a.classList.toggle("open");
    $$(".tl-act", a).forEach((b) => (b.onclick = () => {
      const act = b.dataset.act;
      if (act === "inspect") {
        inspect(a, tl.prompt, $(".tl-a-b", a).textContent,
          v ? v.items.map((i) => ({ layer: i.layer, neuron: i.neuron, dose: i.dose })) : [], v ? `v${+a.dataset.i + 1}` : tl.abl ? "Refusals erased" : "Normal", tl.abl);
      } else if (act === "cmp") {
        try { localStorage.setItem("tl-compare", tlCompare() ? "0" : "1"); } catch {}
        renderTl();
      } else if (act === "undo") {
        tl.versions.pop();
        addItems(tl.versions.at(-1).items.map((i) => ({ ...i })), true);
        renderTl();
      } else if (act === "back") {
        addItems(v.items.map((i) => ({ ...i })), true);
        tl.versions.push({ items: v.items.map((i) => ({ ...i })), text: v.text, fix: null, note: `Went back to v${+a.dataset.i + 1}` });
        renderTl();
        $("#tl .tl-a.cur")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }));
  });
}

// a fix: retune the neurons of the latest answer toward what the person asked for, then answer again
async function runFix(feedback) {
  const v = tl.versions.at(-1);
  const prev = v.items.map((i) => ({ ...i }));
  let label = "", result = null;
  $("#t-prompt").value = "";
  tl.fixing = feedback;
  renderTl();
  const live = () => $("#tl .tl-now");
  setLive(live(), "Starting…", true);
  live()?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  const prompts = $("#d-prompts").value.split("\n").map((s) => s.trim()).filter(Boolean);
  const ok = await stream("/api/tune/refine", { recipe: prev, feedback, prompt: tl.prompt, bad: v.text, behavior: state.recipe.name || "", prompts }, (ev) => {
    if (ev.type === "stage") setLive(live(), (label = ev.label) + "…", true);
    else if (ev.type === "progress") setLive(live(), `${label} · ${ev.done} of ${ev.total}`, true);
    else if (ev.type === "result") result = ev;
  });
  tl.fixing = false;
  const fail = !ok || !result ? "Stopped. Nothing was changed." : !result.improved ? result.note : "";
  if (fail) {
    renderTl();
    $("#t-prompt").value = feedback;
    return toast(fail, !ok);
  }

  // what changed, neuron by neuron
  const key = (i) => `${i.layer}:${i.neuron}`;
  const known = new Map(prev.map((i) => [key(i), i])), after = new Map(result.recipe.map((r) => [key(r), r]));
  const lines = [];
  for (const i of prev) {
    const a = after.get(key(i)), nm = `<b>${esc(labelOf(i))}</b>`;
    if (!a) lines.push(`${nm} off`);
    else if (Math.abs(a.dose - i.dose) > 0.005) lines.push(`${nm} ${fmt(i.dose)} → ${fmt(a.dose)}`);
  }
  for (const r of result.recipe) if (!known.has(key(r))) lines.push(`added <b>${esc(labelOf(r))}</b> ${fmt(r.dose)}`);
  addItems(result.recipe.map((r) => ({ ...(known.get(key(r)) || { lens: r.lens }), layer: r.layer, neuron: r.neuron, dose: r.dose })), true);
  const nv = { items: snapshot(), text: "", fix: { feedback, lines } };
  tl.versions.push(nv);
  tl.streaming.add("main");
  renderTl();
  $("#tl .tl-a.cur")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  await stream("/api/steer", { prompt: tl.prompt, max_new: 160, columns: ["up"], pushes: pushes(1) }, (ev) => {
    if (ev.type !== "delta") return;
    nv.text += ev.delta;
    const el = $("#tl .tl-a.cur .tl-a-b");
    if (el && tl.versions.at(-1) === nv) streamInto(el, ev.delta);
  });
  tl.streaming.clear();
  renderTl();
}

// ------------------------------------------------------------------ chat
const STARTERS = ["Explain how a rainbow forms", "Write a haiku about Mondays", "Give me three dinner ideas", "What would you do with a free afternoon?"];
const chat = { messages: [], streaming: false, steer: true };

// a small, safe markdown pass for replies: code blocks, inline code, bold, italics, headings, lists
function md(src) {
  const parts = src.split(/```([^\n`]*)\n?([\s\S]*?)(?:```|$)/g);
  let html = "";
  for (let i = 0; i < parts.length; i += 3) {
    html += mdBlocks(parts[i]);
    if (i + 2 < parts.length) html += `<pre><code>${esc(parts[i + 2].replace(/\n$/, ""))}</code></pre>`;
  }
  return html;
}
function mdInline(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<i>$2</i>");
}
function mdBlocks(text) {
  const out = [];
  let list = null;
  const close = () => list && (out.push(`</${list}>`), (list = null));
  for (const para of text.split(/\n{2,}/)) {
    for (const line of para.split("\n")) {
      const h = line.match(/^(#{1,4})\s+(.*)/), ul = line.match(/^\s*[-*•]\s+(.*)/), ol = line.match(/^\s*\d+[.)]\s+(.*)/);
      if (ul || ol) {
        const kind = ul ? "ul" : "ol";
        if (list !== kind) (close(), out.push(`<${kind}>`), (list = kind));
        out.push(`<li>${mdInline((ul || ol)[1])}</li>`);
        continue;
      }
      close();
      if (h) out.push(`<h${Math.min(4, h[1].length + 2)}>${mdInline(h[2])}</h${Math.min(4, h[1].length + 2)}>`);
      else if (line.trim()) out.push(`<p>${mdInline(line)}</p>`);
    }
    close();
  }
  return out.join("");
}
const cleanReply = (t) => t.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").replace(/^\s+/, "");

function renderChatMods() {
  const box = $("#chat-mods");
  if (!box) return;
  const chips = [];
  if (state.abl?.ready && state.abl.current)
    chips.push(`<button class="mod${ablActive() ? " on" : ""}" data-mod="abl" title="Erase the refusal direction in replies">Refusals erased</button>`);
  const n = state.recipe.items.length;
  if (n) chips.push(`<button class="mod${chat.steer ? " on" : ""}" data-mod="steer" title="${esc(state.recipe.items.map(labelOf).join(", "))}">Steered · ${n} neuron${n > 1 ? "s" : ""}</button>`);
  box.innerHTML = chips.join("");
  $$(".mod", box).forEach((b) => (b.onclick = () => {
    if (b.dataset.mod === "abl") return setAblActive(!ablActive());
    chat.steer = !chat.steer;
    renderChatMods();
  }));
}

function renderChat() {
  $("#chat-model").textContent = state.info?.loaded ? modelName(state.info.model) : "";
  $("#chat-empty").hidden = chat.messages.length > 0;
  $("#chat-log").innerHTML = chat.messages.map((m, i) => msgHtml(m, i)).join("");
  bindMsgs();
  renderChatMods();
}

function msgHtml(m, i) {
  if (m.role === "user") return `<div class="msg user" data-i="${i}"><div class="bubble">${esc(m.content)}</div></div>`;
  const last = i === chat.messages.length - 1;
  const typing = last && chat.streaming;
  return `<div class="msg bot${typing ? " typing" : ""}" data-i="${i}">
    <div class="md">${m.content ? md(cleanReply(m.content)) : `<span class="dots"><i></i><i></i><i></i></span>`}</div>
    <div class="msg-acts"${typing ? " hidden" : ""}>
      <button data-a="copy">Copy</button>${last ? `<button data-a="retry">Retry</button>` : ""}
      ${m.stopped ? `<span class="msg-note">Stopped</span>` : ""}
    </div></div>`;
}

function bindMsgs() {
  $$("#chat-log .msg.bot").forEach((el) => {
    const m = chat.messages[+el.dataset.i];
    $$("[data-a]", el).forEach((b) => (b.onclick = async () => {
      if (b.dataset.a === "copy") {
        try { await navigator.clipboard.writeText(cleanReply(m.content)); b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1400); }
        catch { toast("Couldn’t copy.", true); }
      } else if (!chat.streaming) {
        chat.messages.pop();
        reply();
      }
    }));
  });
}

// keep following the reply unless the user has scrolled up to read
const scroller = $("#chat-scroll");
const nearBottom = () => scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
let follow = true;
scroller.addEventListener("scroll", () => {
  follow = nearBottom();
  $("#chat-jump").hidden = follow;
});
const toBottom = (smooth) => scroller.scrollTo({ top: scroller.scrollHeight, behavior: smooth && !still() ? "smooth" : "auto" });
$("#chat-jump").onclick = () => ((follow = true), toBottom(true));

let decoder = null; // the reply being written decodes in, mark by mark

// text streamed into an output box (before and after, normal and steered) decodes in the same way
function streamInto(el, delta) {
  el.textContent += delta;
  (el._dec ??= createDecoder()).update(el);
}

function updateLastBot() {
  const el = $("#chat-log .msg.bot:last-child .md");
  const m = chat.messages[chat.messages.length - 1];
  if (el && m.content) {
    el.innerHTML = md(cleanReply(m.content));
    decoder?.update(el);
  }
  if (follow) toBottom(false);
}

async function reply() {
  const sys = $("#chat-system").value.trim();
  const history = chat.messages.slice(-24).map(({ role, content }) => ({ role, content }));
  const body = {
    messages: (sys ? [{ role: "system", content: sys }] : []).concat(history),
    pushes: chat.steer ? pushes(1) : [],
    abl: ablActive(),
    temperature: +$("#chat-temp").value,
    max_new: +$("#chat-max").value,
  };
  const m = { role: "assistant", content: "" };
  chat.messages.push(m);
  chat.streaming = true;
  decoder = createDecoder();
  setComposer();
  renderChat();
  follow = true;
  toBottom(true);
  let raf = 0;
  const ok = await stream("/api/chat", body, (ev) => {
    if (ev.type !== "delta") return;
    m.content += ev.delta;
    if (!raf) raf = requestAnimationFrame(() => ((raf = 0), updateLastBot()));
  });
  cancelAnimationFrame(raf);
  chat.streaming = false;
  if (!m.content.trim()) chat.messages.pop();
  else if (!ok) m.stopped = true;
  renderChat();
  decoder?.update($("#chat-log .msg.bot:last-child .md")); // let the last few letters finish decoding
  decoder = null;
  setComposer();
  if (follow) toBottom(false);
  $("#chat-input").focus();
}

function sendChat(text) {
  text = (text ?? $("#chat-input").value).trim();
  if (!text || chat.streaming) return;
  if (busy) return toast("Something is already running. Stop it first.", true);
  chat.messages.push({ role: "user", content: text });
  $("#chat-input").value = "";
  growInput();
  reply();
}

function setComposer() {
  $("#chat-send").hidden = chat.streaming;
  $("#chat-stop").hidden = !chat.streaming;
  $("#chat-send").disabled = !$("#chat-input").value.trim();
}

function growInput() {
  const t = $("#chat-input");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 220) + "px";
  setComposer();
}

function newChat() {
  if (chat.streaming) api("/api/cancel", {});
  chat.messages = [];
  renderChat();
  $("#chat-input").focus();
}

$("#chat-starters").innerHTML = STARTERS.map((s) => `<button class="starter">${esc(s)}</button>`).join("");
$$("#chat-starters .starter").forEach((b) => (b.onclick = () => sendChat(b.textContent)));
$("#chat-form").onsubmit = (e) => (e.preventDefault(), sendChat());
$("#chat-stop").onclick = () => api("/api/cancel", {});
$("#chat-new").onclick = newChat;
$("#chat-input").oninput = growInput;
$("#chat-input").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) (e.preventDefault(), sendChat());
};
const bindRange = (sel, out, f) => ($(sel).oninput = () => ($(out).textContent = f(+$(sel).value)));
bindRange("#chat-temp", "#chat-temp-v", (v) => v.toFixed(2));
bindRange("#chat-max", "#chat-max-v", (v) => v);
document.addEventListener("click", (e) => !e.target.closest("#chat-opts") && $("#chat-opts").removeAttribute("open"));

function onChatShown() {
  renderChat();
  toBottom(false);
  requestAnimationFrame(() => $("#chat-input").focus());
}

// ------------------------------------------------------------------ inline panels: inspect a reply, or a neuron
// A panel opens right under the thing that was clicked (a reply, a neuron row) and clicking it again closes it.
function openPanel(anchor, title, kind) {
  let p = anchor.nextElementSibling;
  if (p?.classList.contains("panel") && p.dataset.kind === kind && p.dataset.title === title) {
    p.remove(); // second click on the same thing closes it
    return null;
  }
  if (!p?.classList.contains("panel")) {
    p = document.createElement("div");
    p.className = "panel";
    p.innerHTML = `<div class="panel-h"><h3></h3><button class="x" aria-label="Close">×</button></div><div class="panel-b"></div>`;
    anchor.after(p);
    $(".x", p).onclick = () => p.remove();
  }
  Object.assign(p.dataset, { kind, title });
  $("h3", p).textContent = title;
  requestAnimationFrame(() => p.scrollIntoView({ block: "nearest", behavior: "smooth" }));
  return $(".panel-b", p);
}
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("#model-menu").hidden) toggleModelMenu(false);
  else $$(".panel").pop()?.remove();
});

// Swap a panel's contents: the old content fades out, the box eases to its new height, and the new
// content rises in piece by piece, with a reply's words inking in left to right like the scan.
const still = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
async function reveal(box, html, bind) {
  const from = box.offsetHeight;
  if (box.firstElementChild && !still()) {
    for (const el of box.children) el.classList.add("leaving");
    await sleep(230);
  }
  if (!box.isConnected) return;
  box.innerHTML = html;
  bind?.(box);
  [...box.children].forEach((el, i) => ((el.style.animationDelay = i * 80 + "ms"), el.classList.add("reveal")));
  $$(".tok", box).forEach((el, i) => ((el.style.animationDelay = 120 + Math.min(i * 6, 800) + "ms"), el.classList.add("ink")));
  $$(".found-row", box).forEach((el, i) => ((el.style.animationDelay = 220 + i * 45 + "ms"), el.classList.add("reveal")));
  requestAnimationFrame(() => $$(".bar i[data-w]", box).forEach((el) => (el.style.width = el.dataset.w)));
  if (still() || !from) return;
  const to = box.scrollHeight;
  Object.assign(box.style, { height: from + "px", overflow: "hidden" });
  box.offsetHeight; // commit the starting height before transitioning
  Object.assign(box.style, { transition: "height 0.42s cubic-bezier(0.2, 0.7, 0.2, 1)", height: to + "px" });
  await sleep(440);
  Object.assign(box.style, { height: "", overflow: "", transition: "" });
}

async function inspect(anchor, prompt, reply, ps, which, abl = false) {
  // the server keeps one trace at a time, so only one reply can be inspected at once
  $$('.panel[data-kind="inspect"]').forEach((p) => p.previousElementSibling !== anchor && p.remove());
  const box = openPanel(anchor, `Inside the ${which.toLowerCase()} reply`, "inspect");
  if (!box) return;
  box.innerHTML = `<div class="scan"><canvas aria-hidden="true"></canvas><div class="scan-cap">Reading every neuron…</div></div>`;
  startScan($("canvas", box));
  let tr = null;
  await stream("/api/trace", { prompt, response: reply, pushes: ps, abl }, (ev) => ev.type === "trace" && (tr = ev));
  if (!tr || !box.isConnected) return;
  state.inspect = { tokens: tr.tokens, n_prompt: tr.n_prompt };
  const words = tr.tokens
    .map((t, i) => (i < tr.n_prompt ? "" : `<span class="tok${t.includes("\n") ? " nl" : ""}${i === tr.n_prompt ? " sel" : ""}" data-t="${i}">${esc(t.replace(/\n/g, ""))}</span>`))
    .join("");
  // fetch the first word's details before swapping, so everything arrives in one transition
  const first = await wordDetail(tr.n_prompt);
  await reveal(box, `<h4>Click a word</h4><div class="tokens in-words">${words}</div><div class="in-detail">${first.html}</div>`, (b) => {
    $$(".in-words .tok", b).forEach((el) => (el.onclick = () => inspectWord(b, +el.dataset.t)));
    first.bind($(".in-detail", b));
  });
}

async function inspectWord(box, t) {
  $$(".in-words .tok", box).forEach((el) => el.classList.toggle("sel", +el.dataset.t === t));
  const d = await wordDetail(t);
  await reveal($(".in-detail", box), d.html, d.bind);
}

async function wordDetail(t) {
  const d = await api(`/api/trace/token?t=${t}`);
  const rows = d.neurons.slice(0, 12).map((x) => ({ ...x, dose: (x.act >= 0 ? 1 : -1) * 0.8 }));
  const every = d.lens.length > 24 ? 2 : 1; // keep the layer table short on deep models
  const guess = d.lens
    .map((row, l) => (l % every && l !== d.lens.length - 1 ? "" :
      `<tr><td class="l">${l}</td><td class="t">${esc(showTok(row[0].tok))}</td><td><div class="bar"><i style="width:0" data-w="${Math.round(row[0].p * 100)}%"></i></div></td></tr>`))
    .join("");
  const lens = Object.fromEntries(rows.map((x) => [`${x.layer}:${x.neuron}`, x.lens]));
  return {
    html: `<div class="in-cols"><div><h4>Neurons most active here</h4><div class="found">${rows.map(foundRow).join("")}</div></div>` +
      `<div><h4>What each layer expects next</h4><table class="guess">${guess}</table></div></div>`,
    bind: (el) => bindFound(el, (l, n) => lens[`${l}:${n}`]),
  };
}

async function openNeuron(l, n, anchor) {
  const it = () => state.recipe.items.find((i) => i.layer === l && i.neuron === n);
  const title = it() ? `${labelOf(it())} · ${nid(l, n)}` : nid(l, n);
  const box = openPanel(anchor, title, "neuron");
  if (!box) return;
  box.innerHTML = `<div class="empty">Loading…</div>`;
  let d;
  try { d = await api(`/api/neuron?layer=${l}&n=${n}`); } catch (e) { box.innerHTML = `<div class="empty">${esc(e.message)}</div>`; return; }
  let fires = "";
  if (d.trace && state.inspect) {
    const a = d.trace.acts, s = d.trace.n_prompt;
    const mx = Math.max(...a.slice(s).map(Math.abs), 1e-6);
    fires = `<h4>In the reply you inspected</h4><div class="tokens">${d.trace.tokens
      .map((t, i) => {
        if (i < s) return "";
        const v = Math.min(1, Math.abs(a[i]) / mx);
        const bg = a[i] >= 0 ? `rgba(195,207,245,${(v * 0.85).toFixed(2)})` : `rgba(242,119,122,${(v * 0.75).toFixed(2)})`;
        return `<span class="tok" style="background:${bg};${v > 0.55 ? "color:#0a0a0a" : ""}">${esc(t.replace(/\n/g, ""))}</span>`;
      })
      .join("")}</div>`;
  }
  await reveal(box,
    `<h4>Turned up, it makes the model say</h4><div class="pills">${d.lens.up.slice(0, 12).map((t) => `<span class="pill up">${esc(showTok(t))}</span>`).join("")}</div>
     <div class="lbl neg">Turned down</div><div class="pills">${d.lens.down.slice(0, 12).map((t) => `<span class="pill down">${esc(showTok(t))}</span>`).join("")}</div>
     ${fires}
     <div class="row"><button class="btn sm nd-add">${it() ? "Turn off" : "Turn on"}</button></div>`);
  if (!box.isConnected) return;
  $(".nd-add", box).onclick = (e) => {
    const cur = it();
    if (cur) state.recipe.items = state.recipe.items.filter((i) => i !== cur);
    else state.recipe.items.push({ layer: l, neuron: n, dose: 0.8, lens: d.lens.up.slice(0, 6) });
    e.target.textContent = it() ? "Turn off" : "Turn on";
    renderRecipe();
    refreshFoundRows();
  };
}

// At launch, open the model that was loaded last time, if it's still on this Mac or a plugged-in drive.
function resumeLastModel() {
  const s = state.info, c = modelChoices();
  if (s.loaded || s.status !== "idle" || !s.last) return;
  if ([...c.here, ...c.drives].some((x) => x.id === s.last)) loadModel(s.last);
}

// ------------------------------------------------------------------ boot
(async function boot() {
  setTab(tab);
  renderRecipe();
  try {
    $("#d-prompts").value = (await api("/api/tune/defaults")).prompts.join("\n");
    await refreshModels();
    await refreshAbl();
    renderStatus(await api("/api/status"));
    resumeLastModel();
    if ((await api("/api/models/copy")).status === "copying") pollCopy();
    const ex = await api("/api/abl/export");
    if (ex.status === "working") pollExport(ex.kind === "steer" ? "st" : "abl");
    if (!state.info.loaded && state.info.status !== "idle") pollStatus();
  } catch (e) {
    toast("Can’t reach the Lodestar engine: " + e.message, true);
  }
})();
