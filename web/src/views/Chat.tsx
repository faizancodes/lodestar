import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { setAblActive } from "../abl/engine";
import {
  addNeurons, continueReply, copyReply, toggleGauge, findNeurons, growInput, jumpDown, newChat, onScroll, rateOf, rateText, recipeNeurons, replyHtml, retry, sendChat,
  setPickBy, STARTERS, toggleCompare, toggleFound, toggleNeurons, toggleSteer,
} from "../chat/engine";
import { api } from "../lib/api";
import { isDir, itemKey, labelOf, lensLabel, modelName } from "../lib/format";
import { patchAt, removeAt } from "../steer/recipe";
import { ablActive, useApp } from "../store/app";
import { SEND } from "../steer/Live";
import { useChat, type ColKey, type Mods, type Reading } from "../store/chat";
import { MOD } from "../lib/platform";

const cancel = () => api("/api/cancel", {});

// one small run of ASCII marks per starter, from the scenes' dot / plus / cross / block ramp
const MARKS = ["·+×", "+×▪", "×+·", "▪×+"];

// the switches for what shapes replies: the erased refusal direction, and the recipe that's on
function Mods() {
  const found = useApp((s) => !!s.abl?.ready && !!s.abl.current), items = useApp((s) => s.recipe.items), on = useApp(ablActive);
  const steer = useChat((s) => s.steer);
  const n = items.length, nd = items.filter(isDir).length, nn = n - nd;
  const what = [nn && `${nn} neuron${nn > 1 ? "s" : ""}`, nd && `${nd} direction${nd > 1 ? "s" : ""}`].filter(Boolean).join(", ");

  return (
    <div className="chat-mods" id="chat-mods">
      {found && <button className={on ? "mod on" : "mod"} data-mod="abl" title="Erase the refusal direction in replies"
        onClick={() => setAblActive(!on)}>Refusals erased</button>}
      {!!n && <button className={steer ? "mod on" : "mod"} data-mod="steer" title={items.map(labelOf).join(", ")}
        onClick={toggleSteer}>Steered · {what}</button>}
    </div>
  );
}

/** A reply's markdown. Set as HTML, like the streamed updates that write into it, so the decoder can work on it. */
function Md({ m, done }: { m: { content: string }; done: boolean }) {
  const el = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => { el.current!.innerHTML = replyHtml(m, done); });
  return <div className="md" ref={el} />;
}

// Compare: on when there's something to put beside the plain model
function CompareSwitch() {
  const compare = useChat((s) => s.compare), steer = useChat((s) => s.steer);
  const steered = useApp((s) => steer && s.recipe.items.length > 0), erased = useApp(ablActive);

  if (!steered && !erased) return null;

  return (
    <button id="chat-compare" className="switch" role="switch" aria-checked={compare} onClick={toggleCompare}
      title="Show each reply beside the plain model’s">
      <i />Compare
    </button>
  );
}

// Gauges: one strip per direction under a reply, a cell for each piece of text as it came, coloured toward the
// direction's pole (+) or away from it (−), full at ±0.5, about where text written in either pole's voice sits.
// Hovering a cell shows the text and every direction's reading there.
const signedV = (v: number) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(2);
const cell = (v: number) => ({ background: `color-mix(in oklab, var(--g-${v < 0 ? "neg" : "pos"}) ${Math.round(Math.min(1, Math.abs(v) * 2) * 100)}%, transparent)` });

// the few words leading up to a piece of text, to place it in the reply
const before = (read: Reading, at: number) => {
  const s = read.toks.slice(Math.max(0, at - 12), at).map((t) => t.t).join("").replace(/\s+/g, " ");
  return s.length > 28 ? "…" + s.slice(-28).replace(/^\S*\s/, "") : s;
};

function Gauges({ read }: { read?: Reading }) {
  useChat((s) => s.gv);
  const [at, setAt] = useState<number | null>(null);
  const n = read?.toks.length ?? 0;
  if (!read || !n) return null;

  return (
    <div className="gauges" onMouseLeave={() => setAt(null)}>
      {read.ids.map((id, d) => {
        const vals = read.toks.map((t) => t.v[d]), avg = vals.reduce((a, b) => a + b, 0) / n;

        return (
          <div key={id} className="gauge" data-gauge={id}>
            <span className="gauge-n">{read.names[d]}</span>
            <div className="gauge-strip">
              {vals.map((v, k) => <i key={k} className={k === at ? "at" : undefined} style={cell(v)} onMouseEnter={() => setAt(k)} />)}
            </div>
            <span className="gauge-v" title={at === null ? "Average over the reply" : undefined}>{signedV(at === null ? avg : vals[at])}</span>
          </div>
        );
      })}
      {at !== null && <div className="gauge-tip" style={{ left: `${((at + 0.5) / n) * 100}%` }}>
        <span>{before(read, at)}</span><b>{read.toks[at].t.replace(/\s+/g, " ") || "␣"}</b>
      </div>}
    </div>
  );
}

// What was on for a reply: the neurons and directions pushed with their doses, and the erased refusal direction. In
// Compare, the steered column shows what it pushed (the erased one's heading already says so).
function ModsLine({ mods, only }: { mods?: Mods; only?: ColKey }) {
  const steer = only && only !== "steer" ? undefined : mods?.steer, abl = !only && mods?.abl;
  if (!steer?.length && !abl) return null;

  return (
    <div className="msg-mods">
      {steer?.map((p, k) => (
        <span key={k} className="msg-mod"><b>{p.label}</b>{p.erase ? " erased" : ` ${p.dose > 0 ? "+" : "−"}${+Math.abs(p.dose).toFixed(2)}`}</span>
      ))}
      {abl && <span className="msg-mod"><b>Refusals erased</b></span>}
    </div>
  );
}

// the directions to read as replies are written, picked from the ones the loaded model has
function GaugesPick() {
  const dirs = (useApp((s) => s.dirs) ?? []).filter((d) => d.ready), on = useChat((s) => s.gauges);
  const box = useRef<HTMLDetailsElement>(null);
  const n = dirs.filter((d) => on.includes(d.id)).length;

  useEffect(() => {
    const close = (e: MouseEvent) => !(e.target as Element).closest("#chat-gauges") && box.current?.removeAttribute("open");
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, []);

  return (
    <details className="gauges-pick" id="chat-gauges" ref={box}>
      <summary className={n ? "btn ghost sm on" : "btn ghost sm"}>Gauges{n ? ` · ${n}` : ""}</summary>
      <div className="gauges-pop">
        {dirs.length ? dirs.map((d) => (
          <button key={d.id} className="switch" role="switch" aria-checked={on.includes(d.id)} data-dir={d.id} onClick={() => toggleGauge(d.id)}>
            <i />{d.name}
          </button>
        )) : <p className="gauges-none">This model has no directions yet. Make one in Steer.</p>}
      </div>
    </details>
  );
}

// how fast it wrote: live while it streams (written straight into the element), then the final figure
const Rate = ({ part }: { part: { content: string; wps?: number } }) => <span className="msg-rate">{rateText(rateOf(part))}</span>;

function Copy({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const toast = useApp((s) => s.toastMsg);

  const copy = async () => {
    try { await copyReply(text); setCopied(true); setTimeout(() => setCopied(false), 1400); }
    catch { toast("Couldn’t copy.", true); }
  };

  return <button data-a="copy" onClick={copy}>{copied ? "Copied" : "Copy"}</button>;
}

function Log() {
  const messages = useChat((s) => s.messages), streaming = useChat((s) => s.streaming);
  useChat((s) => s.v);
  const wide = messages.some((m) => m.cols);

  return (
    <div className={wide ? "chat-log wide" : "chat-log"} id="chat-log">{messages.map((m, i) => {
      if (m.role === "user") return <div key={i} className="msg user" data-i={i}><div className="bubble">{m.content}</div></div>;

      const last = i === messages.length - 1, typing = last && streaming;

      if (m.cols) {
        return (
          <div key={i} className={typing ? "msg bot cmp typing" : "msg bot cmp"} data-i={i}>
            <div className="cmp-cols">{m.cols.map((c) => (
              <div key={c.key} className="cmp-col" data-col={c.key}>
                <div className="cmp-h">{c.label}</div>
                <Md m={c} done={!typing} />
                <Gauges read={c.read} />
                <ModsLine mods={m.mods} only={c.key} />
                <div className="msg-acts">
                  {!typing && !!c.content && <Copy text={c.content} />}
                  <Rate part={c} />
                </div>
              </div>
            ))}</div>
            <div className="msg-acts" hidden={typing || (!last && !m.stopped)}>
              {last && m.cols.some((c) => c.cut) && <button data-a="continue" className="hint" onClick={continueReply}>Continue</button>}
              {last && <button data-a="retry" onClick={retry}>Retry</button>}
              {m.stopped && <span className="msg-note">Stopped</span>}
            </div>
          </div>
        );
      }

      return (
        <div key={i} className={typing ? "msg bot typing" : "msg bot"} data-i={i}>
          <Md m={m} done={!typing} />
          <Gauges read={m.read} />
          <ModsLine mods={m.mods} />
          <div className="msg-acts">
            {!typing && <>
              {last && m.cut && <button data-a="continue" className="hint" onClick={continueReply}>Continue</button>}
              <Copy text={m.content} />{last && <button data-a="retry" onClick={retry}>Retry</button>}</>}
            {m.stopped && <span className="msg-note">Stopped</span>}
            <Rate part={m} />
          </div>
        </div>
      );
    })}</div>
  );
}

// how far along its range a slider is, for its track to light up to there
const fill = (v: number, lo: number, hi: number) => ({ "--fill": `${((v - lo) / (hi - lo)) * 100}%` }) as React.CSSProperties;

const SLIDERS = (
  <svg viewBox="0 0 16 16" aria-hidden="true">
    <path d="M2 4.5h7M13 4.5h1M2 11.5h1M7 11.5h7" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    <circle cx="11" cy="4.5" r="1.9" fill="none" stroke="currentColor" strokeWidth="1.5" />
    <circle cx="5" cy="11.5" r="1.9" fill="none" stroke="currentColor" strokeWidth="1.5" />
  </svg>
);

function Settings() {
  const system = useChat((s) => s.system), temp = useChat((s) => s.temp), max = useChat((s) => s.max), set = useChat((s) => s.set);
  const box = useRef<HTMLDetailsElement>(null);

  // a click anywhere else closes the settings
  useEffect(() => {
    const close = (e: MouseEvent) => !(e.target as Element).closest("#chat-opts") && box.current!.removeAttribute("open");
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, []);

  return (
    <details className="chat-opts" id="chat-opts" ref={box}>
      <summary aria-label="Settings" title="Settings">{SLIDERS}</summary>
      <div className="chat-opts-pop">
        <label className="field">System prompt
          <textarea id="chat-system" rows={3} spellCheck="false" placeholder="Optional" value={system} onChange={(e) => set({ system: e.target.value })} />
        </label>
        <label className="field">Temperature
          <div className="slider-row"><input id="chat-temp" type="range" min="0" max="1.5" step="0.05" value={temp} style={fill(+temp, 0, 1.5)} onChange={(e) => set({ temp: e.target.value })} /><output id="chat-temp-v">{(+temp).toFixed(2)}</output></div>
        </label>
        <label className="field">Reply length
          <div className="slider-row"><input id="chat-max" type="range" min="64" max="1024" step="32" value={max} style={fill(+max, 64, 1024)} onChange={(e) => set({ max: e.target.value })} /><output id="chat-max-v">{+max}</output></div>
        </label>
      </div>
    </details>
  );
}

function Composer() {
  const input = useChat((s) => s.input), streaming = useChat((s) => s.streaming), jump = useChat((s) => s.jump), set = useChat((s) => s.set);

  return (
    <div className="composer-wrap">
      <button className="jump" id="chat-jump" aria-label="Scroll to the latest message" hidden={!jump} onClick={jumpDown}>↓</button>
      <form className="composer" id="chat-form" onSubmit={(e) => (e.preventDefault(), sendChat())}>
        <textarea id="chat-input" rows={1} placeholder="Message the model" spellCheck="false" value={input}
          onChange={(e) => (set({ input: e.target.value }), growInput())}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) (e.preventDefault(), sendChat()); }} />
        <div className="composer-bar">
          <Settings />
          <span className="grow" />
          <button type="button" className="send stop" id="chat-stop" aria-label="Stop" hidden={!streaming} onClick={cancel}><i /></button>
          <button type="submit" className="send" id="chat-send" aria-label="Send" hidden={streaming} disabled={!input.trim()}>
            {SEND}
          </button>
        </div>
      </form>
    </div>
  );
}

// ------------------------------------------------------------------ neurons by hand
const STEP = 0.25;
const signed = (d: number) => (d > 0 ? "+" : "") + +d.toFixed(2);

// a neuron's dose: typed, or nudged a quarter at a time
function Dose({ i, dose }: { i: number; dose: number }) {
  const [text, setText] = useState(signed(dose));
  useEffect(() => setText(signed(dose)), [dose]);

  const commit = () => {
    const d = parseFloat(text.replace(/[−–]/g, "-"));
    if (Number.isFinite(d) && d) patchAt(i, { dose: d });
    else setText(signed(dose));
  };

  const step = (by: number) => {
    const d = Math.round((dose + by) / STEP) * STEP;
    if (d) patchAt(i, { dose: d });
  };

  return (
    <div className={dose < 0 ? "nr-dose neg" : "nr-dose"}>
      <button aria-label="Less" onClick={() => step(-STEP)}>−</button>
      <input aria-label="Dose" value={text} spellCheck="false" onChange={(e) => setText(e.target.value)}
        onBlur={commit} onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()} />
      <button aria-label="More" onClick={() => step(STEP)}>+</button>
    </div>
  );
}

// The neurons behind a word, strongest first, each a card to switch on or off
function Found() {
  const found = useChat((s) => s.found)!;
  const items = useApp((s) => s.recipe.items);
  const on = new Set(items.map(itemKey));

  return (
    <div className="nr-found" id="chat-found">
      {found.partial && <p className="nr-note">“{found.word}” is split into pieces, so these match “{found.piece}”.</p>}
      {found.results.map((r) => {
        const k = `${r.layer}:${r.neuron}`, lit = on.has(k);

        return (
          <button key={k} className="nr-card" data-card={k} role="switch" aria-checked={lit} onClick={() => toggleFound(r)}>
            <div className="nr-id"><b>L{r.layer}</b><span>N{r.neuron}</span><em>{r.score > 0 ? "up" : "down"}</em></div>
            <i />
            <div className="nr-lens">{lensLabel(r.lens)}</div>
          </button>
        );
      })}
    </div>
  );
}

// The neurons in the recipe that's on, to add to by layer and number and to dose by hand
function Neurons() {
  const open = useChat((s) => s.neurons), add = useChat((s) => s.add), err = useChat((s) => s.addErr), lens = useChat((s) => s.lens);
  const set = useChat((s) => s.set);
  const by = useChat((s) => s.pickBy), find = useChat((s) => s.find), found = useChat((s) => s.found), finding = useChat((s) => s.finding);
  const items = useApp((s) => s.recipe.items);

  if (!open) return null;

  return (
    <aside className="nr" id="chat-neurons">
      <div className="nr-h">
        <span className="eyebrow light">Neurons</span>
        <button className="nr-x" aria-label="Close" onClick={toggleNeurons}>×</button>
      </div>
      <div className="seg nr-by" id="chat-pick-by">
        <button className={by === "type" ? "on" : undefined} data-by="type" onClick={() => setPickBy("type")}>By number</button>
        <button className={by === "map" ? "on" : undefined} data-by="map" onClick={() => setPickBy("map")}>By word</button>
      </div>
      {by === "type" ? (
        <form className={err ? "nr-add bad" : "nr-add"} onSubmit={(e) => (e.preventDefault(), addNeurons())}>
          <input id="chat-add" placeholder="L14 N8949 +1" spellCheck="false" autoComplete="off" value={add}
            onChange={(e) => set({ add: e.target.value, addErr: "" })} />
          <button className="btn sm" disabled={!add.trim()}>Add</button>
        </form>
      ) : (
        <>
          <form className={err ? "nr-add bad" : "nr-add"} onSubmit={(e) => (e.preventDefault(), findNeurons())}>
            <input id="chat-find" placeholder="evil" spellCheck="false" autoComplete="off" value={find}
              onChange={(e) => set({ find: e.target.value, addErr: "" })} />
            <button className="btn sm" disabled={!find.trim() || finding}>{finding ? "Finding…" : "Find"}</button>
          </form>
          {found && <Found />}
        </>
      )}
      <p className="nr-err" id="chat-add-err" hidden={!err}>{err}</p>
      <div className="nr-pins">{recipeNeurons(items).map(({ it, i }) => {
        const l = lens[`${it.layer}:${it.neuron}`], words = l && lensLabel(it.dose < 0 ? l.down : l.up);

        return (
          <div key={`${it.layer}:${it.neuron}`} className="nr-pin" data-pin={`${it.layer}:${it.neuron}`}>
            <div className="nr-id"><b>L{it.layer}</b><span>N{it.neuron}</span></div>
            <Dose i={i} dose={it.dose} />
            <button className="nr-del" aria-label="Remove" onClick={() => removeAt(i)}>×</button>
            <div className="nr-lens" title={words ? `Pushes toward: ${words}` : undefined}>{words}</div>
          </div>
        );
      })}</div>
    </aside>
  );
}

function NeuronsButton() {
  const open = useChat((s) => s.neurons);
  return <button id="chat-neurons-btn" className={open ? "btn ghost sm on" : "btn ghost sm"} aria-expanded={open} onClick={toggleNeurons}>Neurons</button>;
}

export function Chat({ hidden }: { hidden: boolean }) {
  const model = useApp((s) => (s.info?.loaded ? modelName(s.info.model) ?? "" : ""));
  const started = useChat((s) => s.messages.length > 0);

  return (
    <section id="view-chat" className="view chat" data-view="chat" hidden={hidden}>
      <div className="chat-main">
        <header className="chat-h">
          <div className="chat-title"><b id="chat-model">{model}</b></div>
          <Mods />
          <div className="chat-acts">
            <CompareSwitch />
            <GaugesPick />
            <NeuronsButton />
            <button id="chat-new" className="btn ghost sm" title={`New chat (${MOD}K)`} onClick={newChat}>New chat</button>
          </div>
        </header>
        <div className="chat-scroll" id="chat-scroll" onScroll={onScroll}>
          <Log />
          <div className="chat-empty" id="chat-empty" hidden={started}>
            <h2>What’s on your mind?</h2>
            <div className="chat-starters" id="chat-starters">{STARTERS.map((s, i) => (
              <button key={s} className="starter" onClick={() => sendChat(s)}>
                <span className="starter-mark" aria-hidden="true">{MARKS[i]}</span>
                <span>{s}</span>
                <span className="starter-go" aria-hidden="true">↗</span>
              </button>
            ))}</div>
          </div>
        </div>
        <Composer />
      </div>
      <Neurons />
    </section>
  );
}
