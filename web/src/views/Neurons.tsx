import { useEffect, useState } from "react";
import { STARTERS } from "../chat/engine";
import { api } from "../lib/api";
import { lensLabel } from "../lib/format";
import { addPins, clearChat, removePin, retry, send, toggleCompare, updatePin } from "../neurons/engine";
import { SEND } from "../steer/Live";
import { useNeurons, type Pin } from "../store/neurons";
import { Copy, MARKS, Md } from "./Chat";

const STEP = 0.25;
const fmt = (d: number) => (d > 0 ? "+" : "") + (+d.toFixed(2)).toString();

// a neuron's dose: typed, or nudged a quarter at a time
function Dose({ i, dose }: { i: number; dose: number }) {
  const [text, setText] = useState(fmt(dose));
  useEffect(() => setText(fmt(dose)), [dose]);

  const commit = () => {
    const d = parseFloat(text.replace(/[−–]/g, "-"));
    if (Number.isFinite(d)) updatePin(i, { dose: d });
    else setText(fmt(dose));
  };

  const step = (by: number) => updatePin(i, { dose: Math.round((dose + by) / STEP) * STEP });

  return (
    <div className={dose < 0 ? "nr-dose neg" : "nr-dose"}>
      <button aria-label="Less" onClick={() => step(-STEP)}>−</button>
      <input aria-label="Dose" value={text} spellCheck="false" onChange={(e) => setText(e.target.value)}
        onBlur={commit} onKeyDown={(e) => e.key === "Enter" && (e.currentTarget as HTMLInputElement).blur()} />
      <button aria-label="More" onClick={() => step(STEP)}>+</button>
    </div>
  );
}

function PinRow({ p, i }: { p: Pin; i: number }) {
  const words = p.lens && lensLabel(p.dose < 0 ? p.lens.down : p.lens.up);

  return (
    <div className={p.on ? "nr-pin" : "nr-pin off"} data-pin={`${p.layer}:${p.neuron}`}>
      <button className="nr-on" role="switch" aria-checked={p.on} aria-label={p.on ? "Switch off" : "Switch on"}
        onClick={() => updatePin(i, { on: !p.on })}><i /></button>
      <div className="nr-id"><b>L{p.layer}</b><span>N{p.neuron}</span></div>
      <Dose i={i} dose={p.dose} />
      <button className="nr-del" aria-label="Remove" onClick={() => removePin(i)}>×</button>
      <div className="nr-lens" title={words ? `Pushes toward: ${words}` : undefined}>{words}</div>
    </div>
  );
}

function Pins() {
  const pins = useNeurons((s) => s.pins), add = useNeurons((s) => s.add), err = useNeurons((s) => s.err), set = useNeurons((s) => s.set);

  return (
    <div className="nr-side">
      <div className="nr-side-h"><span className="eyebrow light">Neurons</span></div>
      <form className={err ? "nr-add bad" : "nr-add"} onSubmit={(e) => (e.preventDefault(), addPins())}>
        <input id="nr-add" placeholder="L14 N8949 +1" spellCheck="false" autoComplete="off" value={add}
          onChange={(e) => set({ add: e.target.value, err: "" })} />
        <button className="btn sm" id="nr-add-go" disabled={!add.trim()}>Add</button>
      </form>
      <p className="nr-err" id="nr-err" hidden={!err}>{err}</p>
      <div className="nr-pins" id="nr-pins">{pins.map((p, i) => <PinRow key={`${p.layer}:${p.neuron}`} p={p} i={i} />)}</div>
    </div>
  );
}

function Log() {
  const messages = useNeurons((s) => s.messages), streaming = useNeurons((s) => s.streaming);
  useNeurons((s) => s.v);
  const wide = messages.some((m) => m.cols);

  return (
    <div className={wide ? "chat-log wide" : "chat-log"} id="nr-log">{messages.map((m, i) => {
      if (m.role === "user") return <div key={i} className="msg user"><div className="bubble">{m.content}</div></div>;

      const last = i === messages.length - 1, typing = last && streaming;
      const acts = (
        <div className="msg-acts" hidden={typing || (!last && !m.stopped)}>
          {last && <button data-a="retry" onClick={retry}>Retry</button>}
          {m.stopped && <span className="msg-note">Stopped</span>}
        </div>
      );

      if (m.cols) {
        return (
          <div key={i} className={typing ? "msg bot cmp typing" : "msg bot cmp"}>
            <div className="cmp-cols">{m.cols.map((c) => (
              <div key={c.key} className="cmp-col" data-col={c.key}>
                <div className="cmp-h">{c.label}</div>
                <Md m={c} done={!typing} />
                <div className="msg-acts">{!typing && !!c.content && <Copy text={c.content} />}</div>
              </div>
            ))}</div>
            {acts}
          </div>
        );
      }

      return (
        <div key={i} className={typing ? "msg bot typing" : "msg bot"}>
          <Md m={m} done={!typing} />
          <div className="msg-acts">
            {!typing && <><Copy text={m.content} />{last && <button data-a="retry" onClick={retry}>Retry</button>}</>}
            {m.stopped && <span className="msg-note">Stopped</span>}
          </div>
        </div>
      );
    })}</div>
  );
}

function Composer() {
  const input = useNeurons((s) => s.input), streaming = useNeurons((s) => s.streaming), set = useNeurons((s) => s.set);

  return (
    <div className="composer-wrap">
      <form className="composer" onSubmit={(e) => (e.preventDefault(), send())}>
        <textarea id="nr-input" rows={1} placeholder="Message the model" spellCheck="false" value={input}
          onChange={(e) => {
            set({ input: e.target.value });
            e.target.style.height = "auto";
            e.target.style.height = Math.min(e.target.scrollHeight, 220) + "px";
          }}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) (e.preventDefault(), send()); }} />
        <div className="composer-bar">
          <span className="grow" />
          <button type="button" className="send stop" aria-label="Stop" hidden={!streaming} onClick={() => api("/api/cancel", {})}><i /></button>
          <button type="submit" className="send" id="nr-send" aria-label="Send" hidden={streaming} disabled={!input.trim()}>{SEND}</button>
        </div>
      </form>
    </div>
  );
}

export function Neurons({ hidden }: { hidden: boolean }) {
  const started = useNeurons((s) => s.messages.length > 0), compare = useNeurons((s) => s.compare);
  const live = useNeurons((s) => s.pins.some((p) => p.on && p.dose));

  return (
    <section id="view-neurons" className="view nr" data-view="neurons" hidden={hidden}>
      <Pins />
      <div className="nr-chat">
        <header className="chat-h">
          <span className="grow" />
          <button id="nr-compare" className="switch" role="switch" aria-checked={compare && live} disabled={!live} onClick={toggleCompare}>
            <i />Side by side
          </button>
          <button id="nr-clear" className="btn ghost sm" disabled={!started} onClick={clearChat}>Clear</button>
        </header>
        <div className="chat-scroll" id="nr-scroll">
          <Log />
          <div className="chat-empty" hidden={started}>
            <div className="chat-starters" id="nr-starters">{STARTERS.map((s, i) => (
              <button key={s} className="starter" onClick={() => send(s)}>
                <span className="starter-mark" aria-hidden="true">{MARKS[i]}</span>
                <span>{s}</span>
                <span className="starter-go" aria-hidden="true">↗</span>
              </button>
            ))}</div>
          </div>
        </div>
        <Composer />
      </div>
    </section>
  );
}
