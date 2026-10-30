import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { setAblActive } from "../abl/engine";
import {
  copyReply, growInput, jumpDown, newChat, onScroll, replyHtml, retry, sendChat, STARTERS, toggleSteer,
} from "../chat/engine";
import { api } from "../lib/api";
import { isDir, labelOf, modelName } from "../lib/format";
import { ablActive, useApp } from "../store/app";
import { SEND } from "../steer/Live";
import { useChat, type Msg } from "../store/chat";

const cancel = () => api("/api/cancel", {});

// the switches for what shapes replies: the erased refusal direction, and the recipe that's on
function Mods() {
  const abl = useApp((s) => s.abl), items = useApp((s) => s.recipe.items), on = useApp(ablActive);
  const steer = useChat((s) => s.steer);
  const n = items.length, nd = items.filter(isDir).length, nn = n - nd;
  const what = [nn && `${nn} neuron${nn > 1 ? "s" : ""}`, nd && `${nd} direction${nd > 1 ? "s" : ""}`].filter(Boolean).join(", ");
  return (
    <div className="chat-mods" id="chat-mods">
      {abl?.ready && abl.current && <button className={on ? "mod on" : "mod"} data-mod="abl" title="Erase the refusal direction in replies"
        onClick={() => setAblActive(!on)}>Refusals erased</button>}
      {!!n && <button className={steer ? "mod on" : "mod"} data-mod="steer" title={items.map(labelOf).join(", ")}
        onClick={toggleSteer}>Steered · {what}</button>}
    </div>
  );
}

/** A reply's markdown. Set as HTML, like the streamed updates that write into it, so the decoder can work on it. */
function Md({ m }: { m: Msg }) {
  const el = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => { el.current!.innerHTML = replyHtml(m); });
  return <div className="md" ref={el} />;
}

function Copy({ m }: { m: Msg }) {
  const [copied, setCopied] = useState(false);
  const toast = useApp((s) => s.toastMsg);
  const copy = async () => {
    try { await copyReply(m); setCopied(true); setTimeout(() => setCopied(false), 1400); }
    catch { toast("Couldn’t copy.", true); }
  };
  return <button data-a="copy" onClick={copy}>{copied ? "Copied" : "Copy"}</button>;
}

function Log() {
  const messages = useChat((s) => s.messages), streaming = useChat((s) => s.streaming);
  useChat((s) => s.v);
  return (
    <div className="chat-log" id="chat-log">{messages.map((m, i) => {
      if (m.role === "user") return <div key={i} className="msg user" data-i={i}><div className="bubble">{m.content}</div></div>;
      const last = i === messages.length - 1, typing = last && streaming;
      return (
        <div key={i} className={typing ? "msg bot typing" : "msg bot"} data-i={i}>
          <Md m={m} />
          <div className="msg-acts" hidden={typing}>
            <Copy m={m} />{last && <button data-a="retry" onClick={retry}>Retry</button>}
            {m.stopped && <span className="msg-note">Stopped</span>}
          </div>
        </div>
      );
    })}</div>
  );
}

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
      <summary>Settings</summary>
      <div className="chat-opts-pop">
        <label className="field">System prompt
          <textarea id="chat-system" rows={3} spellCheck="false" placeholder="Optional" value={system} onChange={(e) => set({ system: e.target.value })} />
        </label>
        <label className="field">Temperature
          <div className="slider-row"><input id="chat-temp" type="range" min="0" max="1.5" step="0.05" value={temp} onChange={(e) => set({ temp: e.target.value })} /><output id="chat-temp-v">{(+temp).toFixed(2)}</output></div>
        </label>
        <label className="field">Reply length
          <div className="slider-row"><input id="chat-max" type="range" min="64" max="1024" step="32" value={max} onChange={(e) => set({ max: e.target.value })} /><output id="chat-max-v">{+max}</output></div>
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

export function Chat({ hidden }: { hidden: boolean }) {
  const model = useApp((s) => (s.info?.loaded ? modelName(s.info.model) ?? "" : ""));
  const started = useChat((s) => s.messages.length > 0);
  useChat((s) => s.v);
  return (
    <section id="view-chat" className="view chat" data-view="chat" hidden={hidden}>
      <header className="chat-h">
        <div className="chat-title"><b id="chat-model">{model}</b></div>
        <Mods />
        <button id="chat-new" className="btn ghost sm" title="New chat (⌘K)" onClick={newChat}>New chat</button>
      </header>
      <div className="chat-scroll" id="chat-scroll" onScroll={onScroll}>
        <Log />
        <div className="chat-empty" id="chat-empty" hidden={started}>
          <h2>What’s on your mind?</h2>
          <div className="chat-starters" id="chat-starters">{STARTERS.map((s) => (
            <button key={s} className="starter" onClick={() => sendChat(s)}>{s}</button>
          ))}</div>
        </div>
      </div>
      <Composer />
    </section>
  );
}
