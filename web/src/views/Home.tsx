import { useEffect, useState } from "react";
import { drawHomeArt, type HomeArt } from "../art/home";
import { api } from "../lib/api";
import { isDir, modelName } from "../lib/format";
import { setTab } from "../shell/engine";
import { ablActive, useApp } from "../store/app";
import { useChat } from "../store/chat";
import { ICONS } from "../components/Sidebar";
import { MOD } from "../lib/platform";

const CARDS: { go: HomeArt; title: string; text: string; key: string }[] = [
  { go: "abliterate", title: "Abliterate", text: "Find the direction it refuses along, erase it, and save the result as a new model.", key: `${MOD}1` },
  { go: "steer", title: "Steer", text: "Push individual neurons to change how it talks, then save what works as recipes.", key: `${MOD}2` },
  { go: "chat", title: "Chat", text: "Talk to it with your steering and erased refusals switched on or off.", key: `${MOD}3` },
];

// Home: the loaded model, and a card for each page with what it's for and where it stands.
export function Home({ hidden }: { hidden: boolean }) {
  const s = useApp((s) => s.info);
  const cur = useApp((s) => s.abl?.current);
  const mine = useApp((s) => (s.saved || []).filter((r) => modelName(r.model) === modelName(s.info?.model)).length);
  const shown = !hidden && !!s?.loaded;

  useEffect(() => {
    if (!shown) return;
    const raf = requestAnimationFrame(() => document.querySelectorAll<HTMLCanvasElement>(".home-art")
      .forEach((c) => drawHomeArt(c, c.dataset.art as HomeArt, c.closest<HTMLElement>(".home-card"))));
    return () => cancelAnimationFrame(raf);
  }, [shown, s?.model, cur, mine]);

  const dev = s?.device && ({ mps: "Apple GPU", cuda: "NVIDIA GPU", cpu: "CPU" }[s.device] || s.device);
  const specs: [string, unknown][] = [["Parameters", s?.params && `${(s.params / 1e9).toFixed(1)}B`], ["Layers", s?.layers], ["Runs on", dev], ["Precision", s?.dtype]];
  const status: Record<HomeArt, JSX.Element | string> = {
    abliterate: cur?.active ? <><i className="hc-on" />Refusals erased at layer {cur.layer}</> : cur ? "Direction saved, switched off" : "",
    steer: mine ? `${mine} saved` : "",
    chat: "",
  };

  return (
    <section id="view-home" className="view home" data-view="home" hidden={hidden}>
      <header className="home-h">
        <h2 id="home-model">{s?.loaded ? modelName(s.model) : ""}</h2>
        <dl id="home-meta" className="home-meta">
          {s?.loaded && specs.filter(([, v]) => v).map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{String(v)}</dd></div>)}
        </dl>
      </header>
      <div className="home-cards">
        {CARDS.map((c) => (
          <button className="home-card" data-go={c.go} key={c.go} onClick={() => setTab(c.go)}>
            <canvas className="home-art" data-art={c.go} aria-hidden="true" />
            <span className="hc-b">
              <span className="hc-t"><span className="hc-ic" data-icon={c.go}>{ICONS[c.go]}</span>{c.title}</span>
              <span className="hc-d">{c.text}</span>
              <span className="hc-f"><kbd>{c.key}</kbd><span className="hc-s" id={`hc-${c.go}`}>{status[c.go]}</span></span>
            </span>
          </button>
        ))}
      </div>
      {shown && <Api />}
    </section>
  );
}

type Serve = {
  models: { id: string; about: string }[];
  log: { at: number; model: string; prompt: number; tokens: number; seconds: number; finish: string }[];
  active: number;
};

// what the default model does right now: what Chat steers with, and whether refusals are erased
function asSet(items: ReturnType<typeof useApp.getState>["recipe"]["items"], steer: boolean, erased: boolean) {
  const nd = items.filter(isDir).length, nn = items.length - nd;
  const what = [nn && `${nn} neuron${nn > 1 ? "s" : ""}`, nd && `${nd} direction${nd > 1 ? "s" : ""}`].filter(Boolean).join(", ");
  return [erased && "Refusals erased", steer && what && `Steered · ${what}`].filter(Boolean).join(" · ") || "Plain right now";
}

const clock = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

// The OpenAI-compatible API: where other apps reach this model, the names it answers to, and the latest requests
function Api() {
  const [s, setS] = useState<Serve | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const items = useApp((a) => a.recipe.items), erased = useApp(ablActive), steer = useChat((c) => c.steer);
  const url = `${location.origin}/v1`;

  useEffect(() => {
    let live = true;
    const poll = () => api<Serve>("/api/serve").then((r) => live && setS(r)).catch(() => {});
    poll();
    const t = setInterval(poll, 2500);
    return () => ((live = false), clearInterval(t));
  }, []);

  const copy = (k: string, text: string) => navigator.clipboard.writeText(text).then(() => (setCopied(k), setTimeout(() => setCopied(null), 1400)));
  const first = s?.models[0]?.id ?? "";
  const example = `from openai import OpenAI\n\nclient = OpenAI(base_url="${url}", api_key="lodestar")\nreply = client.chat.completions.create(\n    model="${first}",\n    messages=[{"role": "user", "content": "Hello!"}],\n)\nprint(reply.choices[0].message.content)`;

  return (
    <section className="step api" id="home-api">
      <div className="step-h"><div><h3>Use it from other apps</h3>
        <p className="step-sub">Any app that speaks the OpenAI API can reach this model at this address.</p></div></div>
      <div className="card">
        <div className="save-row">
          <label className="field grow">Address
            <input id="api-url" readOnly value={url} onFocus={(e) => e.currentTarget.select()} />
          </label>
          <button className="btn" onClick={() => copy("url", url)}>{copied === "url" ? "Copied" : "Copy"}</button>
        </div>
        {!!s?.models.length && <div className="api-sec">
          <div className="api-l">Models</div>
          <dl className="api-models" id="api-models">{s.models.map((m, i) => (
            <div key={m.id} data-model={m.id}>
              <dt>{m.id}</dt>
              <dd>{i ? m.about : <>{(steer && items.length > 0) || erased ? <i className="hc-on" /> : null}{m.about}: {asSet(items, steer, erased)}</>}</dd>
            </div>
          ))}</dl>
        </div>}
        <details className="adv api-ex">
          <summary>Python example</summary>
          <pre><code>{example}</code></pre>
          <button className="btn ghost sm" onClick={() => copy("ex", example)}>{copied === "ex" ? "Copied" : "Copy"}</button>
        </details>
        {!!s?.log.length && <div className="api-sec">
          <div className="api-l">Recent requests{!!s.active && <i className="hc-on" title="Answering a request" />}</div>
          <div className="api-log" id="api-log">{s.log.map((r) => (
            <div key={r.at}><span>{clock(r.at)}</span><b>{r.model}</b><span>{r.prompt} → {r.tokens} tokens</span><span>{r.seconds} s</span></div>
          ))}</div>
        </div>}
      </div>
    </section>
  );
}
