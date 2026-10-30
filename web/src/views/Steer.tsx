import { useEffect, useRef, type CSSProperties, type ReactNode } from "react";
import { drawHomeArt } from "../art/home";
import { api } from "../lib/api";
import { bandText, isDir, steerDefaultName } from "../lib/format";
import { loadModel } from "../shell/engine";
import { useApp } from "../store/app";
import { useSteer } from "../store/steer";
import { findChanges, setInput } from "../steer/describe";
import { deleteDir, dnExample, findDirection, importDirs, setDn, toggleDn, TRIES, turnOnDir, turnOnMade } from "../steer/directions";
import { bakeSteer, cancelExport } from "../steer/exporting";
import { FoundRow } from "../steer/Found";
import { LiveLine, Out, SEND } from "../steer/Live";
import { Plan } from "../steer/Plan";
import { RecipeBox } from "../steer/RecipeBox";
import { addItems, replaceRecipe } from "../steer/recipe";
import { FIX_CHIPS, newQuestion, PROMPTS, send, submit, tlMode } from "../steer/refine";
import { Thread } from "../steer/Thread";

const cancel = () => api("/api/cancel", {});
const n3 = { "--n": 3 } as CSSProperties;

// the status panel: how many neurons are on, under which recipe
function StatusCard() {
  const panel = useRef<HTMLDivElement>(null), art = useRef<HTMLCanvasElement>(null);
  const items = useApp((s) => s.recipe.items), name = useApp((s) => s.recipe.name);
  const shown = useApp((s) => s.tab === "steer" && !!s.info?.loaded);
  const n = items.length, nd = items.filter(isDir).length;
  useEffect(() => {
    if (!shown) return;
    const raf = requestAnimationFrame(() => drawHomeArt(art.current!, "steer", panel.current));
    return () => cancelAnimationFrame(raf);
  }, [shown, n, nd, name]);
  // the recipe's name is the headline, so a long one wraps across the panel instead of squeezing the facts
  const title = n ? name || "Unsaved recipe" : "No recipe on";
  const facts: [string, ReactNode][] = [...(n - nd ? [["Neurons on", n - nd] as [string, number]] : []), ...(nd ? [["Directions on", nd] as [string, number]] : []),
    ...(n ? [] : [["Neurons on", 0] as [string, number]]), ["Status", n ? <><i className="hc-on" />Steering every reply</> : "Off"]];
  return (
    <div className={n > 0 ? "status-panel on" : "status-panel"} id="steer-status" ref={panel}>
      <canvas id="steer-status-art" aria-hidden="true" ref={art} />
      <div className="status-panel-b">
        <span className="eyebrow">Steering</span>
        <h3 className={!n || !name ? "sp-title dim" : "sp-title"} id="steer-title" title={title}>{title}</h3>
        <dl className="facts" id="steer-facts">{facts.map(([k, v]) => <div key={k}><dt>{k}</dt><dd className={v === "Off" ? "dim" : undefined}>{v}</dd></div>)}</dl>
      </div>
    </div>
  );
}

// examples: one click sets the changes and runs the example question
function Examples() {
  const ex = useApp((s) => s.examples);
  return (
    <div className="ex-line" id="ex-line" hidden={!ex.length}><span>Or try</span><div id="ex-chips" className="chips">{ex.map((e, i) => (
      <button key={i} className="chip ex-chip" data-i={i} title={e.note} onClick={() => {
        replaceRecipe({ id: null, name: e.name, items: [], model: useApp.getState().info!.model });
        addItems(e.items.map((x) => ({ ...x })), true);
        send(e.prompt);
      }}>{e.name}</button>
    ))}</div></div>
  );
}

function DirChips() {
  const dirs = useApp((s) => s.dirs), loaded = useApp((s) => !!s.info?.loaded);
  return (
    <div className="ex-line" id="dir-line" hidden={dirs === null || !loaded}><span>Directions</span><div id="dir-chips" className="chips">{dirs && <>
      {dirs.map((d) => (
        <button key={d.id} className={d.ready ? "chip dir-chip" : "chip dir-chip gone"} data-id={d.id}
          title={bandText(d.band) + (d.auc != null ? ` · separates ${Math.round(d.auc * 100)}%` : "")} onClick={() => turnOnDir(d)}>
          {d.name}<span className="del" aria-label="Delete" onClick={(e) => (e.stopPropagation(), deleteDir(d))}> ×</span></button>
      ))}
      <button className="chip dir-new" id="dir-new" onClick={toggleDn}>New</button>
    </>}</div></div>
  );
}

function NewDirection() {
  const d = useSteer((s) => s.dn);
  const hint = d.kind === "persona" ? "one system prompt per line" : "one prompt per line";
  return (
    <div id="dn" className="dn" hidden={!d.open}>
      <div className="dn-h"><b>New direction</b><span className="grow" />
        <div className="seg" id="dn-kind">
          <button data-k="persona" className={d.kind === "persona" ? "on" : ""} onClick={() => setDn({ kind: "persona" })}>Personas</button>
          <button data-k="topic" className={d.kind === "topic" ? "on" : undefined} onClick={() => setDn({ kind: "topic" })}>Prompts</button>
        </div>
        <button id="dn-ex" className="btn ghost sm" onClick={dnExample}>Example</button><button id="dn-close" className="x" aria-label="Close" onClick={() => setDn({ open: false })}>×</button></div>
      <label className="field">Name<input id="dn-name" placeholder="e.g. Nationalist" spellCheck="false" value={d.name} onChange={(e) => setDn({ name: e.target.value })} /></label>
      <div className="dn-poles">
        <label className="field">Toward <span className="opt" id="dn-pos-opt">{hint}</span><textarea id="dn-pos" rows={4} spellCheck="false" value={d.pos} onChange={(e) => setDn({ pos: e.target.value })} /></label>
        <label className="field">Away from <span className="opt" id="dn-neg-opt">{hint}</span><textarea id="dn-neg" rows={4} spellCheck="false" value={d.neg} onChange={(e) => setDn({ neg: e.target.value })} /></label>
      </div>
      <div className="row tight"><button id="dn-go" className="btn" onClick={findDirection}>Find direction</button><button className="btn ghost" data-cancel="" onClick={cancel}>Stop</button>
        <span className="grow" />
        <input id="dn-path" className="dn-path" placeholder="or import a .npz path" spellCheck="false" value={d.path} onChange={(e) => setDn({ path: e.target.value })} />
        <button id="dn-import" className="btn ghost sm" onClick={importDirs}>Import</button></div>
      <LiveLine id="dn-status" className="status-line" live={d.live} />
      <div id="dn-result" hidden={!d.result}>
        <dl className="facts dn-facts" id="dn-facts">{d.result?.facts.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
        <div className="run-q" id="dn-q">{d.result?.q}</div>
        <div className="cols" style={n3} id="dn-tries">{d.result?.tries.map((s, i) => (
          <div key={i} className={i === 1 ? "out" : "out steered"}><div className="out-h">{TRIES[i]}</div><Out s={s} className={d.typing ? "out-b typing" : "out-b"} /></div>
        ))}</div>
        <div className="row"><button id="dn-on" className="btn" onClick={turnOnMade}>Turn on</button></div>
      </div>
    </div>
  );
}

function Results() {
  const r = useSteer((s) => s.results);
  if (!r) return <div id="c-results" />;
  return (
    <div id="c-results" key={r.n}>
      {r.word && <div className="found">{r.word.map((x, i) => <FoundRow key={i} x={x} scope="c" />)}</div>}
      {r.plan && <Plan plan={r.plan} />}
    </div>
  );
}

function Step1() {
  const input = useSteer((s) => s.input), note = useSteer((s) => s.note), prompts = useSteer((s) => s.prompts);
  const done = useApp((s) => s.recipe.items.length > 0);
  return (
    <div className={done ? "step done" : "step"} id="st-s1">
      <div className="step-h"><span className="num">01</span><div><h3>Change how it acts</h3>
        <p className="step-sub">Describe a behaviour or a word. It searches the model for the neurons behind it and turns on the ones that work.</p></div></div>
      <div className="card change">
        <div className="ask">
          <input id="c-input" placeholder="How should it act? e.g. talk like a pirate, or a word like cookie" spellCheck="false" autoComplete="off"
            value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") findChanges(); }} />
          <button className="btn ghost" data-cancel="" onClick={cancel}>Stop</button>
          <button id="c-go" className="btn" onClick={findChanges}>Find</button>
        </div>
        <Examples />
        <DirChips />
        <div id="c-status" className="status-line">{note}</div>
        <Results />
        <details className="adv">
          <summary>Options</summary>
          <label className="field">Words it must say <span className="opt">for sentences</span>
            <input id="d-focus" placeholder="e.g. arr, matey" spellCheck="false" />
          </label>
          <label className="field">Test questions <span className="opt">one per line</span><textarea id="d-prompts" rows={5} spellCheck="false"
            value={prompts} onChange={(e) => useSteer.setState({ prompts: e.target.value })} /></label>
          <div className="adv-grid">
            <label className="field">Most neurons<input id="d-maxn" type="number" defaultValue="6" min="1" max="16" /></label>
            <label className="field">Neurons to test<input id="d-cands" type="number" placeholder="auto" min="8" max="128" /></label>
            <label className="field">Strengths to try<input id="d-doses" defaultValue="0.25, 0.5, 1.0" /></label>
            <label className="field">Allowed quality loss<input id="d-flu" type="number" defaultValue="0.35" step="0.05" min="0" /></label>
          </div>
          <label className="check"><input id="w-last" type="checkbox" /> Search the last layers for words <span className="opt">they tend to repeat the word</span></label>
        </details>
        <NewDirection />
        <RecipeBox />
      </div>
    </div>
  );
}

function Step2() {
  const input = useSteer((s) => s.tInput), tl = useSteer((s) => s.tl);
  useSteer((s) => s.v);
  const fix = tlMode(tl) === "fix", box = useRef<HTMLInputElement>(null);
  // a fix chip adds to what's written; a question chip asks it
  const chip = (c: string) => {
    if (!fix) return send(c);
    const t = input.trim();
    useSteer.setState({ tInput: t ? `${t.replace(/[.\s]*$/, "")}. ${c}.` : `${c}.` });
    box.current!.focus();
  };
  return (
    <div className="step" id="st-s2">
      <div className="step-h"><span className="num">02</span><div><h3>Try it and refine</h3>
        <p className="step-sub">Ask it something. If the answer isn’t right, say what should change and it tries again.</p></div></div>
      <Thread />
      <form id="tl-form" className="tl-comp" autoComplete="off" onSubmit={(e) => (e.preventDefault(), submit())}>
        <div className="composer">
          <input id="t-prompt" ref={box} spellCheck="false" placeholder={fix ? "What should change?" : "Ask it something"} value={input}
            onChange={(e) => useSteer.setState({ tInput: e.target.value })} />
          <button type="button" className="tl-newq" id="tl-newq" hidden={!tl?.versions.length} onClick={newQuestion}>{fix ? "New question" : "Cancel"}</button>
          <button type="button" className="send stop" id="tl-stop" aria-label="Stop" onClick={cancel}><i /></button>
          <button type="submit" className="send" id="t-go" aria-label="Send">{SEND}</button>
        </div>
        <div className="tl-sugg" id="t-chips">{(fix ? FIX_CHIPS : PROMPTS).map((p) => <button key={p} type="button" className="chip" onClick={() => chip(p)}>{p}</button>)}</div>
      </form>
    </div>
  );
}

function Step3() {
  const n = useApp((s) => s.recipe.items.length), model = useApp((s) => s.info?.model), name = useApp((s) => s.recipe.name);
  const ablRow = useApp((s) => !!s.abl?.current?.active);
  const exp = useSteer((s) => s.exp), saved = exp.saved.st;
  return (
    <div className={n ? "step" : "step locked"} id="st-s3">
      <div className="step-h"><span className="num">03</span><div><h3>Save it as a new model</h3>
        <p className="step-sub">Writes a copy with these neurons built into its weights. It steers exactly when loaded here, and very nearly the same in other apps.</p></div></div>
      <div className="card">
        <div className="save-row">
          <label className="field grow">Name
            <input id="st-name" spellCheck="false" autoComplete="off" placeholder={steerDefaultName(model, name)} />
          </label>
          <button id="st-bake" className="btn" disabled={exp.running} onClick={bakeSteer}>Save model</button>
          <button id="st-export-stop" className="btn ghost" hidden={!(exp.running && exp.page === "st")} onClick={cancelExport}>Stop</button>
        </div>
        <label className="check" id="st-abl-row" hidden={!ablRow}><input id="st-abl" type="checkbox" defaultChecked /> Keep its refusals erased too</label>
        <div id="st-export" className="export" hidden={exp.page !== "st"}>
          <div className="export-bar"><i style={exp.width ? { width: exp.width } : undefined} /></div>
          <div className="export-line status-line">{exp.line}</div>
        </div>
        <div id="st-saved" className="saved-model" hidden={!saved}>
          <div><b>{saved?.name}</b><span>On this Mac</span></div>
          <button className="btn" id="st-load" onClick={() => saved && loadModel(saved.dst)}>Load it</button>
        </div>
      </div>
    </div>
  );
}

export function Steer({ hidden }: { hidden: boolean }) {
  return (
    <section id="view-steer" className="view page" data-view="steer" hidden={hidden}>
      <header className="page-h">
        <h2>Steer</h2>
        <p>Find the neurons behind a behaviour, push them up or down, and see how its answers change.</p>
      </header>
      <StatusCard />
      <div className="flow">
        <Step1 />
        <Step2 />
        <Step3 />
      </div>
    </section>
  );
}
