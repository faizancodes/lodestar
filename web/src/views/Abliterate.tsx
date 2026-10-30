import { useEffect, useRef, useState } from "react";
import {
  ablAsk, analyze, backToCoef, bakeAbl, nudge, resetProbes, saveProbes, setName, toggleAbl, undoNudge,
} from "../abl/engine";
import { drawHomeArt } from "../art/home";
import { api } from "../lib/api";
import { loadModel } from "../shell/engine";
import { touchAbl, useAbl, type Entry } from "../store/abl";
import { ablActive, useApp } from "../store/app";
import { useSteer } from "../store/steer";
import { cancelExport } from "../steer/exporting";
import { Ans, LiveLine, SEND } from "../steer/Live";
import { setCompare } from "../steer/refine";

const cancel = () => api("/api/cancel", {});
const xs = (c: number | null) => (c == null ? "" : `${(+c).toFixed(2)}×`);
const STEPS = ["Read", "Compare", "Try layers", "Check"];

const act = (label: string, onClick: () => void, cls = "") => <button className={"tl-act" + cls} onClick={onClick}>{label}</button>;
const verdict = (ref: boolean | null, worse = false) => ref == null ? null
  : <span className={ref ? "tl-n bad" : "tl-n"}>{ref ? "Refused" : "Answered"}{worse ? " · writing got worse" : ""}</span>;

const useReady = () => useApp((s) => !!s.abl?.ready && !!s.abl.current);
const count = (text: string) => text.split("\n").filter((l) => l.trim()).length;

// the status panel: where the direction is, how strongly it's erased, and the switch for it
function StatusCard() {
  const panel = useRef<HTMLDivElement>(null), art = useRef<HTMLCanvasElement>(null);
  const c = useApp((s) => (s.abl?.ready ? s.abl.current : null));
  const shown = useApp((s) => s.tab === "abliterate" && !!s.info?.loaded);
  const on = useApp(ablActive), ready = useReady();

  useEffect(() => {
    if (!shown) return;
    const raf = requestAnimationFrame(() => drawHomeArt(art.current!, "abliterate", panel.current));
    return () => cancelAnimationFrame(raf);
  }, [shown, c?.active, c?.layer, c?.coef]);

  return (
    <div className={c?.active ? "status-panel on" : "status-panel"} id="abl-status" ref={panel}>
      <canvas id="abl-status-art" aria-hidden="true" ref={art} />
      <div className="status-panel-b">
        <span className="eyebrow">Refusal direction</span>
        <dl className="facts" id="abl-facts">{c ? <>
          <div><dt>Layer</dt><dd>{c.layer}</dd></div>
          <div><dt>Strength</dt><dd>{`${(+c.coef || 1).toFixed(2)}×`}</dd></div>
        </> : <div><dt>Status</dt><dd className="dim">Not found yet</dd></div>}</dl>
        <button id="abl-toggle" className="switch" role="switch" aria-checked={ready ? on : false} hidden={!ready}
          title={ready ? (on ? "On for every reply in Steer and Chat" : "Off: the model answers as it normally would") : undefined}
          onClick={toggleAbl}><span>Erase refusals</span><i /></button>
      </div>
    </div>
  );
}

function Rank() {
  const r = useAbl((s) => s.rank), n = useAbl((s) => s.rankN);
  if (!r) return <div id="abl-rank" />;
  const top = Math.max(r.top_score, 1e-9);

  return (
    <div id="abl-rank"><details className="adv" key={n}><summary>Match by layer</summary>{r.ranking.map((x) => (
      <div key={x.layer} className={x.layer === r.layer ? "rank-row sel" : "rank-row"}>
        <span className="nid">L{x.layer}</span><span className="pt-bar"><i style={{ width: `${Math.round((x.score / top) * 100)}%` }} /></span>
        <span className="rk-v">{Math.round(x.score * 100)}%</span>
        <span className="rk-tag">{x.tried ? `${x.refused} of ${x.tried} refused` : "not tried"}{x.layer === r.layer ? " · chosen" : ""}</span>
      </div>
    ))}</details></div>
  );
}

function Step1() {
  const ready = useReady();
  const plan = useAbl((s) => s.plan);
  const harmful = useAbl((s) => s.harmful), harmless = useAbl((s) => s.harmless);
  const [editing, setEditing] = useState(false);

  return (
    <div className={ready ? "step done" : "step"} id="abl-s1">
      <div className="step-h"><span className="num">01</span><div><h3>Find the refusal direction</h3>
        <p className="step-sub">Reads how the model handles bad and everyday requests, then tests which layer erases its refusals. Takes a few minutes.</p></div></div>
      <div className="card abl">
        <div className="abl-find">
          <div className="abl-q">
            <span><b>{count(harmful)}</b> questions it should refuse</span>
            <span><b>{count(harmless)}</b> everyday questions</span>
            <button id="abl-edit" className="link" aria-expanded={editing} onClick={() => setEditing(!editing)}>
              {editing ? "Done" : "Edit"}
            </button>
          </div>
          <button className="btn ghost" data-cancel="" onClick={cancel}>Stop</button>
          <button id="abl-go" className={ready ? "btn ghost" : "btn"} disabled={plan.running} onClick={analyze}>{ready ? "Find it again" : "Find it"}</button>
        </div>

        {editing && (
          <div className="abl-probes" id="abl-probes">
            <label className="field">Questions it should refuse <span className="opt">one per line</span>
              <textarea id="abl-probes-harmful" rows={8} spellCheck="false" value={harmful} onChange={(e) => useAbl.setState({ harmful: e.target.value, dirty: true })} />
            </label>
            <label className="field">Everyday questions <span className="opt">one per line</span>
              <textarea id="abl-probes-harmless" rows={8} spellCheck="false" value={harmless} onChange={(e) => useAbl.setState({ harmless: e.target.value, dirty: true })} />
            </label>
            <div className="abl-probes-acts">
              <button id="abl-probes-reset" className="btn ghost sm" onClick={resetProbes}>Reset to the defaults</button>
              <button id="abl-probes-save" className="btn sm" onClick={saveProbes}>Save</button>
            </div>
          </div>
        )}

        <div id="abl-plan" className={plan.done ? "plan done" : "plan"} hidden={!plan.shown}>
          <div className="plan-h"><b className="plan-goal">{plan.goal}</b><span className="plan-clock">{plan.clock}</span></div>
          <ol className="plan-steps abl-steps">{STEPS.map((label, i) => (
            <li key={i} className={plan.steps?.[i]}><span>{label}</span><i style={plan.widths[i] ? { width: plan.widths[i] } : undefined} /></li>
          ))}</ol>
          <div className="abl-scene" id="abl-scene" hidden={!plan.scene}><canvas aria-hidden="true" /></div>
          <LiveLine className="plan-now status-line" live={plan.live} />
        </div>
        <Rank />
      </div>
    </div>
  );
}

/** The try-it thread: each question, its answer with refusals erased, and each nudge of the strength. */
function Thread() {
  const at = useAbl((s) => s.at), cmp = useSteer((s) => s.compare), cur = useApp((s) => +(s.abl?.current?.coef ?? 1) || 1);
  useAbl((s) => s.v);

  if (!at?.length) return <div id="at" className="tl" hidden />;
  const busy = at.some((e) => e.typing);

  return <div id="at" className="tl">{at.map((e, i) => <Answer key={i} e={e} i={i} last={i === at.length - 1} cmp={cmp} cur={cur} busy={busy} />)}</div>;
}

function Answer({ e, i, last, cmp, cur, busy }: { e: Entry; i: number; last: boolean; cmp: boolean; cur: number; busy: boolean }) {
  const acts = last
    ? <>{act("Compare with normal", () => setCompare(!cmp), cmp ? " on" : "")}{act("Stronger", () => nudge(1), e.refused ? " hint" : "")}
        {act("Weaker", () => nudge(-1), e.worse ? " hint" : "")}{e.change && act("Undo", undoNudge)}</>
    : e.coef != null && e.coef !== cur && !busy && act(`Go back to ${xs(e.coef)}`, () => backToCoef(e.coef!));

  // an older answer folds to two lines; a click (not on its buttons) unfolds it
  const fold = (ev: React.MouseEvent) => !(ev.target as Element).closest("button") && ((e.open = !e.open), touchAbl());

  return (
    <>
      {e.change ? <div className="tl-note">{e.change.to > e.change.from ? "Stronger" : "Weaker"} · {xs(e.change.from)} → {xs(e.change.to)}</div>
        : <div className="tl-you">{e.prompt}</div>}
      <div className={"tl-a" + (last ? " cur" : " old") + (e.open ? " open" : "")} data-i={i} onClick={last ? undefined : fold}>
        <div className="tl-a-h"><span className="tl-tag">{xs(e.coef) || "Erased"}</span>{verdict(e.refused, e.worse)}<span className="grow" />{!e.typing && acts}</div>
        <Ans s={e.text} typing={e.typing} />
        {last && (cmp || (e.typing && e.coef == null)) && <div className="tl-normal">
          <div className="tl-a-h"><span className="tl-tag">Normal</span>{verdict(e.refusedNormal)}</div>
          <Ans s={e.normal} typing={e.typing} />
        </div>}
      </div>
    </>
  );
}

function Step2() {
  const ready = useReady(), started = useAbl((s) => !!s.at);
  const q = useAbl((s) => s.q), harmful = useApp((s) => s.abl?.probes?.harmful);

  return (
    <div className={ready || started ? "step" : "step locked"} id="abl-s2">
      <div className="step-h"><span className="num">02</span><div><h3>Try it</h3>
        <p className="step-sub">Ask it something it would normally refuse. If it still refuses, make it stronger; if its writing gets worse, weaker.</p></div></div>
      <Thread />
      <form id="abl-form" className="tl-comp" autoComplete="off" onSubmit={(e) => (e.preventDefault(), ablAsk(q))}>
        <div className="composer">
          <input id="abl-q" spellCheck="false" placeholder="Ask it something" value={q} onChange={(e) => useAbl.setState({ q: e.target.value })} />
          <button type="button" className="send stop" id="abl-stop" aria-label="Stop" onClick={cancel}><i /></button>
          <button type="submit" className="send" aria-label="Send">{SEND}</button>
        </div>
        <div className="tl-sugg" id="abl-sugg">{(harmful || []).slice(-3).map((p) => (
          <button key={p} type="button" className="chip" title={p} onClick={() => ablAsk(p)}>{p}</button>
        ))}</div>
      </form>
    </div>
  );
}

function Step3() {
  const ready = useReady(), name = useAbl((s) => s.name);
  const exp = useSteer((s) => s.exp), saved = exp.saved.abl;

  return (
    <div className={ready ? "step" : "step locked"} id="abl-s3">
      <div className="step-h"><span className="num">03</span><div><h3>Save it as a new model</h3>
        <p className="step-sub">Writes a copy with the direction removed from its weights, so it answers this way anywhere, without this app.</p></div></div>
      <div className="card abl">
        <div className="save-row">
          <label className="field grow">Name
            <input id="abl-name" spellCheck="false" autoComplete="off" value={name} onChange={(e) => setName(e.target.value)} />
          </label>
          <button id="abl-bake" className="btn" disabled={exp.running} onClick={bakeAbl}>Save model</button>
          <button id="abl-export-stop" className="btn ghost" hidden={!(exp.running && exp.page === "abl")} onClick={cancelExport}>Stop</button>
        </div>
        <div id="abl-export" className="export" hidden={exp.page !== "abl"}>
          <div className="export-bar"><i style={exp.width ? { width: exp.width } : undefined} /></div>
          <div className="export-line status-line">{exp.line}</div>
        </div>
        <div id="abl-saved" className="saved-model" hidden={!saved}>
          <div><b>{saved?.name}</b><span>On this Mac</span></div>
          <button className="btn" id="abl-load" onClick={() => saved && loadModel(saved.dst)}>Load it</button>
        </div>
      </div>
    </div>
  );
}

export function Abliterate({ hidden }: { hidden: boolean }) {
  return (
    <section id="view-abliterate" className="view page" data-view="abliterate" hidden={hidden}>
      <header className="page-h">
        <h2>Abliterate</h2>
        <p>Erase the model’s refusal direction, check the result, then save it as a new model{" "}
          (<a href="https://arxiv.org/abs/2406.11717" target="_blank">Arditi et al.</a>).</p>
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
