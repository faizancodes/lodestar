import { useEffect, useLayoutEffect, useRef, type CSSProperties } from "react";
import {
  analyze, bakeAbl, commitCoef, resetProbes, saveProbes, setCoef, setName, testAbl, toggleAbl,
} from "../abl/engine";
import { drawHomeArt } from "../art/home";
import { api } from "../lib/api";
import { loadModel } from "../shell/engine";
import { useAbl, type Pair } from "../store/abl";
import { ablActive, useApp } from "../store/app";
import { useSteer } from "../store/steer";
import { cancelExport } from "../steer/exporting";
import { LiveLine, Out } from "../steer/Live";

const cancel = () => api("/api/cancel", {});
const n2 = { "--n": 2 } as CSSProperties;
const STEPS = ["Read", "Compare", "Try layers", "Check"];

const useReady = () => useApp((s) => !!s.abl?.ready && !!s.abl.current);

// the status panel: where the direction is, how strongly it's erased, and the switch for it
function StatusCard() {
  const panel = useRef<HTMLDivElement>(null), art = useRef<HTMLCanvasElement>(null);
  const abl = useApp((s) => s.abl), tab = useApp((s) => s.tab), loaded = useApp((s) => !!s.info?.loaded);
  const on = useApp(ablActive), ready = useReady();
  const c = abl?.ready ? abl.current : null;
  useEffect(() => { requestAnimationFrame(() => drawHomeArt(art.current!, "abliterate", panel.current)); }, [abl, tab, loaded]);
  return (
    <div className={c?.active ? "status-panel on" : "status-panel"} id="abl-status" ref={panel}>
      <canvas id="abl-status-art" aria-hidden="true" ref={art} />
      <div className="status-panel-b">
        <span className="eyebrow">Refusal direction</span>
        <dl className="facts" id="abl-facts">{c ? <>
          <div><dt>Layer</dt><dd>{c.layer}</dd></div>
          <div><dt>Strength</dt><dd>{`${(+c.coef || 1).toFixed(2)}×`}</dd></div>
          <div><dt>Status</dt><dd>{c.active ? <><i className="hc-on" />Erasing refusals</> : "Switched off"}</dd></div>
        </> : <div><dt>Status</dt><dd className="dim">Not found yet</dd></div>}</dl>
        <button id="abl-toggle" className="switch" role="switch" aria-checked={ready ? on : false} hidden={!ready}
          title={ready ? (on ? "On for every reply in Steer and Chat" : "Off: the model answers as it normally would") : undefined}
          onClick={toggleAbl}><span>Erase refusals</span><i /></button>
      </div>
    </div>
  );
}

/** Questions with their replies before and after, and whether each refused. */
function Pairs({ pairs }: { pairs: Pair[] }) {
  const verdict = (ref?: boolean) => <span className={ref === undefined ? "verdict" : ref ? "verdict ref" : "verdict ans"}>{ref === undefined ? null : ref ? "Refused" : "Answered"}</span>;
  return <>{pairs.map((p, i) => (
    <div key={i} className="run abl-pair" data-p={i}>
      <div className="run-q">{p.prompt}</div>
      <div className="cols" style={n2}>
        <div className="out" data-c="base"><div className="out-h">Before{verdict(p.verdict.base)}</div><Out s={p.outs.base} className={p.typing ? "out-b typing" : "out-b"} /></div>
        <div className="out steered" data-c="abl"><div className="out-h">After{verdict(p.verdict.abl)}</div><Out s={p.outs.abl} className={p.typing ? "out-b typing" : "out-b"} /></div>
      </div>
    </div>
  ))}</>;
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
  const plan = useAbl((s) => s.plan), examples = useAbl((s) => s.examples);
  const harmful = useAbl((s) => s.harmful), harmless = useAbl((s) => s.harmless);
  useAbl((s) => s.v);
  return (
    <div className={ready ? "step done" : "step"} id="abl-s1">
      <div className="step-h"><span className="num">01</span><div><h3>Find the refusal direction</h3>
        <p className="step-sub">Reads how the model handles bad and everyday requests, then tests which layer erases its refusals. Takes a few minutes.</p></div></div>
      <div className="card abl">
        <div className="row tight">
          <button id="abl-go" className={ready ? "btn ghost" : "btn"} disabled={plan.running} onClick={analyze}>{ready ? "Find it again" : "Find it"}</button>
          <button className="btn ghost" data-cancel="" onClick={cancel}>Stop</button>
        </div>
        <div id="abl-plan" className="plan" hidden={!plan.shown}>
          <div className="plan-h"><b className="plan-goal">{plan.goal}</b><span className="plan-clock">{plan.clock}</span></div>
          <ol className="plan-steps abl-steps">{STEPS.map((label, i) => (
            <li key={i} className={plan.steps?.[i]}><span>{label}</span><i style={plan.widths[i] ? { width: plan.widths[i] } : undefined} /></li>
          ))}</ol>
          <div className="abl-scene" id="abl-scene" hidden={!plan.scene}><canvas aria-hidden="true" /></div>
          <LiveLine className="plan-now status-line" live={plan.live} />
          <div id="abl-examples" className="abl-examples"><Pairs pairs={examples} /></div>
        </div>
        <Rank />
        <details className="adv">
          <summary>Probe questions</summary>
          <label className="field">Questions it should refuse <span className="opt">one per line</span>
            <textarea id="abl-probes-harmful" rows={6} spellCheck="false" value={harmful} onChange={(e) => useAbl.setState({ harmful: e.target.value, dirty: true })} />
          </label>
          <label className="field">Everyday questions <span className="opt">one per line</span>
            <textarea id="abl-probes-harmless" rows={5} spellCheck="false" value={harmless} onChange={(e) => useAbl.setState({ harmless: e.target.value, dirty: true })} />
          </label>
          <div className="row tight">
            <button id="abl-probes-reset" className="btn ghost sm" onClick={resetProbes}>Reset to the defaults</button>
            <button id="abl-probes-save" className="btn ghost sm" onClick={saveProbes}>Save probe lists</button>
          </div>
        </details>
      </div>
    </div>
  );
}

function Step2() {
  const ready = useReady();
  const coef = useAbl((s) => s.coef), probes = useAbl((s) => s.testProbes), test = useAbl((s) => s.test);
  useAbl((s) => s.v);
  const range = useRef<HTMLInputElement>(null);
  // the slider's fill up to the handle
  useLayoutEffect(() => {
    const el = range.current!;
    el.style.setProperty("--pct", ((+el.value - +el.min) / (+el.max - +el.min)) * 100 + "%");
  }, [coef]);
  // the strength is sent once it's let go of (its change event), not while it's dragged
  useEffect(() => {
    const el = range.current!;
    el.addEventListener("change", commitCoef);
    return () => el.removeEventListener("change", commitCoef);
  }, []);
  const sum = test.summary;
  return (
    <div className={ready ? "step" : "step locked"} id="abl-s2">
      <div className="step-h"><span className="num">02</span><div><h3>Check it</h3>
        <p className="step-sub">Ask a few bad requests and compare its answers with refusals erased. Raise the strength if it still refuses.</p></div></div>
      <div className="card abl">
        <div className="abl-checks">
          <div className="field">Strength
            <div className="abl-range">
              <div className="abl-range-track">
                <input ref={range} id="abl-coef" type="range" value={coef} min="0.2" max="2" step="0.05" aria-label="Strength" onChange={(e) => setCoef(e.target.value)} />
                <div className="abl-ticks" aria-hidden="true"><span style={{ left: "0%" }}>0.2</span><span style={{ left: "44.44%" }}>1 · full</span><span style={{ left: "100%" }}>2</span></div>
              </div>
              <output id="abl-coef-v">{(+coef).toFixed(2) + "×"}</output>
            </div>
          </div>
          <label className="field">Test questions <span className="opt">one per line, up to four</span>
            <textarea id="abl-test-probes" rows={3} spellCheck="false" value={probes} onChange={(e) => useAbl.setState({ testProbes: e.target.value })} />
          </label>
          <div className="row tight">
            <button id="abl-test-go" className="btn" onClick={testAbl}>Compare before and after</button>
            <button className="btn ghost" data-cancel="" onClick={cancel}>Stop</button>
          </div>
          <div id="abl-test-summary" className="status-line">{sum && <><b>Refused {sum.rb} of {sum.n} before, {sum.ra} of {sum.n} after.</b>{sum.note}</>}</div>
          <div id="abl-test-rows"><Pairs pairs={test.pairs} /></div>
        </div>
      </div>
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
