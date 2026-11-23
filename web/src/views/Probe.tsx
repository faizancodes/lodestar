import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { ask, exportExperiment, open, openExample, openFile, refresh, remove, run, TRAVEL_MS } from "../probe/engine";
import { drawMap } from "../probe/map";
import { useApp } from "../store/app";
import { useProbe, type Metrics } from "../store/probe";

const cancel = () => api("/api/cancel", {});
const n2 = (x: number | undefined) => (x == null ? "–" : x.toFixed(2));
const setForm = (p: Partial<ReturnType<typeof useProbe.getState>["form"]>) => useProbe.setState((s) => ({ form: { ...s.form, ...p } }));
const HINT = "one per line · # group";

function Build() {
  const f = useProbe((s) => s.form), exp = useProbe((s) => s.exp), examples = useProbe((s) => s.examples);
  const live = useProbe((s) => s.live), pct = useProbe((s) => s.pct);
  const file = useRef<HTMLInputElement>(null);

  return (
    <div className="card pb-build">
      <div className="row tight">
        {examples.length > 0 && <span className="pb-lead">Experiments</span>}
        {examples.map((e) => <button key={e.file} className="chip ex-chip" title={e.about} onClick={() => openExample(e.file)}>{e.name}</button>)}
        <span className="grow" />
        <button className="btn ghost sm" onClick={() => file.current!.click()}>Open experiment</button>
        <input ref={file} type="file" accept=".json,application/json" hidden onChange={(e) => {
          const x = e.target.files?.[0];
          if (x) openFile(x);
          e.target.value = "";
        }} />
      </div>
      {exp?.about && <p className="pb-about">{exp.about}</p>}
      <label className="field">Name<input spellCheck="false" placeholder="e.g. Made up or real" value={f.name} onChange={(e) => setForm({ name: e.target.value })} /></label>
      <div className="pb-sides">
        <div className="field">
          <input className="pb-label a" spellCheck="false" placeholder="Side A, e.g. Made up" value={f.aLabel} onChange={(e) => setForm({ aLabel: e.target.value })} />
          <textarea rows={8} spellCheck="false" placeholder={HINT} value={f.a} onChange={(e) => setForm({ a: e.target.value })} />
        </div>
        <div className="field">
          <input className="pb-label b" spellCheck="false" placeholder="Side B, e.g. Real" value={f.bLabel} onChange={(e) => setForm({ bLabel: e.target.value })} />
          <textarea rows={8} spellCheck="false" placeholder={HINT} value={f.b} onChange={(e) => setForm({ b: e.target.value })} />
        </div>
      </div>
      <div className="row tight">
        <button className="btn" onClick={run}>Build probe</button>
        <button className="btn ghost" data-cancel="" onClick={cancel}>Stop</button>
        <span className="status-line">{live}</span>
      </div>
      {pct != null && <div className="progress"><i style={{ width: `${pct * 100}%` }} /></div>}
    </div>
  );
}

function Saved() {
  const list = useProbe((s) => s.list), cur = useProbe((s) => s.cur?.id);
  if (!list?.length) return null;

  return (
    <div className="pb-saved">
      <span className="pb-lead">Saved</span>
      {list.map((p) => (
        <span key={p.id} className={p.id === cur ? "chip on" : "chip"} role="button" tabIndex={0} onClick={() => open(p.id)}>
          {p.name}<button className="del" aria-label="Delete" onClick={(e) => (e.stopPropagation(), remove(p.id))}>×</button>
        </span>
      ))}
    </div>
  );
}

function StateMap() {
  const cv = useRef<HTMLCanvasElement>(null);
  const cur = useProbe((s) => s.cur)!, at = useProbe((s) => s.test.at);
  const [manual, setManual] = useState<number | null>(null);
  const [shown, setShown] = useState(cur.metrics.layer);
  const last = cur.map.length - 1;

  useEffect(() => setManual(null), [at, cur.id]);

  useEffect(() => {
    let raf = 0;
    const tick = (now: number) => {
      const { test } = useProbe.getState();
      const f = test.at ? Math.min(1, (now - test.at) / TRAVEL_MS) : 1;
      const layer = manual ?? (test.at ? cur.metrics.layer * (1 - (1 - f) ** 2) : cur.metrics.layer);
      drawMap(cv.current!, {
        map: cur.map, sides: cur.items.map((it) => it.side), labels: cur.labels, layer,
        path: test.path, hot: test.p != null && f >= 1 && manual == null ? test.p > 0.5 : null,
      }, now);
      setShown(Math.round(layer));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [cur, manual]);

  return (
    <div className="pb-map">
      <canvas ref={cv} />
      <input type="range" min={0} max={last} step={1} value={shown} aria-label="Layer"
        onChange={(e) => setManual(+e.target.value)} />
    </div>
  );
}

function Facts({ m, labels }: { m: Metrics; labels: [string, string] }) {
  const picked = m.layers_picked.filter((l) => l === m.layer).length;
  const rows: [string, string, string][] = [
    ["Tells them apart", n2(m.separation), "AUC with the layer picked inside each training fold, scored on prompts it never saw (0.5 is chance)"],
    ...(m.new_groups != null ? [["On new groups", n2(m.new_groups), "AUC on each group with the probe trained on the other groups"] as [string, string, string]] : []),
    ["From words alone", n2(m.new_groups != null ? m.words_only_new_groups : m.words_only), "The same test with a classifier that only sees the prompt's tokens"],
    ["Shuffled labels", n2(m.shuffled), "The same test with the labels shuffled"],
    [`${labels[0]} caught`, `${Math.round(m.caught * 100)}%`, `Share of ${labels[0]} scored above 90% of ${labels[1]}`],
    ["Layer", String(m.layer), `Picked by ${picked} of ${m.layers_picked.length} folds`],
  ];
  return <dl className="facts pb-facts">{rows.map(([k, v, t]) => <div key={k} title={t}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>;
}

function Checks() {
  const checks = useProbe((s) => s.checks), same = useProbe((s) => s.sameModel), model = useProbe((s) => s.exp?.model);
  if (!checks?.length) return null;

  return (
    <div className="pb-checks">
      {checks.map((c) => (
        <span key={c.metric} className={c.ok ? "pb-check ok" : "pb-check"}>
          {c.ok ? "✓" : "✗"} {c.metric.replace(/_/g, " ")} <b>{c.got ?? "–"}</b> <i>expected {c.want[0]}–{c.want[1]}</i>
        </span>
      ))}
      {same === false && <span className="pb-check note">Made on {model}, so the ranges may not apply to this model.</span>}
    </div>
  );
}

function Tester() {
  const cur = useProbe((s) => s.cur)!, t = useProbe((s) => s.test);
  const [q, setQ] = useState("");
  const [landed, setLanded] = useState(false);

  useEffect(() => {
    setLanded(false);
    if (!t.at) return;
    const id = setTimeout(() => setLanded(true), TRAVEL_MS);
    return () => clearTimeout(id);
  }, [t.at]);

  const p = landed ? t.p : null, hot = p != null && p > 0.5;
  return (
    <div className="pb-test">
      <form className="row tight" onSubmit={(e) => (e.preventDefault(), ask(q || t.q))}>
        <input className="grow" spellCheck="false" placeholder="Ask it something new" value={q} onChange={(e) => setQ(e.target.value)} />
        <button className="btn" type="submit">Ask</button>
      </form>
      {t.q && <>
        <div className="pb-q">{t.q}</div>
        <div className={p == null ? "pb-meter" : hot ? "pb-meter a" : "pb-meter b"}>
          <span>{p == null ? "Reading…" : hot ? cur.labels[0] : cur.labels[1]}</span>
          <b>{p == null ? "––" : `${Math.round((hot ? p : 1 - p) * 100)}%`}</b>
          <div className="pb-bar"><i style={{ width: `${(p ?? 0) * 100}%` }} /></div>
        </div>
        {t.answer && <div className="pb-answer">{t.answer.trim()}</div>}
      </>}
    </div>
  );
}

function Result() {
  const cur = useProbe((s) => s.cur);
  if (!cur) return null;

  return (
    <div className="panel pb-result">
      <div className="panel-h"><h3>{cur.name}</h3><button className="btn ghost sm" onClick={exportExperiment}>Save as experiment</button></div>
      <div className="panel-b">
        <Facts m={cur.metrics} labels={cur.labels} />
        <Checks />
        <StateMap />
        <Tester />
      </div>
    </div>
  );
}

export function Probe({ hidden }: { hidden: boolean }) {
  const model = useApp((s) => s.info?.model);
  useEffect(() => {
    if (!hidden) refresh();
  }, [hidden, model]);

  return (
    <section id="view-probe" className="view page" data-view="probe" hidden={hidden}>
      <header className="page-h">
        <h2>Probe</h2>
        <p>Can the model tell two kinds of prompt apart before it answers? Read its state, check the result holds up, then ask it new questions.</p>
      </header>
      <Build />
      <Saved />
      <Result />
    </section>
  );
}
