import type { CSSProperties } from "react";
import { markSVG } from "../art/mark";
import { Html } from "../components/Html";
import { fmt, lensLabel, nid } from "../lib/format";
import { useApp } from "../store/app";
import { useSteer, type Cand, type Plan as P } from "../store/steer";
import { findChanges, planSummary, renderPlan, setInput, STAGES } from "./describe";
import { Decoded, LiveLine } from "./Live";
import { PanelSlot } from "./Panel";
import { openNeuron } from "./Found";
import { addItems, replaceRecipe } from "./recipe";
import { PROMPTS, send, setQuestion } from "./refine";

const MARK = markSVG(30, 15);
const IDEAS = ["talk like a pirate", "be extremely cheerful", "answer in rhyme", "cooking"];

// The live steering plan: its stages, how the model read the instruction, the neurons found so far, a check on a
// question it hasn't seen, and the button that turns them on.
export function Plan({ plan }: { plan: P }) {
  useSteer((s) => s.v);
  const live = plan.noEffect
    ? { text: <><b>This didn’t change how the model answers</b>, so there are no neurons to tune for it. With your instruction, the model still gives its ordinary reply (below). Steering can shift the model’s style or topic, not add a skill or make it answer things it declines.</>, running: false }
    : { text: plan.stopped || (plan.done ? planSummary(plan) : (plan.label || "") + "…"), running: !plan.stopped && !plan.done };
  return (
    <div className="plan">
      <div className="plan-h"><span className="eyebrow">Steering plan</span><b>{plan.goal}</b><span className="plan-clock">{plan.clock}</span></div>
      <ol className="plan-steps">{plan.noEffect ? null : STAGES.map((st, i) => (
        <li key={`${plan.renders}:${i}`} className={i < plan.stage || plan.done ? "done" : i === plan.stage && !plan.stopped ? "on" : ""}>
          <span>{st.label}</span><i style={{ width: `${i === plan.stage ? Math.round(plan.frac * 100) : 0}%` }} /></li>
      ))}</ol>
      <LiveLine className="plan-now status-line" live={live} />
      <div className="plan-ex" hidden={!plan.example}>
        <div className="px-h">{plan.ordinary ? <span className="eyebrow">The model’s ordinary answer</span>
          : <><span className="eyebrow">How it reads your instruction</span><span className="px-note">The model, told to act that way</span></>}</div>
        <div className="px-turn"><span className="px-who">Asked</span><p className="q">{plan.examplePrompt}</p></div>
        {/* the example decodes in as it's written, like chat replies */}
        <div className="px-turn"><span className="px-who">Answers</span><Decoded as="p" className="a" text={plan.example.trim().slice(0, 320) + (plan.example.length > 320 ? "…" : "")} /></div>
      </div>
      <div className="plan-tbl">{plan.noEffect ? (
        <div className="plan-ideas"><span>Try a style or topic:</span>{IDEAS.map((g) => (
          <button key={g} className="chip ex-chip" data-g={g} onClick={() => (setInput(g), findChanges())}>{g}</button>
        ))}</div>
      ) : <Table plan={plan} />}</div>
      <div className="plan-proof" hidden={plan.noEffect || !plan.proof.prompt}>{plan.proof.prompt && <Proof plan={plan} />}</div>
      <div className="plan-foot">{plan.noEffect ? plan.resume && <button className="btn ghost sm plan-force" onClick={() => plan.resume!()}>Search anyway</button> : <Foot plan={plan} />}</div>
    </div>
  );
}

const key = (c: { layer: number; neuron: number }) => `${c.layer}:${c.neuron}`;

// chosen neurons first, then the strongest candidates (tested beat estimates)
function ranked(plan: P) {
  const chosen = new Set(plan.recipe.map(key));
  const rows = [...plan.cands.values()];
  const rank = (c: Cand) => (chosen.has(key(c)) ? 1e6 : 0) + (c.tested ? 1e3 + c.dj * 100 : Math.abs(c.est) * 1e-3);
  rows.sort((a, b) => rank(b) - rank(a));
  return { rows, chosen };
}

function Table({ plan }: { plan: P }) {
  const { rows, chosen } = ranked(plan);
  const shown = plan.done ? rows.filter((c) => chosen.has(key(c))) : rows.slice(0, 8);
  if (!shown.length) return null;
  let maxDj = 1e-6, maxEst = 1e-9;
  for (const c of rows) {
    if (c.tested) maxDj = Math.max(maxDj, c.dj);
    maxEst = Math.max(maxEst, Math.abs(c.est || 0));
  }
  const inRecipe = new Map(plan.recipe.map((x) => [key(x), x]));
  return (
    <>
      <div className="pt-head"><span>Neuron</span><span>Pushes the model toward</span><span>Tune to</span><span>Effect</span></div>
      {shown.map((c) => {
        const r = inRecipe.get(key(c));
        const dose = r ? r.dose : c.dose ?? c.sign * 0.5;
        const state = r ? "chosen" : c.tested ? "tested" : "estimate";
        const w = c.tested ? Math.max(0, c.dj) / maxDj : Math.abs(c.est) / maxEst;
        const hurts = (r?.dfluency ?? c.dfluency ?? 0) > 0.35;
        const anchor = `pt:${key(c)}`;
        // a neuron's details stay open across live updates, under its row
        return [
          <div key={key(c)} className={`pt-row ${state}`} data-l={c.layer} data-n={c.neuron} onClick={() => openNeuron(c.layer, c.neuron, anchor)}>
            <span className="nid">{nid(c.layer, c.neuron)}</span>
            <span className="pt-lens">{lensLabel(c.lens) || "—"}</span>
            <span className={`pt-dose ${dose < 0 ? "down" : "up"}`}>{dose < 0 ? "↓ down" : "↑ up"} <b>{fmt(dose)}</b></span>
            <span className="pt-eff"><span className={`pt-bar ${c.tested ? "" : "est"}`}><i style={{ width: `${Math.round(w * 100)}%` }} /></span><span className="pt-tag">{state}{hurts ? " · rougher text" : ""}</span></span>
          </div>,
          <PanelSlot key={anchor} anchor={anchor} />,
        ];
      })}
    </>
  );
}

function Proof({ plan }: { plan: P }) {
  return (
    <>
      <div className="lbl">On a question it hasn’t seen: “{plan.proof.prompt}”</div>
      <div className="cols" style={{ "--n": 2 } as CSSProperties}>
        <div className="out"><div className="out-h">Normal</div><Decoded className="out-b" data-k="baseline" text={(plan.proof.baseline || "").trim()} /></div>
        <div className="out steered"><div className="out-h">Tuned</div><Decoded className="out-b" data-k="steered" text={(plan.proof.steered || "").trim()} /></div>
      </div>
    </>
  );
}

// apply: usable as soon as anything is tested, even mid-search: the combination once there is one, before that
// the single best tested neuron
function Foot({ plan }: { plan: P }) {
  let bestTested: Cand | undefined;
  for (const c of plan.cands.values()) if (c.tested && c.dj > 0 && (!bestTested || c.dj > bestTested.dj)) bestTested = c;
  const use = plan.recipe.length ? plan.recipe : bestTested && !plan.done ? [{ ...bestTested, dose: bestTested.dose! }] : [];
  const n = use.length;
  if (!n) return null;
  const nn = `${n} neuron${n > 1 ? "s" : ""}`;
  const apply = () => {
    if (plan.done) ((plan.applied = true), renderPlan(plan));
    replaceRecipe({ id: null, name: plan.goal.slice(0, 40), items: [], model: useApp.getState().info!.model });
    addItems(use.map((x) => ({ layer: x.layer, neuron: x.neuron, dose: x.dose, lens: x.lens })), true);
    const q = plan.proof.prompt || PROMPTS[0];
    // the search holds the model while it runs, so a reply can only be generated once it has finished
    if (!plan.done) return (setQuestion(q), useApp.getState().toastMsg("Turned on. Send a question once the search finishes, or press Stop."));
    send(q);
    requestAnimationFrame(() => document.querySelector("#view-steer .step:nth-of-type(2)")?.scrollIntoView({ behavior: "smooth", block: "start" }));
  };
  if (!plan.done) return <button className="btn ghost plan-apply" onClick={apply}>Turn on the best so far · {nn}</button>;
  return (
    <div className={plan.applied ? "plan-done applied" : "plan-done"}><Html className="pd-mark" html={MARK} />
      <div className="pd-b"><b>{plan.applied ? `${nn} on` : `${nn} found`}</b>
        <span>{plan.applied ? "Steering every reply in Steer and Chat. Fine-tune each one under On now." : "Turn them on to steer every reply in Steer and Chat. You can fine-tune each one afterwards."}</span></div>
      {plan.applied ? <span className="pd-on"><i className="hc-on" />On</span> : <button className="btn btn-lg plan-apply" onClick={apply}>Turn on {nn}</button>}</div>
  );
}
