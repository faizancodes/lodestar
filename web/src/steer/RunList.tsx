import { useEffect, useRef, type CSSProperties } from "react";
import { touch, useSteer, type Fix, type Run } from "../store/steer";
import { Out, LiveLine } from "./Live";
import { PanelSlot } from "./Panel";
import { FIX_CHIPS, NAMES, openFix, runFix } from "./runs";

export function Runs() {
  const runs = useSteer((s) => s.runs);
  useSteer((s) => s.v);
  return <div id="runs" className="runs">{runs.map((r) => <RunView key={r.id} run={r} />)}</div>;
}

function RunView({ run }: { run: Run }) {
  const panel = useSteer((s) => s.panels.find((p) => p.anchor === `run:${run.id}`));
  const up = run.outs.up;
  const fix = run.fix && <FixBox key="fix" run={run} fix={run.fix} />, slot = <PanelSlot key="panel" anchor={`run:${run.id}`} />;
  // the fix box and an inspect panel each open right after the replies, so the newer one comes first
  const after = run.fix && panel && panel.at > run.fixAt ? [slot, fix] : [fix, slot];
  return (
    <div className="run">
      <div className="run-q">{run.prompt}<span>{run.desc}</span></div>
      <div className="cols" style={{ "--n": run.cols.length } as CSSProperties}>{run.cols.map((c) => (
        <div key={c} className={`out ${c === "up" || c === "abl" ? "steered" : ""}`} data-c={c}>
          <div className="out-h">{NAMES[c]}<button className="insp" hidden={!run.done || !run.outs[c].text} onClick={() => run.inspect(c)}>Inspect</button>
            {c === "up" && <button className="insp fix-b" hidden={!run.done || !up.text} onClick={() => openFix(run)}>Fix this</button>}</div>
          <Out s={run.outs[c]} className={run.done ? "out-b" : "out-b typing"} />
        </div>
      ))}</div>
      {after}
    </div>
  );
}

function FixBox({ run, fix }: { run: Run; fix: Fix }) {
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => input.current!.focus(), [fix.focus]);
  const go = () => runFix(run, fix);
  return (
    <div className="fix" hidden={!fix.open}>
      <b className="fix-h">What’s wrong with it?</b>
      <div className="ask"><input ref={input} spellCheck="false" autoComplete="off" placeholder="e.g. It repeats itself. It should sound like a devout Muslim, not quote the Bible."
        value={fix.input} onChange={(e) => ((fix.input = e.target.value), touch())} onKeyDown={(e) => e.key === "Enter" && go()} />
        <button className="btn" disabled={fix.busy} onClick={go}>Fix</button></div>
      <div className="chips">{FIX_CHIPS.map((c) => (
        <button key={c} className="chip" onClick={() => {
          const v = fix.input.trim();
          fix.input = v ? `${v.replace(/[.\s]*$/, "")}. ${c}.` : `${c}.`;
          fix.focus++;
          touch();
        }}>{c}</button>
      ))}</div>
      <LiveLine className="plan-now status-line fix-now" live={fix.live} />
      <div className="fix-aim" hidden={!fix.aim}><span className="eyebrow">Aiming for</span>{fix.aim ? <Out s={fix.aim} className="fix-aim-b" /> : <div className="fix-aim-b" />}</div>
      <div className="fix-done" hidden={!fix.done}>{fix.done === "undone" ? <span className="status-line">Undone: the neurons are back as they were.</span>
        : fix.done && <><ul>{fix.done.lines.map((l, i) => <li key={i}>{l}</li>)}</ul><button className="btn ghost sm" onClick={fix.done.undo}>Undo</button></>}</div>
    </div>
  );
}
