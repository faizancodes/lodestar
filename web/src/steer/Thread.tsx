import { Fragment } from "react";
import { touch, useSteer, type Version } from "../store/steer";
import { Ans, LiveLine } from "./Live";
import { PanelSlot } from "./Panel";
import { goBack, inspectAnswer, nOn, setCompare, undo } from "./refine";

const act = (label: string, onClick: () => void, on = false) => <button className={on ? "tl-act on" : "tl-act"} onClick={onClick}>{label}</button>;

/** Try it and refine: the question, its answers in versions, and the fix being made. */
export function Thread() {
  const tl = useSteer((s) => s.tl);
  useSteer((s) => s.v);
  if (!tl) return <div id="tl" className="tl" hidden />;
  const main = tl.streaming.has("main");
  return (
    <div id="tl" className="tl">
      <div className="tl-you">{tl.prompt}</div>
      {!tl.versions.length && <>
        <div className="tl-a cur">
          <div className="tl-a-h"><span className="tl-tag">{tl.abl ? "Refusals erased" : "Normal"}</span><span className="grow" />
            {!main && act("Inspect", () => inspectAnswer(null))}</div>
          <Ans s={tl.other} typing={main} />
        </div>
        <PanelSlot anchor={`tl:${tl.id}:x`} />
      </>}
      {tl.versions.map((v, i) => <Answer key={i} v={v} i={i} last={i === tl.versions.length - 1} />)}
      {tl.fixing && <><div className="tl-you">{tl.fixing}</div><LiveLine className="tl-now status-line plan-now" live={tl.fixLive} /></>}
    </div>
  );
}

function Answer({ v, i, last }: { v: Version; i: number; last: boolean }) {
  const tl = useSteer((s) => s.tl)!, cmp = useSteer((s) => s.compare);
  const typing = last && tl.streaming.has("main");
  // an older answer folds to two lines; a click (not on its buttons or a panel under it) unfolds it
  const fold = (e: React.MouseEvent) => !(e.target as Element).closest("button, .panel") && ((v.open = !v.open), touch());
  return (
    <>
      {v.fix ? <>
        <div className="tl-you">{v.fix.feedback}</div>
        {!!v.fix.lines.length && <div className="tl-chg">{v.fix.lines.map((l, k) => <Fragment key={k}>{k ? " · " : ""}{l}</Fragment>)}</div>}
      </> : v.note && <div className="tl-note">{v.note}</div>}
      <div className={"tl-a" + (last ? " cur" : " old") + (v.open ? " open" : "")} data-i={i} onClick={last ? undefined : fold}>
        <div className="tl-a-h"><span className="tl-tag">v{i + 1}</span><span className="tl-n">{nOn(v.items)}</span><span className="grow" />
          {!typing && (last
            ? <>{act("Compare with normal", () => setCompare(!cmp), cmp)}{act("Inspect", () => inspectAnswer(i))}{i > 0 && act("Undo", undo)}</>
            : <>{act("Inspect", () => inspectAnswer(i))}{act("Go back to this", () => goBack(i))}</>)}</div>
        <Ans s={v.out} typing={typing} />
        {last && cmp && <div className="tl-normal"><div className="tl-a-h"><span className="tl-tag">Normal</span></div><Ans s={tl.normal} typing={tl.streaming.has("normal")} /></div>}
      </div>
      <PanelSlot anchor={`tl:${tl.id}:${i}`} />
    </>
  );
}
