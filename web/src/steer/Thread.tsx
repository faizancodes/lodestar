import { Fragment } from "react";
import { isDir } from "../lib/format";
import { useSteer, type Thread as Conversation, type Turn } from "../store/steer";
import { Ans, LiveLine } from "./Live";
import { PanelSlot } from "./Panel";
import { FIX_CHIPS, fixLatest, inspectAnswer, nOn, retry, setCompare, showVersion, shownOf, toggleOther } from "./refine";

const act = (label: string, onClick: () => void, on = false) => <button className={on ? "tl-act on" : "tl-act"} onClick={onClick}>{label}</button>;

/** Try it and refine: the conversation so far, each question with the reply that's shown for it. */
export function Thread() {
  const tl = useSteer((s) => s.tl);
  useSteer((s) => s.v);

  if (!tl?.turns.length) return <div id="tl" className="tl" hidden />;

  return (
    <div id="tl" className="tl">
      {tl.turns.map((t, i) => <TurnView key={i} tl={tl} t={t} i={i} last={i === tl.turns.length - 1} />)}
    </div>
  );
}

function TurnView({ tl, t, i, last }: { tl: Conversation; t: Turn; i: number; last: boolean }) {
  const cmp = useSteer((s) => s.compare);
  const v = shownOf(t), n = t.versions.length;
  const typing = last && tl.streaming.has("main"), idle = last && !tl.streaming.size && !tl.fixing;
  const tag = v.items.length ? <span className="tl-n">{nOn(v.items)}</span> : <span className="tl-tag">{t.abl ? "Refusals erased" : "Normal"}</span>;
  const fixable = v.items.some((it) => !isDir(it));

  return (
    <>
      <div className="tl-you">{t.prompt}</div>
      <div className={last ? "tl-a cur" : "tl-a"} data-i={i}>
        <div className="tl-a-h">
          {tag}
          {n > 1 && <span className="tl-ver" aria-label="Versions of this reply">
            {last && <button aria-label="Previous version" disabled={!idle || t.shown === 0} onClick={() => showVersion(t.shown - 1)}>‹</button>}
            <span>{t.shown + 1} of {n}</span>
            {last && <button aria-label="Next version" disabled={!idle || t.shown === n - 1} onClick={() => showVersion(t.shown + 1)}>›</button>}
          </span>}
        </div>
        {v.fix ? <div className="tl-chg"><span>{v.fix.feedback}</span>{v.fix.lines.map((l, k) => <Fragment key={k}> · {l}</Fragment>)}</div>
          : v.note && <div className="tl-chg"><span>{v.note}</span></div>}
        <Ans s={v.out} typing={typing} />
        {idle && <>
          <div className="tl-a-f">
            {v.items.length > 0 && act("Compare with normal", () => setCompare(!cmp), cmp)}
            {act("Inspect", inspectAnswer)}
            {act("Try again", retry)}
          </div>
          {fixable && <Fix open={tl.other} />}
        </>}
        {last && tl.fixing && <div className="tl-fixing"><div className="tl-chg"><span>{tl.fixing}</span></div>
          <LiveLine className="tl-now status-line plan-now" live={tl.fixLive} /></div>}
        {last && cmp && v.items.length > 0 && <div className="tl-normal">
          <div className="tl-a-h"><span className="tl-tag">Normal</span></div>
          <Ans s={t.normal} typing={tl.streaming.has("normal")} />
        </div>}
      </div>
      {last && <PanelSlot anchor={`tl:${tl.id}:${i}`} />}
    </>
  );
}

// what's wrong with the latest reply: a common complaint, or anything in words
function Fix({ open }: { open: boolean }) {
  const text = useSteer((s) => s.fixInput);

  return (
    <div className="tl-fix" id="tl-fix">
      <span className="tl-fix-l">Not right?</span>
      {FIX_CHIPS.map((c) => <button key={c} className="chip" onClick={() => fixLatest(c + ".")}>{c}</button>)}
      <button className={open ? "chip on" : "chip"} aria-expanded={open} onClick={toggleOther}>Something else</button>
      {open && <form className="tl-fix-other" onSubmit={(e) => (e.preventDefault(), fixLatest(text))}>
        <input id="tl-fix-input" placeholder="What should change?" spellCheck="false" autoComplete="off" value={text}
          onChange={(e) => useSteer.setState({ fixInput: e.target.value })} />
        <button className="btn sm" disabled={!text.trim()}>Fix it</button>
      </form>}
    </div>
  );
}
