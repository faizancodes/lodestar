import { Fragment, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { flushSync } from "react-dom";
import { still } from "../shell/engine";
import { sleep } from "../lib/format";
import { useSteer, type Body, type Panel } from "../store/steer";
import { closeAt, closePanel, panelEls } from "./panels";

/** Goes right after an anchor: draws the panel opened under it, and drops it when the anchor goes. */
export function PanelSlot({ anchor }: { anchor: string }) {
  const p = useSteer((s) => s.panels.find((x) => x.anchor === anchor));
  useEffect(() => () => closeAt(anchor), [anchor]);
  return p ? <PanelView key={p.id} p={p} /> : null;
}

export function PanelView({ p }: { p: Panel }) {
  return (
    <div className="panel" data-kind={p.kind} data-title={p.title}
      ref={(el) => (el ? panelEls.set(p.id, el) : panelEls.delete(p.id))}>
      <div className="panel-h"><h3>{p.title}</h3><button className="x" aria-label="Close" onClick={() => closePanel(p.id)}>×</button></div>
      <RevealBox className="panel-b" body={p.body} />
    </div>
  );
}

// Swap a box's contents: the old content fades out, the box eases to its new height, and the new content rises in
// piece by piece, with a reply's words inking in left to right like the scan.
async function reveal(box: HTMLElement, swap: () => void) {
  const from = box.offsetHeight;
  if (box.firstElementChild && !still()) {
    for (const el of box.children) el.classList.add("leaving");
    await sleep(230);
  } else await null; // out of the effect that started this, so the swap can be drawn synchronously
  if (!box.isConnected) return;
  flushSync(swap);
  const $$ = (sel: string) => [...box.querySelectorAll<HTMLElement>(sel)];
  ([...box.children] as HTMLElement[]).forEach((el, i) => ((el.style.animationDelay = i * 80 + "ms"), el.classList.add("reveal")));
  $$(".tok").forEach((el, i) => ((el.style.animationDelay = 120 + Math.min(i * 6, 800) + "ms"), el.classList.add("ink")));
  $$(".found-row").forEach((el, i) => ((el.style.animationDelay = 220 + i * 45 + "ms"), el.classList.add("reveal")));
  const bars = $$(".bar i[data-w]");
  bars.forEach((el) => (el.style.width = "0"));
  requestAnimationFrame(() => bars.forEach((el) => (el.style.width = el.dataset.w!)));
  if (still() || !from) return;
  const to = box.scrollHeight;
  Object.assign(box.style, { height: from + "px", overflow: "hidden" });
  void box.offsetHeight; // commit the starting height before transitioning
  Object.assign(box.style, { transition: "height 0.42s cubic-bezier(0.2, 0.7, 0.2, 1)", height: to + "px" });
  await sleep(440);
  Object.assign(box.style, { height: "", overflow: "", transition: "" });
}

/** A box whose contents are set straight away or swapped in with the reveal transition. */
export function RevealBox({ className, body }: { className: string; body: Body }) {
  const ref = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState<{ node: ReactNode; n: number }>({ node: body.node, n: body.n });
  useLayoutEffect(() => {
    if (body.n === shown.n) return;
    const next = { node: body.node, n: body.n };
    if (!body.reveal) return setShown(next);
    reveal(ref.current!, () => setShown(next));
  }, [body]); // eslint-disable-line react-hooks/exhaustive-deps
  // keyed, so each swap draws new elements, as setting innerHTML did
  return <div className={className} ref={ref}><Fragment key={shown.n}>{shown.node}</Fragment></div>;
}
