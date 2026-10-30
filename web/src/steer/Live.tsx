import { useEffect, useLayoutEffect, useRef, type HTMLAttributes } from "react";
import { createDecoder } from "../art/decode";
import { startSpinner } from "../art/glyphs";
import type { Live } from "../store/steer";
import { attach, type Sink } from "./sink";

/** A status line: empty until first set; then, while it runs, a lit strip with the Lodestar spinner, and once it
 * ends plain text. */
export function LiveLine({ id, className, live }: { id?: string; className: string; live: Live }) {
  const el = useRef<HTMLDivElement>(null), canvas = useRef<HTMLCanvasElement>(null), spin = useRef<(() => void) | null>(null);
  useEffect(() => {
    if (!live) return;
    spin.current ??= startSpinner(canvas.current!, () => !!el.current?.classList.contains("live"));
    if (live.running) spin.current();
  });
  if (!live) return <div id={id} className={className} ref={el} />;
  return (
    <div id={id} className={className + (live.running ? " live" : "")} ref={el}>
      <canvas className="spin" aria-hidden="true" ref={canvas} /><span className="pn-t">{live.text}</span>
    </div>
  );
}

/** An output box that a stream writes into (see sink.ts). */
export const Out = ({ s, className }: { s: Sink; className: string }) => <div className={className} ref={attach(s)} />;

/** Text that decodes in as it changes, like a chat reply. */
export function Decoded({ as: Tag = "div", text, ...rest }: HTMLAttributes<HTMLElement> & { as?: "div" | "p"; text: string }) {
  const ref = useRef<HTMLElement>(null), dec = useRef<ReturnType<typeof createDecoder> | null>(null);
  useLayoutEffect(() => {
    const el = ref.current!;
    if (el.textContent === text) return;
    el.textContent = text;
    (dec.current ??= createDecoder()).update(el);
  });
  return <Tag ref={ref as never} {...rest} />;
}
