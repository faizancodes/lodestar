import { createDecoder } from "../art/decode";

// Text streamed into an output box (before and after, normal and steered) decodes in mark by mark. The decoder
// rewrites the box's text nodes, so React renders the box empty and never touches its children: the text is
// appended here, straight into the element, as each delta arrives.
export type Sink = { text: string; el: HTMLElement | null; dec: ReturnType<typeof createDecoder> | null };

export const sink = (): Sink => ({ text: "", el: null, dec: null });

export function streamInto(s: Sink, delta: string) {
  s.text += delta;
  if (!s.el) return;
  s.el.textContent += delta;
  (s.dec ??= createDecoder()).update(s.el);
}

const refs = new WeakMap<Sink, (el: HTMLElement | null) => void>();

/** A ref callback that attaches a box to its sink, filling in what arrived before it was drawn. */
export function attach(s: Sink) {
  let ref = refs.get(s);
  if (!ref) refs.set(s, (ref = (el) => {
    s.el = el;
    if (el && el.textContent !== s.text) el.textContent = s.text;
  }));
  return ref;
}
