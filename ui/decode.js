// Text that decodes as it streams in, in Lodestar's marks: each new character arrives as a periwinkle dot, grows
// into a plus, then an ×, then resolves into its letter, which fades from periwinkle to the text colour. The letter
// holds its place from the first frame (the mark is drawn over it), so lines never reflow as marks resolve.
//
// The host re-renders the whole reply from markdown as tokens arrive, so the state lives here, keyed by visible
// character index (when each one arrived), and is re-applied to whatever the latest render is.
const DUR = 340;                 // ms from first mark to settled letter
const MARKS = [[0.22, "·"], [0.46, "+"], [0.68, "×"]];

export function createDecoder() {
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let arrivals = [], el = null, raf = 0;

  const textNodes = (root) => {
    const out = [], walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => (n.parentElement.closest(".dots") ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
    });
    for (let n = walk.nextNode(); n; n = walk.nextNode()) out.push(n);
    return out;
  };

  // wrap every character still decoding in its own span, working back from the end (new text is at the end)
  const wrap = (now) => {
    const nodes = textNodes(el);
    let i = nodes.reduce((a, n) => a + n.data.length, 0);
    for (let k = nodes.length - 1; k >= 0; k--) {
      const node = nodes[k], text = node.data;
      let cut = text.length;
      while (cut > 0 && now - (arrivals[i - (text.length - cut) - 1] ?? 0) < DUR) cut--;
      i -= text.length;
      if (cut === text.length) break; // everything earlier has settled
      const frag = document.createDocumentFragment();
      if (cut) frag.append(text.slice(0, cut));
      for (let j = cut; j < text.length; j++) {
        const c = text[j];
        if (/\s/.test(c)) { frag.append(c); continue; }
        const s = document.createElement("span");
        s.className = "dc";
        s.dataset.i = i + j;
        s.textContent = c;
        frag.append(s);
      }
      node.replaceWith(frag);
    }
  };

  const paint = () => {
    raf = 0;
    if (!el || !el.isConnected) return;
    const now = performance.now();
    let live = 0;
    for (const s of el.querySelectorAll(".dc")) {
      const p = (now - arrivals[+s.dataset.i]) / DUR;
      if (p >= 1) { s.replaceWith(s.textContent); continue; }
      live++;
      const mark = p < 0 ? "" : MARKS.find(([upto]) => p < upto)?.[1];
      if (mark !== undefined) {
        s.className = p < 0 ? "dc w" : "dc";
        s.dataset.g = mark;
        s.style.color = "";
      } else { // the letter itself, cooling from periwinkle to the text colour
        s.className = "dc r";
        s.style.color = `color-mix(in srgb, var(--ascii) ${Math.round((1 - (p - 0.68) / 0.32) * 100)}%, var(--text))`;
      }
    }
    if (live) raf = requestAnimationFrame(paint);
  };

  return {
    /** Call right after `root` was (re-)rendered with the reply so far. */
    update(root) {
      el = root;
      if (still || !root) return;
      const now = performance.now();
      const n = textNodes(root).reduce((a, t) => a + t.data.length, 0);
      if (n < arrivals.length) arrivals.length = n; // markdown can swallow a few characters (** becoming bold)
      const fresh = n - arrivals.length, gap = fresh ? Math.min(14, 110 / fresh) : 0;
      for (let k = 0; k < fresh; k++) arrivals.push(now + k * gap); // a token's letters arrive in a quick ripple
      wrap(now);
      if (!raf) raf = requestAnimationFrame(paint);
      paint();
    },
  };
}
