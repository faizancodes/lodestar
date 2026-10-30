import { useEffect, useLayoutEffect, useRef } from "react";
import { dirReady, fmt, isDir, labelOf, nid } from "../lib/format";
import type { DirItem, NeuronItem } from "../lib/types";
import { useApp } from "../store/app";
import { useSteer } from "../store/steer";
import { openNeuron } from "./Found";
import { PanelSlot } from "./Panel";
import { newRecipe } from "../shell/engine";
import { patchAt, removeAt, saveRecipe, setOp } from "./recipe";

// a strength fills out from the middle: periwinkle when pushed up, red when pushed down. Painted from the range's
// own value, which the browser snaps to its step.
function usePaint() {
  const ref = useRef<HTMLInputElement>(null);
  useLayoutEffect(() => {
    const el = ref.current!, v = (+el.value - +el.min) / (+el.max - +el.min);
    el.style.setProperty("--lo", Math.min(v, 0.5) * 100 + "%");
    el.style.setProperty("--hi", Math.max(v, 0.5) * 100 + "%");
    el.style.setProperty("--fill", v < 0.5 ? "var(--bad)" : "var(--ascii)");
  });
  return ref;
}

/** On now: what's on, each with its strength. */
export function RecipeBox() {
  const items = useApp((s) => s.recipe.items);
  const gen = useSteer((s) => s.recipeGen);
  return (
    <div id="rc" className="rc" hidden={!items.length}>
      <div className="rc-h"><b>On now</b><span className="grow" /><button id="rc-save" className="btn ghost sm" onClick={saveRecipe}>Save</button><button id="rc-clear" className="btn ghost sm" onClick={newRecipe}>Turn all off</button></div>
      {/* redrawn as a whole where ui/app.js redrew it, so rows (and panels under them) are new each time */}
      <div id="rc-list" className="rc-list" key={gen}>{items.map((it, i) => isDir(it) ? <DirRow key={i} it={it} i={i} /> : <NeuronRow key={i} it={it} i={i} />)}</div>
    </div>
  );
}

function NeuronRow({ it, i }: { it: NeuronItem; i: number }) {
  const range = usePaint(), anchor = `rc:${it.layer}:${it.neuron}`;
  return (
    <>
      <div className="rc-row" data-i={i}>
        <span className="who" data-k="open" title="What this neuron does" onClick={() => openNeuron(it.layer, it.neuron, anchor)}><b>{labelOf(it)}</b><span className="nid">{nid(it.layer, it.neuron)}</span></span>
        <input ref={range} className="lr" type="range" min="-3" max="3" step="0.05" value={it.dose} data-k="dose" aria-label="Strength"
          onChange={(e) => patchAt(i, { dose: +e.target.value })} />
        <span className="dose">{fmt(it.dose)}</span>
        <button className="x" data-k="rm" aria-label="Turn off" onClick={() => removeAt(i)}>×</button>
      </div>
      <PanelSlot anchor={anchor} />
    </>
  );
}

// a direction's row: its layers (editable), push or erase, and a wider strength range than a neuron's
function DirRow({ it, i }: { it: DirItem; i: number }) {
  const range = usePaint(), band = useRef<HTMLInputElement>(null);
  const ok = useApp((s) => dirReady(it, s.dirs)), er = it.op === "erase";
  // the layers are read when the field is committed (its change event), not on every keystroke
  useEffect(() => {
    const el = band.current!;
    const commit = () => {
      const { dirLayers, recipe } = useApp.getState(), cur = recipe.items[i] as DirItem;
      const L = dirLayers || 0, ls = [...new Set((el.value.match(/\d+/g) || []).map(Number))].filter((l) => l < L).sort((a, b) => a - b);
      if (ls.length) patchAt(i, { layers: ls });
      el.value = (ls.length ? ls : cur.layers).join(" ");
    };
    el.addEventListener("change", commit);
    return () => el.removeEventListener("change", commit);
  }, [i]);
  return (
    <div className={ok ? "rc-row dir" : "rc-row dir gone"} data-i={i}>
      <span className="who static" title={ok ? String(it.name) : "Not on this model"}><b>{labelOf(it)}</b><span className="nid">direction</span></span>
      <span className="row tight"><input ref={band} className="band" data-k="band" defaultValue={it.layers.join(" ")} title="Layers" spellCheck="false" />
        <span className="seg"><button data-op="push" className={er ? undefined : "on"} onClick={() => setOp(i, "push")}>Push</button><button data-op="erase" className={er ? "on" : undefined} onClick={() => setOp(i, "erase")}>Erase</button></span></span>
      <input ref={range} className="lr" type="range" min={er ? 0 : -8} max={er ? 1 : 8} step={er ? 0.05 : 0.25} value={it.dose} data-k="dose" aria-label="Strength"
        onChange={(e) => patchAt(i, { dose: +e.target.value })} />
      <span className="dose">{fmt(it.dose, er ? 2 : 1)}</span>
      <button className="x" data-k="rm" aria-label="Turn off" onClick={() => removeAt(i)}>×</button>
    </div>
  );
}
