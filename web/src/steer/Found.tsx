import { useEffect, useRef, useState } from "react";
import { startScan } from "../art/scan";
import { api } from "../lib/api";
import { isDir, labelOf, lensLabel, nid, showTok } from "../lib/format";
import { stream } from "../lib/stream";
import type { NeuronDetail, NeuronItem, NeuronRow, Push, TokenDetail } from "../lib/types";
import { useApp } from "../store/app";
import type { Body } from "../store/steer";
import { PanelSlot, RevealBox } from "./Panel";
import { alive, closeWhere, openPanel, setBody } from "./panels";
import { addItems, redrawn } from "./recipe";
import { computeJlens, jlensLeft, stopJlens, watchJlens } from "../shell/engine";

const recipeHas = (l: number, n: number) => (s: ReturnType<typeof useApp.getState>) =>
  s.recipe.items.some((i) => !isDir(i) && i.layer === l && i.neuron === n);

/** A neuron row with a Turn on button, used by By word and Inspect; `scope` tells apart rows of the same neuron. */
export function FoundRow({ x, scope }: { x: NeuronRow & { dose: number }; scope: string }) {
  const on = useApp(recipeHas(x.layer, x.neuron));
  const anchor = `found:${scope}:${x.layer}:${x.neuron}`;

  return (
    <>
      <div className={on ? "found-row in" : "found-row"} data-l={x.layer} data-n={x.neuron} data-d={x.dose}>
        <span className="who" title="What this neuron does" onClick={() => openNeuron(x.layer, x.neuron, anchor)}>
          <b>{lensLabel(x.lens) || nid(x.layer, x.neuron)}</b><span className="nid">{nid(x.layer, x.neuron)}</span></span>
        <button className="btn ghost sm" onClick={() => addItems([{ layer: x.layer, neuron: x.neuron, dose: x.dose, lens: x.lens }])}>
          {on ? "On" : "Turn on"}</button>
      </div>
      <PanelSlot anchor={anchor} />
    </>
  );
}

// ------------------------------------------------------------------ a neuron
export async function openNeuron(l: number, n: number, anchor: string) {
  const it = () => useApp.getState().recipe.items.find((i): i is NeuronItem => !isDir(i) && i.layer === l && i.neuron === n);
  const cur = it();
  const title = cur ? `${labelOf(cur)} · ${nid(l, n)}` : nid(l, n);

  const id = openPanel(anchor, title, "neuron");
  if (id === null) return;
  setBody(id, <div className="empty">Loading…</div>);

  let d: NeuronDetail;
  try { d = await api<NeuronDetail>(`/api/neuron?layer=${l}&n=${n}`); } catch (e) { return setBody(id, <div className="empty">{(e as Error).message}</div>); }
  setBody(id, <NeuronBody d={d} on={!!it()} />, true);
}

function NeuronBody({ d, on }: { d: NeuronDetail; on: boolean }) {
  const [label, setLabel] = useState(on ? "Turn off" : "Turn on");
  const inspected = useApp.getState().inspect;

  let fires = null;
  if (d.trace && inspected) {
    const a = d.trace.acts, s = d.trace.n_prompt;
    let mx = 1e-6;
    for (let i = s; i < a.length; i++) mx = Math.max(mx, Math.abs(a[i]));

    fires = (
      <>
        <h4>In the reply you inspected</h4>
        <div className="tokens">{d.trace.tokens.map((t, i) => {
          if (i < s) return null;
          const v = Math.min(1, Math.abs(a[i]) / mx);
          const bg = a[i] >= 0 ? `rgba(195,207,245,${(v * 0.85).toFixed(2)})` : `rgba(242,119,122,${(v * 0.75).toFixed(2)})`;
          return <span key={i} className="tok" style={{ background: bg, color: v > 0.55 ? "#0a0a0a" : undefined }}>{t.replace(/\n/g, "")}</span>;
        })}</div>
      </>
    );
  }

  const toggle = () => {
    const app = useApp.getState(), has = (i: (typeof app.recipe.items)[number]) => !isDir(i) && i.layer === d.layer && i.neuron === d.neuron;
    const cur = app.recipe.items.find(has);
    const items = cur ? app.recipe.items.filter((i) => i !== cur) : [...app.recipe.items, { layer: d.layer, neuron: d.neuron, dose: 0.8, lens: d.lens.up.slice(0, 6) }];

    useApp.setState({ recipe: { ...app.recipe, items } });
    setLabel(items.some(has) ? "Turn off" : "Turn on");
    redrawn();
  };

  return (
    <>
      <h4>Turned up, it makes the model say</h4>
      <div className="pills">{d.lens.up.slice(0, 12).map((t, i) => <span key={i} className="pill up">{showTok(t)}</span>)}</div>
      <div className="lbl neg">Turned down</div>
      <div className="pills">{d.lens.down.slice(0, 12).map((t, i) => <span key={i} className="pill down">{showTok(t)}</span>)}</div>
      {fires}
      <div className="row"><button className="btn sm nd-add" onClick={toggle}>{label}</button></div>
    </>
  );
}

// ------------------------------------------------------------------ inspect a reply
function Scan() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => startScan(ref.current!), []);
  return <div className="scan"><canvas aria-hidden="true" ref={ref} /><div className="scan-cap">Reading every neuron…</div></div>;
}

type Trace = { tokens: string[]; n_prompt: number };

/** Opens the inside of a reply under it (`anchor` names the reply). */
export async function inspect(anchor: string, prompt: string, reply: string, ps: Push[], which: string, abl: boolean) {
  // the server keeps one trace at a time, so only one reply can be inspected at once
  closeWhere((p) => p.kind !== "inspect" || p.anchor === anchor);

  const id = openPanel(anchor, `Inside the ${which.toLowerCase()} reply`, "inspect");
  if (id === null) return;
  setBody(id, <Scan />);

  let tr: Trace | null = null;
  await stream("/api/trace", { prompt, response: reply, pushes: ps, abl }, (ev) => { if (ev.type === "trace") tr = ev as unknown as Trace; });

  if (!tr || !alive(id)) return;
  const t: Trace = tr;
  useApp.setState({ inspect: { tokens: t.tokens, n_prompt: t.n_prompt } });

  // fetch the first word's details before swapping, so everything arrives in one transition
  const first = await wordDetail(t.n_prompt);
  setBody(id, <InspectBody tr={t} first={first} />, true);
}

function InspectBody({ tr, first }: { tr: Trace; first: JSX.Element }) {
  const [sel, setSel] = useState(tr.n_prompt);
  const [detail, setDetail] = useState<Body>({ node: first, n: 0, reveal: false });

  const pick = async (t: number) => {
    setSel(t);
    const node = await wordDetail(t);
    setDetail((d) => ({ node, n: d.n + 1, reveal: true }));
  };

  return (
    <>
      <h4>Click a word</h4>
      <div className="tokens in-words">{tr.tokens.map((t, i) => i < tr.n_prompt ? null : (
        <span key={i} className={"tok" + (t.includes("\n") ? " nl" : "") + (i === sel ? " sel" : "")} data-t={i} onClick={() => pick(i)}>{t.replace(/\n/g, "")}</span>
      ))}</div>
      <RevealBox className="in-detail" body={detail} />
    </>
  );
}

async function wordDetail(t: number) {
  const d = await api<TokenDetail>(`/api/trace/token?t=${t}`);
  const rows = d.neurons.slice(0, 12).map((x) => ({ ...x, dose: ((x.act as number) >= 0 ? 1 : -1) * 0.8 }));

  return (
    <div className="in-cols">
      <div><h4>Neurons most active here</h4><div className="found">{rows.map((x, i) => <FoundRow key={i} x={x} scope="in" />)}</div></div>
      <LayerLens t={t} lens={d.lens} jlens={d.jlens} />
    </div>
  );
}

let lensMode: "logit" | "j" | null = null; // the reader's last pick, kept from word to word

/** Each layer read through the logit lens (what it expects next) or the J-lens (what it holds ready to say). */
function LayerLens({ t, lens, jlens: j0 }: { t: number; lens: TokenDetail["lens"]; jlens: TokenDetail["jlens"] }) {
  const [jlens, setJ] = useState(j0);
  const [mode, setMode] = useState<"logit" | "j">(lensMode ?? (j0 ? "j" : "logit"));
  const [moved, setMoved] = useState(false); // after a switch, bars take their width at once
  const pick = (m: "logit" | "j") => { lensMode = m; setMode(m); setMoved(true); };
  const rows = mode === "j" ? jlens : lens;
  const every = lens.length > 24 ? 2 : 1; // keep the layer table short on deep models

  return (
    <div>
      <div className="lens-h">
        <h4>{mode === "j" ? "What each layer holds" : "What each layer expects next"}</h4>
        <div className="seg"><button className={mode === "logit" ? "on" : ""} onClick={() => pick("logit")}>Logit lens</button>
          <button className={mode === "j" ? "on" : ""} onClick={() => pick("j")}>J-lens</button></div>
      </div>
      {rows ? <table className="guess"><tbody>{rows.map((row, l) => l % every && l !== rows.length - 1 ? null : (
        <tr key={l}><td className="l">{l}</td><td className="t">{showTok(row[0].tok)}</td>
          {mode === "j" && <td className="more">{row.slice(1, 3).map((x) => showTok(x.tok)).join("  ")}</td>}
          <td><div className="bar"><i data-w={`${Math.round(row[0].p * 100)}%`} style={moved ? { width: `${Math.round(row[0].p * 100)}%` } : undefined} /></div></td></tr>
      ))}</tbody></table> : <JLensSetup onReady={async () => setJ((await api<TokenDetail>(`/api/trace/token?t=${t}`)).jlens)} />}
    </div>
  );
}

function JLensSetup({ onReady }: { onReady: () => void }) {
  const job = useApp((s) => s.jlens);

  useEffect(() => { watchJlens(); }, []);
  useEffect(() => { if (job?.ready) onReady(); }, [job?.ready]); // eslint-disable-line react-hooks/exhaustive-deps

  if (job?.status === "computing") {
    const left = jlensLeft(job);
    return (
      <div className="jl-setup">
        <div className="bar wide"><i style={{ width: `${(100 * job.done) / Math.max(1, job.total)}%` }} /></div>
        <div className="jl-note">{left ? `About ${left} min left` : "Starting…"}</div>
        <button className="btn ghost sm" onClick={stopJlens}>Stop</button>
      </div>
    );
  }
  return (
    <div className="jl-setup">
      <p className="jl-note">Computed once per model: a few minutes on a small model, longer on a large one.</p>
      <button className="btn sm" onClick={computeJlens}>Compute J-lens</button>
    </div>
  );
}
