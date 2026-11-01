import { useEffect, useRef } from "react";
import type { Part } from "../arch3d/scene";
import { PART_NAMES, attach, back, clearRun, clicked, go, layerTop, loadDetail, openNeuron, pickHead, pickToken, runText, step, useArch } from "../arch3d/state";
import { modelName } from "../lib/format";
import type { ArchBlock, ArchDetail, ArchTensor } from "../lib/types";
import { addItems } from "../steer/recipe";
import { useApp } from "../store/app";

const num = (n: number) => n.toLocaleString("en-US");
const short = (n: number) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n));
const tok = (t: string) => JSON.stringify(t).slice(1, -1);

// The Architecture page: the loaded model in 3D, to fly through and pick apart, with the inspector beside it.
export function Arch({ hidden }: { hidden: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const d = useArch((s) => s.detail);
  const model = useApp((s) => s.info?.model);

  useEffect(() => { if (!hidden) void loadDetail(); }, [hidden, model]);

  // the scene, made while the page is open and dropped when it's left (it holds a WebGL context)
  useEffect(() => {
    if (hidden || !d || !host.current) return;
    let gone = false, dispose = () => {};
    void import("../arch3d/scene").then(({ createScene }) => {
      if (gone) return;
      const s = createScene(host.current!, d, {
        hover: (pick, x, y) => useArch.setState({ hover: pick ? { pick, x, y } : null }),
        click: clicked,
      });
      attach(s);
      dispose = () => (attach(null), s.dispose());
    });
    return () => ((gone = true), dispose());
  }, [hidden, d]);

  useEffect(() => {
    if (hidden) return;
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest?.("input, textarea") || e.metaKey) return;
      if (e.key === "ArrowLeft") (e.preventDefault(), step(-1));
      if (e.key === "ArrowRight") (e.preventDefault(), step(1));
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [hidden]);

  return (
    <section id="view-arch" className="view a3" data-view="arch" hidden={hidden}>
      <div className="a3-main">
        <Top />
        <div className="a3-stage" ref={host}>
          <Tip />
          {!d && <div className="a3-wait">Reading the model…</div>}
        </div>
        {d && <Bottom d={d} />}
      </div>
      {d && <Inspector d={d} />}
    </section>
  );
}

function Top() {
  const d = useArch((s) => s.detail), f = useArch((s) => s.focus), n = useArch((s) => s.neuron), h = useArch((s) => s.head);
  const text = useArch((s) => s.text), running = useArch((s) => s.running), run = useArch((s) => s.run);
  const crumbs: [string, (() => void) | null][] = [["Home", () => useApp.getState().setTab("home")]];
  if (d) {
    crumbs.push([modelName(d.model), f.level === "model" && !n ? null : () => go({ level: "model" })]);
    if (f.level === "end") crumbs.push([f.end === "embed" ? "Embedding" : "Unembedding", null]);
    if (f.level === "layer" || f.level === "part") crumbs.push([`Layer ${f.layer}`, f.level === "layer" ? null : () => go({ level: "layer", layer: f.layer })]);
    if (f.level === "part") crumbs.push([PART_NAMES[f.part], n || h ? () => go(f) : null]);
    if (n) crumbs.push([`Neuron ${n.neuron}`, null]);
    if (h) crumbs.push([`Head ${h.head}`, null]);
  }
  return (
    <div className="a3-top">
      <nav className="a3-crumbs">
        {crumbs.map(([t, fn], i) => (
          <span key={i}>{i > 0 && <i>/</i>}{fn ? <button onClick={fn}>{t}</button> : <b>{t}</b>}</span>
        ))}
      </nav>
      <form className="a3-run" onSubmit={(e) => (e.preventDefault(), void runText())}>
        <input value={text} onChange={(e) => useArch.setState({ text: e.target.value })} spellCheck={false} aria-label="Text to run" />
        <button className="btn sm" disabled={running || !text.trim()}>{running ? "Running…" : "Run"}</button>
        {run && <button type="button" className="btn ghost sm" onClick={clearRun}>Clear</button>}
      </form>
    </div>
  );
}

function Tip() {
  const hv = useArch((s) => s.hover), d = useArch((s) => s.detail), run = useArch((s) => s.run), t = useArch((s) => s.token), acts = useArch((s) => s.acts);
  if (!hv || !d) return null;
  const k = hv.pick;
  let main = "", sub = "";
  if (k.kind === "layer") (main = `Layer ${k.layer}`), (sub = d.layers[k.layer].attention === "linear" ? "Linear attention" : "Full attention");
  if (k.kind === "part") (main = PART_NAMES[k.part]), (sub = `Layer ${k.layer}`);
  if (k.kind === "end") main = k.end === "embed" ? "Embedding" : "Unembedding";
  if (k.kind === "neuron") {
    main = `Neuron ${k.neuron}`;
    const v = acts && acts.q[k.layer * d.neurons + k.neuron] / 127;
    sub = v != null && run ? `${v >= 0 ? "+" : ""}${(v * 100).toFixed(0)}% of the layer's peak at “${tok(run.tokens[t])}”` : `Layer ${k.layer}`;
  }
  if (k.kind === "head") {
    main = `Head ${k.head}`;
    const v = run?.heads[k.layer][t]?.[k.head];
    sub = v != null ? `wrote ${v.toFixed(1)} at “${tok(run!.tokens[t])}”` : `Layer ${k.layer}`;
  }
  const r = document.querySelector(".a3-stage")?.getBoundingClientRect();
  return (
    <div className="a3-tip" style={{ transform: `translate(${hv.x - (r?.left ?? 0) + 14}px, ${hv.y - (r?.top ?? 0) + 14}px)` }}>
      <b>{main}</b>{sub && <span>{sub}</span>}
    </div>
  );
}

function Bottom({ d }: { d: ArchDetail }) {
  const f = useArch((s) => s.focus), run = useArch((s) => s.run), t = useArch((s) => s.token);
  const open = f.level === "layer" || f.level === "part" ? f.layer : -1;
  const w = run ? d.layers.map((_, l) => (run.attn[l][t] ?? 0) + (run.mlp[l][t] ?? 0)) : null;
  const wm = w ? Math.max(...w) : 1;
  return (
    <div className="a3-bottom">
      {run && (
        <div className="a3-tokens">
          {run.tokens.map((x, i) => (
            <button key={i} className={i === t ? "on" : undefined} onClick={() => void pickToken(i)}>{tok(x)}</button>
          ))}
        </div>
      )}
      <div className="a3-strip">
        <button className={`a3-end-c${f.level === "end" && f.end === "embed" ? " on" : ""}`} onClick={() => go({ level: "end", end: "embed" })}>Emb</button>
        {d.layers.map((ly, l) => (
          <button key={l} className={`a3-cell${ly.attention === "linear" ? " lin" : ""}${l === open ? " on" : ""}`}
            title={`Layer ${l}`} onClick={() => go(f.level === "part" ? { ...f, layer: l } : { level: "layer", layer: l })}>
            <i style={w ? { height: `${18 + 82 * (w[l] / wm)}%` } : undefined} />
            <span>{l}</span>
          </button>
        ))}
        <button className={`a3-end-c${f.level === "end" && f.end === "unembed" ? " on" : ""}`} onClick={() => go({ level: "end", end: "unembed" })}>Out</button>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ inspector
function Inspector({ d }: { d: ArchDetail }) {
  const f = useArch((s) => s.focus), n = useArch((s) => s.neuron);
  return (
    <aside className="a3-side">
      {f.level === "model" && <ModelPane d={d} />}
      {f.level === "end" && <EndPane d={d} end={f.end} />}
      {f.level === "layer" && <LayerPane d={d} l={f.layer} />}
      {f.level === "part" && f.part === "attn" && <AttnPane d={d} l={f.layer} />}
      {f.level === "part" && f.part === "mlp" && (n ? <NeuronPane /> : <MlpPane d={d} l={f.layer} />)}
      {f.level === "part" && f.part.startsWith("norm") && <NormPane d={d} l={f.layer} part={f.part} />}
    </aside>
  );
}

function Head({ title, sub }: { title: string; sub?: string }) {
  const up = useArch((s) => s.focus.level !== "model" || !!s.neuron || !!s.head);
  return (
    <header className="a3-h">
      <div className="a3-h-t">
        {up && <button className="a3-back" onClick={back} aria-label="Back" title="Back (Esc)">
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10 3 5 8l5 5" /></svg>
        </button>}
        <h2>{title}</h2>
      </div>
      {sub && <p>{sub}</p>}
    </header>
  );
}

function Facts({ rows }: { rows: [string, string | number | null | undefined][] }) {
  return <dl className="a3-facts">{rows.filter(([, v]) => v != null && v !== "").map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{typeof v === "number" ? num(v) : v}</dd></div>)}</dl>;
}

function Bar({ v, sign }: { v: number; sign?: number }) {
  return <span className={`a3-bar${sign != null && sign < 0 ? " neg" : sign != null ? " pos" : ""}`}><i style={{ width: `${Math.max(1.5, Math.min(100, v * 100))}%` }} /></span>;
}

function Tensors({ list, strip = "" }: { list: ArchTensor[]; strip?: string }) {
  return (
    <table className="a3-tensors">
      <tbody>{list.map((t) => (
        <tr key={t.name}><td>{t.name.startsWith(strip) ? t.name.slice(strip.length) : t.name}</td><td>{t.shape.join(" × ")}</td><td>{short(t.params)}</td></tr>
      ))}</tbody>
    </table>
  );
}

function Sec({ title, children, right }: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return <section className="a3-sec"><div className="a3-sec-h"><h3>{title}</h3>{right}</div>{children}</section>;
}

function AtToken() {
  const run = useArch((s) => s.run), t = useArch((s) => s.token);
  return run ? <span className="a3-at">at “{tok(run.tokens[t])}”</span> : null;
}

function TopNeurons({ rows }: { rows: { layer: number; neuron: number; v: number }[] }) {
  return (
    <div className="a3-list">
      {rows.map((r) => (
        <button key={`${r.layer}:${r.neuron}`} onClick={() => void openNeuron(r.layer, r.neuron)}>
          <span className="mono">L{r.layer} · {r.neuron}</span><Bar v={Math.abs(r.v)} sign={r.v} />
        </button>
      ))}
    </div>
  );
}

function ModelPane({ d }: { d: ArchDetail }) {
  const acts = useArch((s) => s.acts);
  const sum = (role: ArchBlock["role"]) => d.layers.reduce((a, l) => a + l.blocks.filter((b) => b.role === role).reduce((x, b) => x + b.params, 0), 0);
  const parts: [string, number][] = [
    ["Embedding", d.embedding.params], ["Attention", sum("attention")], ["MLP", sum("mlp")],
    ["Norms", sum("norm") + d.final_norm.params], ["Unembedding", d.unembedding.params], ["Other", sum("other")],
  ].filter(([, v]) => (v as number) > 0) as [string, number][];
  const total = parts.reduce((a, [, v]) => a + v, 0);
  const lin = d.layers.filter((l) => l.attention === "linear").length;
  return (
    <>
      <Head title={modelName(d.model)} sub={`${d.type} · ${d.dtype}`} />
      <Sec title="Parameters" right={<span className="a3-at">{short(total)}</span>}>
        <div className="a3-split">{parts.map(([k, v]) => (
          <div key={k}><span>{k}</span><Bar v={v / total} /><span className="mono">{short(v)}</span></div>
        ))}</div>
      </Sec>
      <Sec title="Shape">
        <Facts rows={[
          ["Layers", d.layers.length],
          ["Hidden size", d.hidden],
          ["Vocabulary", `${num(d.vocab)} tokens`],
          ["Attention", lin ? `${d.layers.length - lin} full · ${lin} linear` : "Full, every layer"],
          ["Heads", d.heads ? `${d.heads}${d.kv_heads && d.kv_heads !== d.heads ? ` · ${d.kv_heads} key/value` : ""}` : null],
          ["Linear heads", lin && d.linear_heads ? `${d.linear_heads} · ${d.linear_k_heads} key` : null],
          [d.experts ? "Experts" : "Neurons", d.experts ? `${d.experts} a layer` : `${num(d.neurons)} a layer`],
          ["Activation", d.activation],
          ["Context", d.context ? `${num(d.context)} tokens` : null],
          ["RoPE base", d.rope_theta ? num(d.rope_theta) : null],
          ["Norm epsilon", d.norm_eps != null ? String(d.norm_eps) : null],
          ["Unembedding", d.tied ? "The embedding, reused" : "Its own weights"],
        ]} />
      </Sec>
      {acts && <Sec title="Strongest neurons" right={<AtToken />}>
        <TopNeurons rows={acts.top.slice(0, 12).map((r) => ({ layer: r.layer, neuron: r.neuron, v: r.impact / Math.abs(acts.top[0].impact || 1) }))} />
      </Sec>}
    </>
  );
}

function EndPane({ d, end }: { d: ArchDetail; end: "embed" | "unembed" }) {
  if (end === "embed") return (
    <>
      <Head title="Embedding" sub={`${num(d.vocab)} tokens × ${num(d.hidden)}`} />
      <Sec title="Tensors"><Tensors list={d.embedding.tensors} /></Sec>
    </>
  );
  return (
    <>
      <Head title="Unembedding" sub={d.tied ? "The embedding, reused" : `${num(d.hidden)} × ${num(d.vocab)} tokens`} />
      <Sec title="Final norm"><Tensors list={d.final_norm.tensors} /></Sec>
      {!d.tied && <Sec title="Tensors"><Tensors list={d.unembedding.tensors} /></Sec>}
      {d.tied && <Sec title="Tensors"><Tensors list={d.embedding.tensors} /></Sec>}
    </>
  );
}

const partOf = (b: ArchBlock): Part | null =>
  b.role === "attention" ? "attn" : b.role === "mlp" ? "mlp" : b.role === "norm" ? (b.name.includes("post") || b.name.includes("2") ? "norm2" : "norm1") : null;

function LayerPane({ d, l }: { d: ArchDetail; l: number }) {
  const ly = d.layers[l], run = useArch((s) => s.run), t = useArch((s) => s.token), acts = useArch((s) => s.acts);
  const lens = run?.lens[l][t];
  return (
    <>
      <Head title={`Layer ${l}`} sub={`${ly.attention === "linear" ? "Linear attention" : "Full attention"} · ${short(ly.params)} parameters`} />
      {run && <Sec title="Stream" right={<AtToken />}>
        <Facts rows={[["Size", run.resid[l][t].toFixed(1)], ["Attention wrote", run.attn[l][t]?.toFixed(1)], ["MLP wrote", run.mlp[l][t].toFixed(1)]]} />
        {lens && <div className="a3-lensl">{lens.map((x, i) => <div key={i}><span className="mono">{tok(x.tok)}</span><Bar v={x.p} /><span className="mono">{(x.p * 100).toFixed(0)}%</span></div>)}</div>}
      </Sec>}
      <Sec title="Blocks">
        <div className="a3-blocks">{ly.blocks.map((b) => {
          const p = partOf(b);
          return (
            <details key={b.name} className="a3-block">
              <summary>
                <span>{b.name}</span><span className="a3-dim">{b.type}</span><span className="mono">{short(b.params)}</span>
                {p && <button className="btn ghost sm" onClick={(e) => (e.preventDefault(), go({ level: "part", layer: l, part: p }))}>Open</button>}
              </summary>
              <Tensors list={b.tensors} strip={`${b.name}.`} />
            </details>
          );
        })}</div>
      </Sec>
      {acts && <Sec title="Strongest neurons here" right={<AtToken />}>
        <TopNeurons rows={layerTop(acts, l, d.neurons).map(([n, v]) => ({ layer: l, neuron: n, v }))} />
      </Sec>}
    </>
  );
}

function AttnPane({ d, l }: { d: ArchDetail; l: number }) {
  const ly = d.layers[l], run = useArch((s) => s.run), t = useArch((s) => s.token), sel = useArch((s) => s.head);
  const lin = ly.attention === "linear";
  const kv = (lin ? d.linear_k_heads : d.kv_heads) || ly.heads.length;
  const gs = Math.max(1, Math.round(ly.heads.length / kv));
  const wm = Math.max(1e-6, ...ly.heads), at = run?.heads[l][t] ?? [], am = Math.max(1e-6, ...at);
  const block = ly.blocks.find((b) => b.role === "attention");
  return (
    <>
      <Head title={`${lin ? "Linear attention" : "Attention"} · layer ${l}`}
        sub={`${ly.heads.length} heads · ${kv} ${lin ? "key" : "key/value"} heads · head size ${ly.head_dim ?? "—"}`} />
      <Sec title="Heads" right={<AtToken />}>
        <div className="a3-heads">
          <div className="a3-heads-h"><span>Head</span><span>Output weights</span>{run && <span>Wrote</span>}</div>
          {ly.heads.map((w, h) => (
            <button key={h} className={sel?.layer === l && sel.head === h ? "on" : undefined} onClick={() => pickHead(l, h)}>
              <span className="mono">{h}<i>{Math.floor(h / gs)}</i></span><Bar v={w / wm} />{run && <Bar v={(at[h] ?? 0) / am} sign={1} />}
            </button>
          ))}
        </div>
      </Sec>
      {block && <Sec title="Tensors"><Tensors list={block.tensors} strip={`${block.name}.`} /></Sec>}
    </>
  );
}

function MlpPane({ d, l }: { d: ArchDetail; l: number }) {
  const norms = useArch((s) => s.norms[l]), acts = useArch((s) => s.acts);
  const block = d.layers[l].blocks.find((b) => b.role === "mlp");
  const big = norms ? norms.map((v, i) => [i, v] as const).sort((a, b) => b[1] - a[1]).slice(0, 8) : [];
  const hist = norms ? histogram(norms, 24) : null;
  return (
    <>
      <Head title={`MLP · layer ${l}`} sub={`${num(d.neurons)} neurons${d.activation ? ` · ${d.activation}` : ""}`} />
      {hist && <Sec title="How much each neuron writes">
        <div className="a3-hist">{hist.bins.map((c, i) => <i key={i} style={{ height: `${(c / hist.max) * 100}%` }} />)}</div>
        <div className="a3-hist-x mono"><span>{hist.lo.toFixed(2)}</span><span>{hist.hi.toFixed(2)}</span></div>
      </Sec>}
      {acts && <Sec title="Strongest here" right={<AtToken />}>
        <TopNeurons rows={layerTop(acts, l, d.neurons).map(([n, v]) => ({ layer: l, neuron: n, v }))} />
      </Sec>}
      {!!big.length && <Sec title="Write the most">
        <TopNeurons rows={big.map(([n, v]) => ({ layer: l, neuron: n, v: v / big[0][1] }))} />
      </Sec>}
      {block && <Sec title="Tensors"><Tensors list={block.tensors} strip={`${block.name}.`} /></Sec>}
    </>
  );
}

function histogram(v: number[], k: number) {
  const lo = Math.min(...v), hi = Math.max(...v), bins = new Array(k).fill(0);
  for (const x of v) bins[Math.min(k - 1, Math.floor(((x - lo) / (hi - lo || 1)) * k))]++;
  return { lo, hi, bins, max: Math.max(...bins) };
}

function NeuronPane() {
  const n = useArch((s) => s.neuron)!, run = useArch((s) => s.run), t = useArch((s) => s.token), toast = useApp((s) => s.toastMsg);
  const am = n.acts ? Math.max(1e-6, ...n.acts.map(Math.abs)) : 1;
  const steer = () => (addItems([{ layer: n.layer, neuron: n.neuron, dose: 1, lens: n.lens.up.slice(0, 6) }]), toast(`Neuron ${n.layer}:${n.neuron} added to Steer`));
  return (
    <>
      <Head title={`Neuron ${n.neuron}`} sub={`Layer ${n.layer} · writes ${n.norm.toFixed(3)} per unit`} />
      <Sec title="Pushes toward"><div className="a3-toks">{n.lens.up.map((x, i) => <span key={i}>{tok(x)}</span>)}</div></Sec>
      <Sec title="Pushes away from"><div className="a3-toks neg">{n.lens.down.map((x, i) => <span key={i}>{tok(x)}</span>)}</div></Sec>
      {run && n.acts && <Sec title="Activation" right={<AtToken />}>
        <div className="a3-tokacts">{run.tokens.map((x, i) => {
          const v = n.acts![i] ?? 0;
          return <button key={i} className={i === t ? "on" : undefined} onClick={() => void pickToken(i)}
            style={{ ["--v" as string]: String(Math.abs(v) / am), ["--c" as string]: v >= 0 ? "var(--g-pos)" : "var(--g-neg)" }} title={v.toFixed(3)}>{tok(x)}</button>;
        })}</div>
      </Sec>}
      <div className="a3-actions"><button className="btn sm" onClick={steer}>Add to Steer</button></div>
    </>
  );
}

function NormPane({ d, l, part }: { d: ArchDetail; l: number; part: Part }) {
  const block = d.layers[l].blocks.find((b) => partOf(b) === part);
  return (
    <>
      <Head title={`${PART_NAMES[part]} · layer ${l}`} sub={block?.type} />
      {block && <Sec title="Tensors">
        <Tensors list={block.tensors} strip={`${block.name}.`} />
        <Facts rows={block.tensors.map((t) => [`${t.name.slice(block.name.length + 1)} size`, t.rms.toFixed(4)])} />
      </Sec>}
    </>
  );
}

export { back as archBack };
