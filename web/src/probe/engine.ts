import { api } from "../lib/api";
import { stream } from "../lib/stream";
import { useApp } from "../store/app";
import { useProbe, type Check, type Example, type Experiment, type Probe, type ProbeRow } from "../store/probe";

const get = useProbe.getState, set = useProbe.setState;
const toast = (m: string, err = false) => useApp.getState().toastMsg(m, err);

/** How long the tested prompt takes to ride from layer 0 to the probe's layer on the map. */
export const TRAVEL_MS = 1800;

type Prompt = Experiment["probe"]["a"]["prompts"][number];

// A set as text: one prompt per line; a line "# name" puts the prompts after it in that group, and "#" alone ends it.
export function toText(prompts: Prompt[]) {
  const out: string[] = [];
  let cur: string | null = null;
  for (const p of prompts) {
    const text = typeof p === "string" ? p : p.text, g = typeof p === "string" ? null : p.group || null;
    if (g !== cur) out.push(g ? `# ${g}` : "#");
    cur = g;
    out.push(text);
  }
  return out.join("\n");
}

export function fromText(text: string): Prompt[] {
  let g: string | null = null;
  const out: Prompt[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith("#")) g = line.slice(1).trim() || null;
    else out.push(g ? { text: line, group: g } : line);
  }
  return out;
}

export async function refresh() {
  const r = await api<{ probes: ProbeRow[]; examples: Example[] }>("/api/probes").catch(() => null);
  if (r) set({ list: r.probes, examples: r.examples });
}

export function loadExperiment(exp: Experiment) {
  set({
    exp, checks: null, sameModel: null,
    form: { name: exp.name, aLabel: exp.probe.a.label, a: toText(exp.probe.a.prompts), bLabel: exp.probe.b.label, b: toText(exp.probe.b.prompts) },
  });
}

export async function openExample(file: string) {
  loadExperiment(await api<Experiment>(`/api/experiments/${encodeURIComponent(file)}`));
}

export async function openFile(f: File) {
  try {
    const exp = JSON.parse(await f.text()) as Experiment;
    if (!exp?.probe?.a || !exp.probe.b) throw new Error();
    loadExperiment(exp);
  } catch {
    toast(`${f.name} isn't a Lodestar experiment.`, true);
  }
}

export async function run() {
  const { form: f, exp } = get();
  const probe = { a: { label: f.aLabel.trim() || "A", prompts: fromText(f.a) }, b: { label: f.bLabel.trim() || "B", prompts: fromText(f.b) } };
  const name = f.name.trim() || `${probe.a.label} or ${probe.b.label}`;
  const [path, body] = exp ? ["/api/experiments/run", { ...exp, name, probe }] : ["/api/probes/run", { name, ...probe }];

  set({ live: "Starting", pct: 0, checks: null, sameModel: null });
  const ok = await stream(path, body, (ev) => {
    if (ev.type === "stage") set({ live: String(ev.label), pct: ev.stage === "read" ? 0 : null });
    else if (ev.type === "progress") set({ pct: (ev.done as number) / (ev.total as number), live: `Reading ${ev.done} of ${ev.total}` });
    else if (ev.type === "result") {
      set({ cur: ev.probe as Probe, checks: (ev.checks as Check[]) ?? null, sameModel: (ev.same_model as boolean) ?? null });
      resetTest();
    }
  });
  set({ live: "", pct: null });
  if (ok) await refresh();
  const tries = get().exp?.try;
  if (ok && tries?.length) for (const q of tries) await ask(q);
}

export async function open(id: string) {
  set({ cur: await api<Probe>(`/api/probes/${id}`), checks: null, sameModel: null });
  resetTest();
}

export async function remove(id: string) {
  await api(`/api/probes/${id}`, undefined, "DELETE");
  if (get().cur?.id === id) set({ cur: null });
  await refresh();
}

function resetTest() {
  set({ test: { q: "", p: null, path: null, at: 0, answer: "", asked: [] } });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Ask the probe about a new prompt: where it lands at every layer, then the model's answer. */
export async function ask(q: string) {
  const cur = get().cur;
  q = q.trim();
  if (!cur || !q || useApp.getState().busy) return;

  const asked = [...get().test.asked.filter((x) => x !== q), q];
  set({ test: { q, p: null, path: null, at: 0, answer: "", asked } });
  const s = await api<{ p: number; path: [number, number][] }>(`/api/probes/${cur.id}/score`, { prompt: q }).catch((e) => (toast(e.message, true), null));
  if (!s) return;

  set({ test: { ...get().test, p: s.p, path: s.path, at: performance.now() } });
  await sleep(TRAVEL_MS + 300);
  await stream("/api/steer", { prompt: q, columns: ["base"], pushes: [], max_new: 120 }, (ev) => {
    if (ev.type === "delta") set({ test: { ...get().test, answer: get().test.answer + String(ev.delta) } });
  });
}

/** Write the open probe as an experiment file anyone can run. */
export async function exportExperiment() {
  const { cur, exp, test } = get();
  if (!cur) return;
  const r = await api<{ path: string }>(`/api/probes/${cur.id}/experiment`, { about: exp?.about ?? "", try: test.asked.length ? test.asked : exp?.try ?? [] })
    .catch((e) => (toast(e.message, true), null));
  if (r) toast(`Saved to ${r.path.replace(/^\/Users\/[^/]+/, "~")}`);
}
