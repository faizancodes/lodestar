// Change how it acts: one box. A word or two searches by word; a sentence is described as a behaviour and a live
// steering plan finds the neurons for it.
import { flushSync } from "react-dom";
import { api } from "../lib/api";
import { instruction, isWord, searchDose, styleShift } from "../lib/format";
import { stream } from "../lib/stream";
import type { WordSearch } from "../lib/types";
import { useApp } from "../store/app";
import { useSteer, type Cand, type Plan } from "../store/steer";

const app = useApp.getState;
const get = useSteer.getState;
const set = useSteer.setState;
const toast = (msg: string, err = false) => app().toastMsg(msg, err);
const field = (id: string) => document.getElementById(id) as HTMLInputElement;
export const lines = (s: string) => s.split("\n").flatMap((x) => (x.trim() ? [x.trim()] : []));

export function setInput(v: string) {
  set({ input: v });
  const t = v.trim();
  if (!app().busy) set({ note: t && !isWord(t) ? "Finds the neurons to tune for this. Takes a few minutes." : "" });
}

export function findChanges() {
  const v = get().input.trim();
  if (!v || app().busy) return;
  isWord(v) ? searchWord(v) : describe(v);
}

// ------------------------------------------------------------------ by word
async function searchWord(w: string) {
  const n = (get().results?.n ?? 0) + 1;
  set({ note: "Searching…", results: { n } });

  try {
    // the last ~10% of layers hold blunt "say this exact word" neurons; mid-late layers hold the concept
    const skip = field("w-last").checked ? 0 : Math.max(2, Math.round(app().info!.layers! / 9));
    const r = await api<WordSearch>(`/api/search?word=${encodeURIComponent(w)}&k=16&skip_last=${skip}`);
    set({
      note: r.partial
        ? `“${w}” is split into pieces, so these match its first piece, “${r.piece}”. Turn on one or two.`
        : `Neurons that push it toward “${w}”. Turn on one or two.`,
    });

    // start each neuron where its direct push is noticeable but not yet a loop
    const rows = r.results.map((x) => ({ ...x, dose: searchDose(x.score) }));
    set({ results: { n, word: rows.slice(0, 6) } });
  } catch (e) {
    set({ note: (e as Error).message });
  }
}

// ------------------------------------------------------------------ describe
// The model first answers with the instruction (so you can see how it read it), then every neuron is scored by
// gradient, the strongest are actually tested, and the best are combined.
export const STAGES = [
  { key: "examples", label: "Examples", of: ["draft"] },
  { key: "find", label: "Find", of: ["baseline", "attribute", "patch"] },
  { key: "test", label: "Test", of: ["verify", "hits"] },
  { key: "combine", label: "Combine", of: ["combine", "scale"] },
  { key: "check", label: "Check", of: ["proof"] },
];

/** Draws the plan again in full, now, as each event arrives. */
export function renderPlan(plan: Plan) {
  plan.renders++;
  flushSync(() => set((s) => ({ v: s.v + 1 })));
}

/** Draws a part of it (the example, the proof, a stage's fill). */
const touchPlan = () => flushSync(() => set((s) => ({ v: s.v + 1 })));

// the instruction produced the model's ordinary answers, so a search would only churn and find nothing
function noEffect(plan: Plan) {
  plan.done = true;
  plan.stage = STAGES.length;
  plan.noEffect = true;
  plan.ordinary = true;
  renderPlan(plan);
}

function planClock(plan: Plan, started: number) {
  return setInterval(() => {
    const s = Math.floor((Date.now() - started) / 1000);
    plan.clock = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    set((st) => ({ v: st.v + 1 }));
  }, 1000);
}

type Pairs = { prompt: string; baseline: string; target: string }[];

async function describe(goal: string) {
  const prompts = lines(get().prompts);
  if (prompts.length < 3) return toast("Add at least three test questions in Options.", true);

  const behavior = instruction(goal);
  const plan: Plan = { goal, cands: new Map(), recipe: [], stage: 0, frac: 0, example: "", examplePrompt: prompts[0], proof: {}, done: false, clock: "0:00", renders: 0 };
  set({ note: "", results: { n: (get().results?.n ?? 0) + 1, plan } });
  const started = Date.now();
  const clock = planClock(plan, started);

  try {
    plan.label = "Writing example answers in that style";
    renderPlan(plan);

    let pairs: Pairs | null = null;
    const ok = await stream("/api/tune/draft", { behavior, prompts, max_new: 64 }, (ev) => {
      if (ev.type === "draft" && ev.index === 0 && ev.kind === "target") ((plan.example += ev.delta as string), touchPlan());
      if (ev.type === "drafted") pairs = ev.pairs as Pairs;
    });
    if (!ok || !pairs) return stopPlan(plan, "Stopped.");
    const p: Pairs = pairs;

    // if the instruction barely changed the answers there is probably nothing to steer toward; say so now
    // instead of spending minutes searching. The user can still force the search from the message.
    if (styleShift(p) < 0.5) {
      plan.resume = () => runSearch(plan, behavior, p);
      return noEffect(plan);
    }

    await runSearch(plan, behavior, p, started);
  } finally {
    clearInterval(clock);
  }
}

async function runSearch(plan: Plan, behavior: string, pairs: Pairs, started = Date.now()) {
  Object.assign(plan, { noEffect: false, done: false, stage: 0, frac: 0, resume: null });
  renderPlan(plan);
  const clock = planClock(plan, started);

  const setStage = (engineStage: string, label: string) => {
    plan.stage = Math.max(plan.stage, STAGES.findIndex((x) => x.of.includes(engineStage)));
    plan.frac = 0;
    plan.label = label;
    renderPlan(plan);
  };

  const settings = {
    max_neurons: +field("d-maxn").value, candidates: +field("d-cands").value || null,
    doses: field("d-doses").value.split(",").flatMap((x) => (+x > 0 ? [+x] : [])),
    fluency_budget: +field("d-flu").value, contrast: 0.5, max_new: 64, focus: field("d-focus").value,
  };

  try {
    let result: Plan["result"] | null = null;
    const ok = await stream("/api/tune/search", { pairs, behavior, settings }, (ev) => {
      const key = (c: Cand) => `${c.layer}:${c.neuron}`;
      const cands = ev.candidates as Cand[];
      if (ev.type === "stage") setStage(ev.stage as string, ev.label as string);
      else if (ev.type === "progress") ((plan.frac = (ev.done as number) / (ev.total as number)), touchPlan());
      else if (ev.type === "candidates") (cands.forEach((c) => plan.cands.set(key(c), c)), renderPlan(plan));
      else if (ev.type === "measured" || ev.type === "verified")
        (cands.forEach((c) => plan.cands.set(key(c), { ...plan.cands.get(key(c))!, ...c, tested: c.dj !== undefined })), renderPlan(plan));
      else if (ev.type === "recipe") ((plan.recipe = ev.recipe as Plan["recipe"]), renderPlan(plan));
      else if (ev.type === "proof" && ev.index === 0) {
        const k = ev.kind as "baseline" | "steered";
        plan.proof.prompt = ev.prompt as string;
        plan.proof[k] = (plan.proof[k] || "") + ev.delta;
        touchPlan();
      } else if (ev.type === "result") result = ev as Plan["result"];
    });

    if (!ok || !result) return stopPlan(plan, "Stopped. The neurons found so far are below.");
    const r: NonNullable<Plan["result"]> = result;

    plan.recipe = r.recipe;
    plan.result = r;
    plan.done = true;
    plan.stage = STAGES.length;
    renderPlan(plan);
  } finally {
    clearInterval(clock);
  }
}

function stopPlan(plan: Plan, msg: string) {
  plan.stopped = msg;
  renderPlan(plan);
}

export function planSummary(plan: Plan) {
  const r = plan.result;
  if (!plan.recipe.length) return r?.note || "No neuron moved the model toward that. Try wording it differently.";

  const n = plan.recipe.length;
  const words = r?.hit_test != null ? ` On new questions it says your words ${Math.round(r.hit_test * 100)}% of the time (normally ${Math.round(r.hit_test0! * 100)}%).` : "";
  return `Tune these ${n} neuron${n > 1 ? "s" : ""} to the strengths shown.${words}`;
}
