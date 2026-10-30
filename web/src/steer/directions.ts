// Directions. A direction is the difference between how the model carries two sides of something (two personas, or
// two kinds of prompt), read off its residual stream. Pushing along it over a band of middle layers moves the whole
// behaviour.
import { flushSync } from "react-dom";
import { api } from "../lib/api";
import { bandText, fmt } from "../lib/format";
import { stream } from "../lib/stream";
import type { Direction } from "../lib/types";
import { refreshDirs } from "../shell/engine";
import { useApp } from "../store/app";
import { useSteer, type Dn } from "../store/steer";
import { lines } from "./describe";
import { addItems } from "./recipe";
import { sink, streamInto } from "./sink";

const app = useApp.getState;
const dn = () => useSteer.getState().dn;
export const setDn = (p: Partial<Dn>) => useSteer.setState({ dn: { ...dn(), ...p } });
const toast = (msg: string, err = false) => app().toastMsg(msg, err);

export function turnOnDir(d: Direction) {
  if (d.ready) addItems([{ kind: "dir", dir: d.id, name: d.name, layers: [...d.band], op: "push", dose: 4 }]);
}

export async function deleteDir(d: Direction) {
  if (!confirm(`Delete the direction “${d.name}”?`)) return;
  try { await api(`/api/directions/${encodeURIComponent(d.id)}`, null, "DELETE"); } catch (err) { return toast((err as Error).message, true); }
  refreshDirs();
}

const focus = (id: string) => document.getElementById(id)?.focus();

export function toggleDn() {
  if (dn().open) return setDn({ open: false });
  flushSync(() => setDn({ open: true })); // shown before it can take focus
  focus("dn-name");
}

export async function dnExample() {
  let ex: { name: string; pos: string[]; neg: string[] };
  try { ex = (await api<{ example: typeof ex }>("/api/directions/defaults")).example; } catch (e) { return toast((e as Error).message, true); }
  setDn({ kind: "persona", name: ex.name, pos: ex.pos.join("\n"), neg: ex.neg.join("\n") });
}

const say = (text: string, running: boolean) => setDn({ live: { text, running } });

export async function findDirection() {
  if (app().busy) return;
  const d = dn(), pos = lines(d.pos), neg = lines(d.neg), name = d.name.trim();
  if (!name) return focus("dn-name");
  if (pos.length < 2 || neg.length < 2) return toast("Give each side at least two lines.", true);

  setDn({ result: null, made: null });
  let label = "", tries: ReturnType<typeof sink>[] = [];
  say("Starting…", true);

  const ok = await stream("/api/directions/extract", { name, kind: d.kind, pos, neg }, (ev) => {
    if (ev.type === "stage") {
      say((label = ev.label as string) + "…", true);
      if (ev.stage === "try") flushSync(() => setDn({ result: { facts: [], q: "", tries: (tries = [sink(), sink(), sink()]) }, typing: true }));
    } else if (ev.type === "progress") say(`${label} · ${ev.done} of ${ev.total}`, true);
    else if (ev.type === "trydelta") streamInto(tries[ev.index as number], ev.delta as string);
    else if (ev.type === "result") setDn({ made: ev as Dn["made"] });
  });

  setDn({ typing: false });
  const m = dn().made;
  if (!ok || !m) return say("Stopped. Nothing was saved.", false);

  const band = m.band, at = (a: number[]) => Math.min(...band.map((l) => a[l]));
  say("", false);
  setDn({
    result: {
      tries, q: m.prompt,
      facts: [["Layers", bandText(band)], ["Separates", `${Math.round(at(m.auc) * 100)}%`], ["Split agreement", at(m.cos_split).toFixed(2)]],
    },
  });

  refreshDirs();
}

export function turnOnMade() {
  const m = dn().made, e = m?.entry;
  if (e) addItems([{ kind: "dir", dir: e.id, name: e.name, layers: [...m.band], op: "push", dose: 4 }]);
}

export async function importDirs() {
  const path = dn().path.trim();
  if (!path || app().busy) return focus("dn-path");

  say("Importing…", true);
  try {
    const r = await api<{ added: { name: string }[] }>("/api/directions/import", { path });
    say(`Imported ${r.added.map((d) => d.name).join(", ") || "nothing"}.`, false);
  } catch (e) { return say((e as Error).message, false); }

  refreshDirs();
}

export const TRIES = [-4, 0, 4].map((k) => (k ? fmt(k, 0) : "Normal"));
