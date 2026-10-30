// Save edited weights as a new model. One copy is written at a time; `p` is the page doing it: "abl" (refusals
// erased) or "st" (steering built in).
import { api } from "../lib/api";
import { sleep, steerDefaultName } from "../lib/format";
import type { ExportJob } from "../lib/types";
import { refreshModels } from "../shell/engine";
import { useApp } from "../store/app";
import { useSteer, type Export } from "../store/steer";

const app = useApp.getState;
const exp = () => useSteer.getState().exp;
const setExp = (p: Partial<Export>) => useSteer.setState({ exp: { ...exp(), ...p } });
const toast = (msg: string, err = false) => app().toastMsg(msg, err);

let polling = false;
export async function pollExport(p: "abl" | "st") {
  if (polling) return;
  polling = true;
  setExp({ page: p, running: true, saved: { ...exp().saved, [p]: undefined } });
  try {
    for (;;) {
      let s: ExportJob;
      try { s = await api<ExportJob>("/api/abl/export"); } catch { await sleep(1500); continue; }
      if (s.status === "working") {
        setExp({ line: s.msg || "Writing…", width: (s.total ? Math.round((100 * s.done) / s.total) : 2) + "%" });
      } else {
        setExp({ page: null, width: "0%" });
        if (s.status === "done") {
          setExp({ saved: { ...exp().saved, [p]: { name: s.name!, dst: s.dst! } } });
          toast(`Saved ${s.name}.`);
          await refreshModels();
        } else if (s.status.startsWith("error")) toast(s.status.replace(/^error: /, "Couldn’t save: "), true);
        break;
      }
      await sleep(800);
    }
  } finally {
    polling = false;
    setExp({ running: false });
  }
}

export const cancelExport = () => api("/api/abl/export/cancel", {});

export async function bakeSteer() {
  if (app().busy) return toast("Something is already running. Stop it first.", true);
  const { recipe, info, abl } = app();
  const name = (document.getElementById("st-name") as HTMLInputElement).value.trim() || steerDefaultName(info?.model, recipe.name);
  const keepAbl = !!abl?.current?.active && (document.getElementById("st-abl") as HTMLInputElement).checked;
  try { await api("/api/steer/export", { items: recipe.items, name, abl: keepAbl, recipe: recipe.name }); } catch (e) { return toast((e as Error).message, true); }
  pollExport("st");
}
