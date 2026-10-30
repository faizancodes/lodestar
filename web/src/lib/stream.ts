import { useApp } from "../store/app";
import type { StreamEvent } from "./types";

/** Runs one server-sent-event stream, handing each event to `on`. Only one stream runs at a time (the engine has one
 * model), and body.busy shows the Stop buttons while it does. Errors and cancellations become toasts; resolves to
 * whether the stream finished cleanly. */
export async function stream(path: string, body: unknown, on: (ev: StreamEvent) => void): Promise<boolean> {
  const { busy, setBusy, toastMsg: toast } = useApp.getState();
  if (busy) { toast("Something is already running. Stop it first.", true); return false; }
  setBusy(true);
  let ok = true;
  try {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    const rd = r.body!.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await rd.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 2);
        if (!line.startsWith("data: ")) continue;
        const ev: StreamEvent = JSON.parse(line.slice(6));
        if (ev.type === "error") { toast(String(ev.message), true); ok = false; }
        else if (ev.type === "cancelled") { toast("Stopped."); ok = false; }
        else if (ev.type !== "end") on(ev);
      }
    }
  } catch (e) {
    toast((e as Error).message, true);
    ok = false;
  } finally {
    setBusy(false);
  }
  return ok;
}
