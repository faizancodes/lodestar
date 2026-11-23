import { useApp } from "../store/app";
import { errorMessage } from "./api";
import type { StreamEvent } from "./types";

/** Runs one server-sent-event stream, handing each event to `on`. Only one stream runs at a time (the engine has one
 * model), and body.busy shows the Stop buttons while it does. Errors and cancellations become toasts; resolves to
 * whether the stream finished cleanly. */
export async function stream(path: string, body: unknown, on: (ev: StreamEvent) => void): Promise<boolean> {
  const { busy, setBusy, toastMsg: toast } = useApp.getState();
  if (busy) {
    toast("Something is already running. Stop it first.", true);
    return false;
  }

  setBusy(true);
  let ok = true;

  try {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    if (!r.body) throw new Error("Empty response stream.");

    const rd = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";

    const drain = () => {
      let i: number;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 2);
        if (!line.startsWith("data: ")) continue;
        const ev: StreamEvent = JSON.parse(line.slice(6));
        if (ev.type === "error") {
          toast(String(ev.message), true);
          ok = false;
        } else if (ev.type === "cancelled") {
          toast("Stopped.");
          ok = false;
        } else if (ev.type !== "end") {
          on(ev);
        }
      }
    };

    try {
      for (;;) {
        const { value, done } = await rd.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        drain();
      }
      buf += dec.decode();
      drain();
    } finally {
      rd.releaseLock();
    }
  } catch (e) {
    toast(errorMessage(e), true);
    ok = false;
  } finally {
    setBusy(false);
  }

  return ok;
}
