import { modelName, ram } from "../lib/format";
import { loadModel } from "../shell/engine";
import { useApp } from "../store/app";

const close = () => useApp.setState({ fitWarn: null });

/** Asked before loading a model that won't fit beside what's already running: loading it makes macOS swap. */
export function FitWarn() {
  const w = useApp((s) => s.fitWarn);
  if (!w) return null;

  const { need = 0, free = 0, others = [] } = w.fit;

  return (
    <div className="dialog-back" onClick={close}>
      <div className="dialog" id="fit-warn" role="alertdialog" aria-labelledby="fit-h" onClick={(e) => e.stopPropagation()}>
        <h3 id="fit-h">{modelName(w.id)} won’t fit in memory</h3>
        <p>It needs about {ram(need)}, and about {ram(free)} is free.</p>

        {!!others.length && (
          <div className="fit-who">
            <div className="eyebrow">Already using memory</div>
            {others.map((o) => (
              <div className="fit-proc" key={o.pid}><span>{o.name}</span><span className="mem-n">{ram(o.bytes)}</span></div>
            ))}
          </div>
        )}

        <p className="dialog-note">Loading it anyway makes macOS swap to disk, and replies slow to a crawl.</p>
        <div className="dialog-acts">
          <button className="btn ghost" onClick={close}>Cancel</button>
          <button className="btn" id="fit-go" onClick={() => (close(), loadModel(w.id, true))}>Load anyway</button>
        </div>
      </div>
    </div>
  );
}
