import { useEffect, useState } from "react";
import { loadModel, modelChoices, toggleModelMenu, type Choice } from "../shell/engine";
import { useApp } from "../store/app";
import { CopyAction } from "./CopyAction";

export function ModelMenu() {
  const menu = useApp((s) => s.menu);

  // a click anywhere outside closes it
  useEffect(() => {
    const close = (e: MouseEvent) => !(e.target as Element).closest("#model-menu") && useApp.getState().menu.open && toggleModelMenu(false);
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, []);

  const placed = menu.maxHeight > 0; // where it goes is worked out when it first opens
  return (
    <div id="model-menu" className="model-menu" role="listbox" hidden={!menu.open}
      style={placed ? { left: menu.left + "px", bottom: menu.bottom + "px", maxHeight: menu.maxHeight + "px" } : undefined}>
      {menu.open && <MenuBody />}
    </div>
  );
}

function MenuBody() {
  const models = useApp((s) => s.models);
  const copy = useApp((s) => s.copy);
  const info = useApp((s) => s.info);
  const [custom, setCustom] = useState("");

  const cur = info?.loaded ? info.model : null;
  const c = modelChoices(models, copy);

  const item = (x: Choice) => (
    <div className="mm-row" data-id={x.id} key={x.id}>
      <button className={x.id === cur ? "mm-item on" : "mm-item"} onClick={() => x.id !== cur && loadModel(x.id)}>
        <b>{x.name}</b><span>{x.sub}</span><i className="tag">{x.id === cur ? "Loaded" : x.tag || ""}</i>
      </button>
      {x.id === cur ? null : <CopyAction x={x} />}
    </div>
  );

  const group = (title: string, rows: Choice[]) => (rows.length ? [<div className="mm-group" key={title}>{title}</div>, ...rows.map(item)] : null);
  const go = () => custom.trim() && loadModel(custom.trim());

  return (
    <>
      {group("On this Mac", c.here)}{group("On external drives", c.drives)}{group("Download", c.get)}
      <div className="mm-custom">
        <input id="mm-custom" placeholder="Any Hugging Face name" spellCheck={false} value={custom}
          onChange={(e) => setCustom(e.target.value)} onKeyDown={(e) => e.key === "Enter" && go()} />
        <button className="btn sm" id="mm-go" onClick={go}>Load</button>
      </div>
    </>
  );
}
