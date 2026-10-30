import { useEffect, useRef } from "react";
import { startSideArt } from "../art/glyphs";
import { markSVG } from "../art/mark";
import { modelName } from "../lib/format";
import { deleteSaved, openSaved, setTab, statusLine, toggleModelMenu } from "../shell/engine";
import { ablActive, useApp, type Tab } from "../store/app";
import { Html } from "./Html";

const MARK = markSVG(36, 15);

export const ICONS: Record<Exclude<Tab, "home">, JSX.Element> = {
  abliterate: (
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8.6 3.8 12 2.5l7 2.6v6.2c0 1.6-.35 3-1 4.25M15.9 19.1A13 13 0 0 1 12 21.5c-4.3-1.7-7-5.3-7-10.2V5.6" /><path d="m3 3 18 18" /></svg>
  ),
  steer: (
    <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" /><path d="m15.6 8.4-2.1 5.1-5.1 2.1 2.1-5.1z" /></svg>
  ),
  chat: (
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 11.6c0 4.1-3.8 7.4-8.5 7.4-1.2 0-2.3-.2-3.3-.6L4 20l1.3-3.7C4.2 15 3.5 13.4 3.5 11.6 3.5 7.5 7.3 4.2 12 4.2s8.5 3.3 8.5 7.4z" /></svg>
  ),
};

function TabButton({ tab, label, children }: { tab: Exclude<Tab, "home">; label: string; children?: JSX.Element }) {
  const on = useApp((s) => s.tab === tab);
  return (
    <button className={on ? "tab on" : "tab"} role="tab" data-tab={tab} aria-selected={on} onClick={() => setTab(tab)}>
      {ICONS[tab]}
      <span>{label}</span>
      {children}
    </button>
  );
}

function SavedList() {
  const saved = useApp((s) => s.saved);
  const cur = useApp((s) => s.info?.model);
  const on = useApp((s) => s.recipe.id);
  if (!saved) return <div id="saved-list" className="saved-list" />;
  const mine = saved.filter((r) => !cur || modelName(r.model) === modelName(cur)).reverse(); // by name, so a copy on this Mac keeps them
  return (
    <div id="saved-list" className="saved-list">
      {!mine.length ? <div className="saved-empty">Nothing saved yet.</div> : mine.map((r) => (
        <div key={r.id} className={r.id === on ? "saved-item on" : "saved-item"} data-id={r.id!} role="button" tabIndex={0} onClick={() => openSaved(r)}>
          {"\n      "}<span className="nm">{r.name}</span><span className="ct">{r.items.length}</span>
          <button className="del" aria-label="Delete" onClick={(e) => (e.stopPropagation(), deleteSaved(r.id!))}>×</button>
        </div>
      ))}
    </div>
  );
}

export function Sidebar() {
  const art = useRef<HTMLCanvasElement>(null);
  const info = useApp((s) => s.info);
  const steerCount = useApp((s) => s.recipe.items.length);
  const ablOn = useApp(ablActive);
  const menuOpen = useApp((s) => s.menu.open);
  const { cls, msg } = statusLine(info);
  useEffect(() => startSideArt(art.current!), []);
  const home = () => info?.loaded && setTab("home");
  return (
    <aside className="side">
      <canvas className="side-art" id="side-art" aria-hidden="true" ref={art} />
      <div className="brand" id="brand" role="button" tabIndex={0} title="Home" onClick={home}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && home()}>
        <Html className="brand-mark" id="brand-mark" html={MARK} />
        <h1>Lodestar</h1>
      </div>
      <nav className="tabs" role="tablist">
        <TabButton tab="abliterate" label="Abliterate"><i className="tab-dot" id="tab-abl-dot" hidden={!ablOn} /></TabButton>
        <TabButton tab="steer" label="Steer"><i className="tab-count" id="tab-steer-count" hidden={!steerCount}>{steerCount || null}</i></TabButton>
        <TabButton tab="chat" label="Chat" />
      </nav>
      <div className="saved" data-for="steer">
        <div className="side-h"><span className="eyebrow light">Saved</span></div>
        <SavedList />
      </div>
      <div className="side-fill" />
      <div className="model-card">
        <div className="eyebrow light">Model</div>
        <button id="model-pick" className="model-pick" aria-haspopup="listbox" aria-expanded={menuOpen}
          onClick={(e) => (e.stopPropagation(), toggleModelMenu())}>
          <span className="mp-name">{info?.loaded ? modelName(info.model) : "Choose a model"}</span><span className="mp-caret" aria-hidden="true" />
        </button>
        <div id="model-status" className={"model-status " + cls}><span className="dot" /><span className="txt">{msg}</span></div>
      </div>
    </aside>
  );
}
