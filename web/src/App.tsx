import { useEffect } from "react";
import { ModelMenu } from "./components/ModelMenu";
import { Sidebar } from "./components/Sidebar";
import { Toast } from "./components/Toast";
import { boot, setTab, toggleModelMenu } from "./shell/engine";
import { TABS, useApp } from "./store/app";
import { Abliterate } from "./views/Abliterate";
import { Home } from "./views/Home";
import { Start } from "./views/Start";
import { Steer } from "./views/Steer";
import { closeLast } from "./steer/panels";

function useShortcuts() {
  useEffect(() => {
    const keys = (e: KeyboardEvent) => {
      const { info, menu } = useApp.getState();
      if (e.key === "Escape") {
        if (menu.open) toggleModelMenu(false);
        else closeLast();
        return;
      }
      if (!e.metaKey || e.shiftKey || e.altKey) return;
      const i = ["1", "2", "3"].indexOf(e.key);
      if (i >= 0) return (e.preventDefault(), setTab(TABS[i]));
      if (e.key === "0" && info?.loaded) return (e.preventDefault(), setTab("home"));
      // TODO(Phase 6): ⌘K starts a new chat
    };
    document.addEventListener("keydown", keys);
    return () => document.removeEventListener("keydown", keys);
  }, []);
}

export function App() {
  const tab = useApp((s) => s.tab);
  const busy = useApp((s) => s.busy);
  const loaded = useApp((s) => !!s.info?.loaded);
  useEffect(() => { document.body.dataset.tab = tab; }, [tab]);
  useEffect(() => { document.body.classList.toggle("busy", busy); }, [busy]);
  useEffect(() => { boot(); }, []);
  useShortcuts();
  const hidden = (view: string) => !loaded || view !== tab;
  return (
    <>
      <Sidebar />
      <main id="main" className={loaded && tab === "chat" ? "is-chat" : undefined}>
        <Start />
        <Home hidden={hidden("home")} />
        {/* TODO(Phase 6): the rest of Chat; only its header so far */}
        <Abliterate hidden={hidden("abliterate")} />
        <Steer hidden={hidden("steer")} />
        <section id="view-chat" className="view chat" data-view="chat" hidden={hidden("chat")}>
          <header className="chat-h">
            <div className="chat-title"><b id="chat-model" /></div>
            <div className="chat-mods" id="chat-mods" />
            <button id="chat-new" className="btn ghost sm" title="New chat (⌘K)">New chat</button>
          </header>
        </section>
      </main>
      <ModelMenu />
      <Toast />
    </>
  );
}
