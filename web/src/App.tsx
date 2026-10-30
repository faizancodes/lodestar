import { useEffect } from "react";
import { ModelMenu } from "./components/ModelMenu";
import { Sidebar } from "./components/Sidebar";
import { Toast } from "./components/Toast";
import { boot, setTab, toggleModelMenu } from "./shell/engine";
import { TABS, useApp } from "./store/app";
import { Home } from "./views/Home";
import { Start } from "./views/Start";

function useShortcuts() {
  useEffect(() => {
    const keys = (e: KeyboardEvent) => {
      const { info, menu } = useApp.getState();
      if (e.key === "Escape") {
        if (menu.open) toggleModelMenu(false);
        // TODO(Phase 7): else close the top panel
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
        {/* TODO(Phases 4–6): the rest of each page; only their headers so far */}
        <section id="view-abliterate" className="view page" data-view="abliterate" hidden={hidden("abliterate")}>
          <header className="page-h">
            <h2>Abliterate</h2>
            <p>Erase the model’s refusal direction, check the result, then save it as a new model{" "}
              (<a href="https://arxiv.org/abs/2406.11717" target="_blank">Arditi et al.</a>).</p>
          </header>
        </section>
        <section id="view-steer" className="view page" data-view="steer" hidden={hidden("steer")}>
          <header className="page-h">
            <h2>Steer</h2>
            <p>Find the neurons behind a behaviour, push them up or down, and see how its answers change.</p>
          </header>
        </section>
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
