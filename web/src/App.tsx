import { useEffect } from "react";
import { FitWarn } from "./components/FitWarn";
import { ModelMenu } from "./components/ModelMenu";
import { Sidebar } from "./components/Sidebar";
import { Toast } from "./components/Toast";
import { boot, setTab, toggleModelMenu } from "./shell/engine";
import { openClipboard } from "./shell/experiments";
import { TABS, useApp } from "./store/app";
import { newChat } from "./chat/engine";
import { Abliterate } from "./views/Abliterate";
import { Arch, archBack } from "./views/Arch";
import { Chat } from "./views/Chat";
import { Home } from "./views/Home";
import { Probe } from "./views/Probe";
import { Start } from "./views/Start";
import { Steer } from "./views/Steer";
import { closeLast } from "./steer/panels";

// Registered as the app loads, not in an effect: React's first render can land after the page's load event, and a
// shortcut pressed before then would be lost.
document.addEventListener("keydown", (e) => {
  const { info, menu, tab, fitWarn } = useApp.getState();
  if (e.key === "Escape") {
    if (fitWarn) useApp.setState({ fitWarn: null });
    else if (menu.open) toggleModelMenu(false);
    else if (tab === "arch") archBack();
    else closeLast();
    return;
  }

  if (!e.metaKey || e.shiftKey || e.altKey) return;

  if (e.key === "v") return void pasted();

  const i = ["1", "2", "3", "4"].indexOf(e.key);
  if (i >= 0) return (e.preventDefault(), setTab(TABS[i]));
  if (e.key === "0" && info?.loaded) return (e.preventDefault(), setTab("home"));
  if (e.key === "k" && tab === "chat" && info?.loaded) (e.preventDefault(), newChat());
});

// ⌘V outside a text field opens an experiment link on the clipboard, so a link from a post is one paste away. In the
// Mac app the Edit menu takes ⌘V before the page sees it, so the app tells the page instead (interp/app.py).
function pasted() {
  if (!document.activeElement?.closest("input, textarea, [contenteditable]")) openClipboard();
}
document.addEventListener("lodestar-paste", pasted);

export function App() {
  const tab = useApp((s) => s.tab);
  const busy = useApp((s) => s.busy);
  const loaded = useApp((s) => !!s.info?.loaded);

  useEffect(() => { document.body.dataset.tab = tab; }, [tab]);
  useEffect(() => { document.body.classList.toggle("busy", busy); }, [busy]);
  useEffect(() => { boot(); }, []);

  const hidden = (view: string) => !loaded || view !== tab;
  return (
    <>
      <Sidebar />
      <main id="main" className={loaded && tab === "chat" ? "is-chat" : loaded && tab === "arch" ? "is-arch" : undefined}>
        <Start />
        <Home hidden={hidden("home")} />
        <Arch hidden={hidden("arch")} />
        <Abliterate hidden={hidden("abliterate")} />
        <Steer hidden={hidden("steer")} />
        <Probe hidden={hidden("probe")} />
        <Chat hidden={hidden("chat")} />
      </main>
      <ModelMenu />
      <FitWarn />
      <Toast />
    </>
  );
}
