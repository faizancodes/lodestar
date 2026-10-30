import { useEffect } from "react";
import { drawHomeArt, type HomeArt } from "../art/home";
import { modelName } from "../lib/format";
import { setTab } from "../shell/engine";
import { useApp } from "../store/app";
import { ICONS } from "../components/Sidebar";

const CARDS: { go: HomeArt; title: string; text: string; key: string }[] = [
  { go: "abliterate", title: "Abliterate", text: "Find the direction it refuses along, erase it, and save the result as a new model.", key: "⌘1" },
  { go: "steer", title: "Steer", text: "Push individual neurons to change how it talks, then save what works as recipes.", key: "⌘2" },
  { go: "chat", title: "Chat", text: "Talk to it with your steering and erased refusals switched on or off.", key: "⌘3" },
];

// Home: the loaded model, and a card for each page with what it's for and where it stands.
export function Home({ hidden }: { hidden: boolean }) {
  const s = useApp((s) => s.info);
  const cur = useApp((s) => s.abl?.current);
  const mine = useApp((s) => (s.saved || []).filter((r) => modelName(r.model) === modelName(s.info?.model)).length);
  const shown = !hidden && !!s?.loaded;

  useEffect(() => {
    if (!shown) return;
    const raf = requestAnimationFrame(() => document.querySelectorAll<HTMLCanvasElement>(".home-art")
      .forEach((c) => drawHomeArt(c, c.dataset.art as HomeArt, c.closest<HTMLElement>(".home-card"))));
    return () => cancelAnimationFrame(raf);
  }, [shown, s?.model, cur, mine]);

  const dev = s?.device && ({ mps: "Apple GPU", cuda: "GPU", cpu: "CPU" }[s.device] || s.device);
  const specs: [string, unknown][] = [["Parameters", s?.params && `${(s.params / 1e9).toFixed(1)}B`], ["Layers", s?.layers], ["Runs on", dev], ["Precision", s?.dtype]];
  const status: Record<HomeArt, JSX.Element | string> = {
    abliterate: cur?.active ? <><i className="hc-on" />Refusals erased at layer {cur.layer}</> : cur ? "Direction saved, switched off" : "",
    steer: mine ? `${mine} saved` : "",
    chat: "",
  };

  return (
    <section id="view-home" className="view home" data-view="home" hidden={hidden}>
      <header className="home-h">
        <h2 id="home-model">{s?.loaded ? modelName(s.model) : ""}</h2>
        <dl id="home-meta" className="home-meta">
          {s?.loaded && specs.filter(([, v]) => v).map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{String(v)}</dd></div>)}
        </dl>
      </header>
      <div className="home-cards">
        {CARDS.map((c) => (
          <button className="home-card" data-go={c.go} key={c.go} onClick={() => setTab(c.go)}>
            <canvas className="home-art" data-art={c.go} aria-hidden="true" />
            <span className="hc-b">
              <span className="hc-t"><span className="hc-ic" data-icon={c.go}>{ICONS[c.go]}</span>{c.title}</span>
              <span className="hc-d">{c.text}</span>
              <span className="hc-f"><kbd>{c.key}</kbd><span className="hc-s" id={`hc-${c.go}`}>{status[c.go]}</span></span>
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}
