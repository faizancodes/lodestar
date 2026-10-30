import { loadModel, modelChoices, type Choice } from "../shell/engine";
import { useApp } from "../store/app";
import { CopyAction } from "../components/CopyAction";

export function Start() {
  const start = useApp((s) => s.start);
  const loaded = useApp((s) => !!s.info?.loaded);
  const models = useApp((s) => s.models);
  const copy = useApp((s) => s.copy);
  const c = modelChoices(models, copy);
  const card = (x: Choice, go: string) => (
    <div className={start.picked === x.id ? "start-model picked" : "start-model"} role="button" tabIndex={0} data-id={x.id} key={x.id}
      onClick={() => loadModel(x.id)} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && loadModel(x.id)}>
      <div><b>{x.name}</b><span>{x.sub}</span></div>
      <span className="acts"><CopyAction x={x} /><span className="go">{go}</span></span>
    </div>
  );
  const here = [...c.drives, ...c.here];
  return (
    <section id="start" className={"start" + (start.loading ? " loading" : "") + (start.leaving ? " leaving" : "")} hidden={loaded}>
      <h2>Pick a model to start.</h2>
      <div id="start-models" className="start-models">
        {here.length ? here.map((x) => card(x, "Load")) : c.get.slice(0, 3).map((x) => card(x, `Download ${x.tag}`))}
      </div>
      <canvas id="start-art" className="start-art" aria-hidden="true" />
      <div id="start-status" className="start-status">{start.status}</div>
    </section>
  );
}
