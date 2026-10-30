import { useState } from "react";
import { gb } from "../lib/format";
import { cancelCopy, removeCopy, startCopy, type Choice } from "../shell/engine";
import { useApp } from "../store/app";

// the action beside a model: copy a drive model onto this Mac (or stop that copy), or remove a copy
export function CopyAction({ x }: { x: Choice }) {
  const copy = useApp((s) => s.copy);
  const [sure, setSure] = useState(false);
  const act = (f: () => unknown) => (e: React.MouseEvent) => (e.stopPropagation(), f());

  if (x.drive) {
    const busy = copy?.status === "copying";
    if (busy && copy!.src === x.id) return <button className="mact" data-act="stop" onClick={act(cancelCopy)}>Stop copy</button>;
    return busy ? null : (
      <button className="mact" data-act="copy" title="The internal disk reads about 8× faster than a USB drive" onClick={act(() => startCopy(x.id))}>
        Copy to Mac · {gb(x.drive.size)}
      </button>
    );
  }

  return x.copy ? (
    <button className="mact" data-act="remove" data-sure={sure ? "1" : undefined} onClick={act(() => (sure ? removeCopy(x.id) : setSure(true)))}>
      {sure ? "Remove? Click again" : "Remove copy"}
    </button>
  ) : null;
}
