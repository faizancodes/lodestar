import { useEffect, useState } from "react";
import { useApp } from "../store/app";

export function Toast() {
  const t = useApp((s) => s.toast);
  const [gone, setGone] = useState(0); // the last toast whose time ran out
  const on = !!t && t.n !== gone;

  useEffect(() => {
    if (!t) return;
    const timer = setTimeout(() => setGone(t.n), t.err ? 5200 : 2400);
    return () => clearTimeout(timer);
  }, [t]);

  return <div id="toast" className={on ? (t!.err ? "toast on err" : "toast on") : "toast"}>{t?.msg}</div>;
}
