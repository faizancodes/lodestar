import { useEffect, useState } from "react";
import { useApp } from "../store/app";

export function Toast() {
  const t = useApp((s) => s.toast);
  const [on, setOn] = useState(false);
  useEffect(() => {
    if (!t) return;
    setOn(true);
    const timer = setTimeout(() => setOn(false), t.err ? 5200 : 2400);
    return () => clearTimeout(timer);
  }, [t]);
  return <div id="toast" className={on ? (t!.err ? "toast on err" : "toast on") : "toast"}>{t?.msg}</div>;
}
