import { useEffect } from "react";
import { Sidebar } from "./components/Sidebar";
import { useApp } from "./store/app";

export function App() {
  const tab = useApp((s) => s.tab);
  const busy = useApp((s) => s.busy);
  useEffect(() => { document.body.dataset.tab = tab; }, [tab]);
  useEffect(() => { document.body.classList.toggle("busy", busy); }, [busy]);
  return (
    <>
      <Sidebar />
      <main id="main" />
    </>
  );
}
