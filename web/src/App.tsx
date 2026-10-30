import { useEffect } from "react";
import { Sidebar } from "./components/Sidebar";
import { useApp } from "./store/app";

export function App() {
  const tab = useApp((s) => s.tab);
  useEffect(() => { document.body.dataset.tab = tab; }, [tab]);
  return (
    <>
      <Sidebar />
      <main id="main" />
    </>
  );
}
