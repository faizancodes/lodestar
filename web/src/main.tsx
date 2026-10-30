import { StrictMode } from "react";
import { createPortal } from "react-dom";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles/fonts/fonts.css";
import "./styles/style.css";

// The page is body > aside.side + main, and style.css lays it out as a flex row on <body>. Rendering through a
// portal into <body> keeps exactly that DOM, with no wrapper element between them.
createRoot(document.createElement("div")).render(
  <StrictMode>{createPortal(<App />, document.body)}</StrictMode>,
);
