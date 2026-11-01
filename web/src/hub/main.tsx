import "@fontsource/inter/latin-400.css";
import "@fontsource/inter/latin-500.css";
import "@fontsource/inter/latin-600.css";
import "@fontsource/jetbrains-mono/latin-400.css";
import "@fontsource/jetbrains-mono/latin-500.css";
import "../styles.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { HubPage } from "./HubPage";

// Hub entry (`vite build --mode hub`): no router, graph or model code; data is inlined.
const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <HubPage data={window.__INFERATLAS__} />
    </StrictMode>,
  );
}
