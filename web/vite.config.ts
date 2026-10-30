import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the engine runs separately (`uv run python -m interp.app --server`) and /api goes to it. The proxy
// passes each chunk on as it arrives, so the SSE streams stay streams.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { target: "safari15" }, // pywebview renders with the system WebKit; macOS 13 ships Safari 16
  server: {
    port: 5173,
    proxy: {
      "/api": `http://127.0.0.1:${process.env.AUTO_INTERP_PORT || 8792}`,
    },
  },
});
