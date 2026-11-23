import { readFileSync } from "node:fs";
import { defineConfig } from "vite";

// Share cards need absolute URLs, so %SITE% in the page becomes the address it's served from: lodestar.so, or
// SITE_URL to build it for somewhere else (a preview, say).
const SITE = (process.env.SITE_URL ?? "https://lodestar.so").replace(/\/$/, "");

// The installer people pipe into bash (curl -fsSL https://lodestar.so/install.sh | bash) is the repo's own
// install.sh, served at the site's root, so there's one copy of it.
const INSTALL = new URL("../install.sh", import.meta.url);

// The landing page. Its ASCII scenes are the app's own, imported from ../web/src/art, so the site and the app
// draw the same light.
export default defineConfig({
  base: "./",
  server: { port: 5180, fs: { allow: [".."] } },
  build: { rollupOptions: { input: { index: "index.html", demo: "demo.html" } } },
  plugins: [
    { name: "site-url", transformIndexHtml: (html) => html.replaceAll("%SITE%", SITE) },
    {
      name: "install-script",
      configureServer(server) {
        server.middlewares.use("/install.sh", (_req, res) => {
          res.setHeader("content-type", "text/plain; charset=utf-8");
          res.end(readFileSync(INSTALL));
        });
      },
      generateBundle() {
        this.emitFile({ type: "asset", fileName: "install.sh", source: readFileSync(INSTALL, "utf8") });
      },
    },
  ],
});
