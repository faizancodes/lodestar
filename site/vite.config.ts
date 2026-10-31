import { defineConfig } from "vite";

// Share cards need absolute URLs, so %SITE% in the page becomes the address it's served from: SITE_URL at build
// (e.g. SITE_URL=https://lodestar.example pnpm build), or the dev server's own.
const SITE = (process.env.SITE_URL ?? "http://localhost:5180").replace(/\/$/, "");

// The landing page. Its ASCII scenes are the app's own, imported from ../web/src/art, so the site and the app
// draw the same light.
export default defineConfig({
  base: "./",
  server: { port: 5180, fs: { allow: [".."] } },
  plugins: [{ name: "site-url", transformIndexHtml: (html) => html.replaceAll("%SITE%", SITE) }],
});
