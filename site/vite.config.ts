import { defineConfig } from "vite";

// Share cards need absolute URLs, so %SITE% in the page becomes the address it's served from: lodestar.so, or
// SITE_URL to build it for somewhere else (a preview, say).
const SITE = (process.env.SITE_URL ?? "https://lodestar.so").replace(/\/$/, "");

// The landing page. Its ASCII scenes are the app's own, imported from ../web/src/art, so the site and the app
// draw the same light.
export default defineConfig({
  base: "./",
  server: { port: 5180, fs: { allow: [".."] } },
  plugins: [{ name: "site-url", transformIndexHtml: (html) => html.replaceAll("%SITE%", SITE) }],
});
