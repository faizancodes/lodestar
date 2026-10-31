import { defineConfig } from "vite";

// The landing page. Its ASCII scenes are the app's own, imported from ../web/src/art, so the site and the app
// draw the same light.
export default defineConfig({
  base: "./",
  server: { port: 5180, fs: { allow: [".."] } },
});
