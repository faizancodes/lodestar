import { defineConfig } from "vitest/config";

// The tests read the recordings in ../e2e
export default defineConfig({ server: { fs: { allow: [".."] } }, test: { include: ["test/**/*.test.ts"] } });
