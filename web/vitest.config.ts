import { defineConfig } from "vitest/config";

// The parity tests load the original modules from ../ui and the recordings from ../e2e
export default defineConfig({ server: { fs: { allow: [".."] } }, test: { include: ["test/**/*.test.ts"] } });
