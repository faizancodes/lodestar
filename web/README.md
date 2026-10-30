# Lodestar UI in React + TypeScript

The app's interface, served by the engine from `dist/`. It was ported from the original plain-JS UI with the same DOM
and stylesheet, and is checked against the parity suite in `../e2e`.

    pnpm install
    pnpm dev                    # http://localhost:5173, /api proxied to the engine on :8792
    pnpm build                  # type-checks and writes dist/
    pnpm test                   # helpers, stream reader, decoder and canvas art on recorded data

Run the engine for `pnpm dev` with `uv run python -m interp.app --server` from the repo root.

Check it against the goldens:

    cd ../e2e && npx playwright test
