# Lodestar UI in React + TypeScript

The app's interface: the port of `ui/` to React. It keeps `ui/style.css` and the DOM exactly as they are, and is
checked against the parity suite in `../e2e`. The engine serves `dist/` (start it with `LODESTAR_UI=legacy` for the
original UI in `ui/`, kept for one release).

    pnpm install
    pnpm dev                    # http://localhost:5173, /api proxied to the engine on :8792
    pnpm build                  # type-checks and writes dist/
    pnpm test                   # runs ui/'s helpers, stream reader and canvas art beside the port on recorded data

Run the engine for `pnpm dev` with `uv run python -m interp.app --server` from the repo root.

Check it against the current UI:

    cd ../e2e && UI_DIR=../web/dist npx playwright test
