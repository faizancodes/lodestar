# Lodestar UI in React + TypeScript

The port of `ui/` to React, in progress. It keeps `ui/style.css` and the DOM exactly as they are, and is checked
against the parity suite in `../e2e`.

    pnpm install
    pnpm dev                    # http://localhost:5173, /api proxied to the engine on :8792
    pnpm build                  # type-checks and writes dist/

Run the engine for `pnpm dev` with `uv run python -m interp.app --server` from the repo root. To serve the build from
the engine itself, start it with `LODESTAR_UI=react`.

Check it against the current UI:

    cd ../e2e && UI_DIR=../web/dist npx playwright test
