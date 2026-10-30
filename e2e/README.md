# UI parity suite

Golden screenshots and request checks for every view and state of the UI. It exists so the UI can be rebuilt (in
React + TypeScript) and proven to look and behave the same.

- `fixtures/`: every `/api` exchange from a real session with Qwen3-0.6B, one file per scenario.
- `goldens/`: what the current UI looks like at each step, in WebKit (what the Mac app renders with).
- `harness/mock.py`: replays the fixtures, so the suite runs without a model in about a minute. It also fails a test
  if the UI sends a request that wasn't recorded or with a different body.
- `harness/record.py`: the real server with recording turned on.

## Run

    npm install && npx playwright install webkit   # once
    npx playwright test                              # replay against ../ui
    UI_DIR=../web/dist npx playwright test           # replay against the React build (`pnpm build` in ../web)

Canvases (the scenes, spinners and art), the plan clock, the load timer and toasts are hidden in the shots. Everything
else must match pixel for pixel: a single changed letter is about 20 pixels, so there's no pixel tolerance.

## When the API or a flow changes

    ./record.sh                         # re-record every scenario from Qwen3-0.6B, about 6 minutes
    npx playwright test --update-snapshots

`record.sh` runs the server on a throwaway data folder (`.data/`), so your recipes, directions and models are
untouched. Scenarios build on each other while recording (the model loaded at the start, the recipe saved in Steer,
the refusal direction found in Abliterate), so record them all together.
