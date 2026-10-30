#!/bin/bash
# Re-records every fixture from a real model: starts the recording server on a fresh, throwaway data folder, runs the
# suite in record mode, and stops the server. Takes a while (it really searches, abliterates and exports).
#   ./record.sh                 # everything
#   ./record.sh -g steer        # extra arguments go to playwright
set -euo pipefail
cd "$(dirname "$0")"

rm -rf .data
mkdir -p .data fixtures

AUTO_INTERP_DATA="$PWD/.data" uv run --project .. python harness/record.py 8795 > .record.log 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT

until curl -sf -o /dev/null http://127.0.0.1:8795/api/status; do
  kill -0 $pid 2>/dev/null || { cat .record.log; exit 1; }
  sleep 1
done

MODE=record npx playwright test "$@"
