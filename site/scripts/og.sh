#!/bin/sh
# Renders og.html (the share card) to public/og.png at 1200 × 630, with the dev server running (pnpm dev).
set -e
cd "$(dirname "$0")/.."
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless=new --hide-scrollbars --force-device-scale-factor=1 --window-size=1200,630 \
  --virtual-time-budget=4000 --screenshot="$PWD/public/og.png" "http://localhost:5180/og.html"
