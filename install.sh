#!/bin/bash
# Installs Lodestar on an Apple silicon Mac:
#
#     curl -fsSL https://lodestar.so/install.sh | bash
#     curl -fsSL https://lodestar.so/install.sh | bash -s hive      # then open an experiment
#
# Downloads the latest release into /Applications (or ~/Applications), sets up its Python packages, and opens it.
# Run it again to update. Given an experiment (one that ships, like hive, or a link to a .lodestar.json file), the app
# opens it: it loads the model it was made on, turns it on and asks its first question.
set -euo pipefail
EXPERIMENT="${1:-}"

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "Lodestar needs a Mac with Apple silicon (M1 or later)." >&2
  exit 1
fi

URL="${LODESTAR_URL:-https://github.com/faizancodes/lodestar/releases/latest/download/Lodestar.zip}"
DEST="${LODESTAR_DIR:-/Applications}"
[ -w "$DEST" ] || DEST="$HOME/Applications"
mkdir -p "$DEST"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading Lodestar…"
curl -fL --progress-bar "$URL" -o "$tmp/Lodestar.zip"
ditto -x -k "$tmp/Lodestar.zip" "$tmp"

# an update replaces the running app, so quit it first: the launcher while it sets up, then the Python it hands over to
RUNNING="Application Support/Lodestar/venv/bin/Lodestar -m interp.app"
pkill -f "$DEST/Lodestar.app/Contents/" 2>/dev/null || true
pkill -f "$RUNNING" 2>/dev/null || true
for _ in $(seq 50); do pgrep -f "$RUNNING" >/dev/null || break; sleep 0.1; done
rm -rf "$DEST/Lodestar.app"
mv "$tmp/Lodestar.app" "$DEST/"
/usr/bin/xattr -dr com.apple.quarantine "$DEST/Lodestar.app" >/dev/null 2>&1 || true  # macOS's own: a pip "xattr" earlier on PATH has no -r

echo "Setting up Python and its packages (about 1 GB the first time)…"
"$DEST/Lodestar.app/Contents/MacOS/Lodestar" --setup

echo "Installed $DEST/Lodestar.app"
if [ -n "$EXPERIMENT" ] && [[ ! "$EXPERIMENT" =~ ^https?:// ]]; then
  REMOTE="https://raw.githubusercontent.com/faizancodes/lodestar/main/experiments/${EXPERIMENT}.lodestar.json"
  if curl -fsI "$REMOTE" >/dev/null 2>&1; then
    EXPERIMENT="$REMOTE"
  fi
fi
[ -n "${LODESTAR_NO_OPEN:-}" ] || open "$DEST/Lodestar.app" ${EXPERIMENT:+--args --open "$EXPERIMENT"}
