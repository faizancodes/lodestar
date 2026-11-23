#!/bin/bash
# Installs Lodestar on an Apple silicon Mac:
#
#     curl -fsSL https://lodestar.so/install.sh | bash
#
# Downloads the latest release into /Applications (or ~/Applications), sets up its Python packages, and opens it.
# Run it again to update.
set -euo pipefail

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

pkill -f "$DEST/Lodestar.app/Contents/" 2>/dev/null || true  # an update replaces the running app
rm -rf "$DEST/Lodestar.app"
mv "$tmp/Lodestar.app" "$DEST/"
xattr -dr com.apple.quarantine "$DEST/Lodestar.app" 2>/dev/null || true

echo "Setting up Python and its packages (about 1 GB the first time)…"
"$DEST/Lodestar.app/Contents/MacOS/Lodestar" --setup

echo "Installed $DEST/Lodestar.app"
[ -n "${LODESTAR_NO_OPEN:-}" ] || open "$DEST/Lodestar.app"
