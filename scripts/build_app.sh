#!/bin/bash
# Builds a self-contained "Lodestar.app" and "Lodestar.zip" in dist/ (or $1): the code, the built UI and a uv binary.
# Its first launch installs Python and the packages into ~/Library/Application Support/Lodestar, so nothing else
# (git, uv, pnpm, Python) has to be on the Mac. The release workflow runs this; so can you.
#
#     ./scripts/build_app.sh [dest] [version]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/dist}"
VERSION="${2:-$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml")}"
UV_VERSION=0.13.6
APP="$DEST/Lodestar.app"
RES="$APP/Contents/Resources"

# the UI the app serves (interp/server.py)
(cd "$ROOT/web" && pnpm install --frozen-lockfile && pnpm build)

rm -rf "$APP" "$DEST/Lodestar.zip"
mkdir -p "$APP/Contents/MacOS" "$RES/lodestar/web" "$RES/bin"

# the code and the lockfile, which the first launch installs from
rsync -a --exclude __pycache__ "$ROOT/interp" "$RES/lodestar/"
cp -R "$ROOT/web/dist" "$RES/lodestar/web/dist"
cp "$ROOT/pyproject.toml" "$ROOT/uv.lock" "$RES/lodestar/"

curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-aarch64-apple-darwin.tar.gz" |
  tar -xz -C "$RES/bin" --strip-components 1 uv-aarch64-apple-darwin/uv

uv run --no-project --python 3.12 --with pillow python "$ROOT/scripts/make_icon.py" "$RES/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Lodestar</string>
  <key>CFBundleDisplayName</key><string>Lodestar</string>
  <key>CFBundleIdentifier</key><string>so.lodestar.app</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleExecutable</key><string>Lodestar</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST

cat > "$APP/Contents/MacOS/Lodestar" <<'SH'
#!/bin/bash
# Starts Lodestar, first bringing its Python environment in line with the bundled lockfile (a no-op once it is).
# `Lodestar --setup` only does the setup, in the terminal, so install.sh can show the download.
CONTENTS="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$CONTENTS/Resources/lodestar"
UV="$CONTENTS/Resources/bin/uv"
SUPPORT="$HOME/Library/Application Support/Lodestar"
LOG="$HOME/Library/Logs/Lodestar.log"
export UV_PROJECT_ENVIRONMENT="$SUPPORT/venv" UV_PYTHON_INSTALL_DIR="$SUPPORT/python" UV_CACHE_DIR="$SUPPORT/cache"
export UV_PYTHON=3.12 UV_PYTHON_PREFERENCE=only-managed UV_NO_CONFIG=1 PYTHONPYCACHEPREFIX="$SUPPORT/pycache"
mkdir -p "$SUPPORT" "$(dirname "$LOG")"

setup() { "$UV" sync --frozen --project "$SRC"; }

if [ "${1:-}" = "--setup" ]; then
  setup
  exit
fi

if [ ! -x "$SUPPORT/venv/bin/python" ]; then
  osascript -e 'display notification "Installing Python and its packages (about 1 GB). Lodestar opens when that’s done." with title "Setting up Lodestar"'
fi
# offline with everything already installed, a failed check is fine; with nothing installed it isn't
if ! setup >>"$LOG" 2>&1 && [ ! -x "$SUPPORT/venv/bin/python" ]; then
  osascript -e 'display alert "Lodestar couldn’t finish setting up" message "Check your internet connection and open it again. Details are in ~/Library/Logs/Lodestar.log."'
  exit 1
fi

# run through a link named Lodestar so the Dock and Activity Monitor show that rather than "python"
ln -sf python "$SUPPORT/venv/bin/Lodestar"
cd "$SRC"
LODESTAR_ICON="$CONTENTS/Resources/AppIcon.icns" exec "$SUPPORT/venv/bin/Lodestar" -m interp.app "$@" >>"$LOG" 2>&1
SH
chmod +x "$APP/Contents/MacOS/Lodestar"

(cd "$DEST" && ditto -c -k --keepParent Lodestar.app Lodestar.zip)
echo "built: $APP and $DEST/Lodestar.zip ($VERSION)"
