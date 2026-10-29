#!/bin/bash
# Builds "Auto Interp.app": a thin bundle that runs this checkout with uv. Re-run after moving the folder.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/dist}"
APP="$DEST/Auto Interp.app"
UV="$(command -v uv)"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
"$UV" run --project "$ROOT" --with pillow python "$ROOT/scripts/make_icon.py" "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Auto Interp</string>
  <key>CFBundleDisplayName</key><string>Auto Interp</string>
  <key>CFBundleIdentifier</key><string>local.auto-interp</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleExecutable</key><string>AutoInterp</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST

cat > "$APP/Contents/MacOS/AutoInterp" <<SH
#!/bin/bash
cd "$ROOT"
mkdir -p "\$HOME/Library/Logs"
exec "$UV" run --project "$ROOT" python -m interp.app >> "\$HOME/Library/Logs/AutoInterp.log" 2>&1
SH
chmod +x "$APP/Contents/MacOS/AutoInterp"
echo "built: $APP"
