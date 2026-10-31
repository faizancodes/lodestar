# Builds Lodestar for Windows: the UI, an icon, and Start menu and desktop shortcuts that run this checkout with uv and
# no console window (output goes to %LOCALAPPDATA%\Lodestar\Lodestar.log). The Windows twin of build_app.sh; re-run
# after moving the folder.
#   powershell -ExecutionPolicy Bypass -File scripts\build_app.ps1
$ErrorActionPreference = "Stop"
$Root = Resolve-Path "$PSScriptRoot\.."
$Dist = Join-Path $Root "dist"
$Uv = (Get-Command uv).Source

# the UI the app serves (interp/server.py)
Push-Location (Join-Path $Root "web")
pnpm install --frozen-lockfile
pnpm build
Pop-Location

# the Python side, with the CUDA build of torch (pyproject.toml)
& $Uv sync --project $Root
New-Item -ItemType Directory -Force $Dist | Out-Null
$Icon = Join-Path $Dist "Lodestar.ico"
& $Uv run --project $Root --with pillow python (Join-Path $Root "scripts\make_icon.py") $Icon

$Shell = New-Object -ComObject WScript.Shell
$Menu = Join-Path ([Environment]::GetFolderPath("Programs")) "Lodestar.lnk"
$Desk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Lodestar.lnk"
foreach ($Path in @($Menu, $Desk)) {
    $Link = $Shell.CreateShortcut($Path)
    $Link.TargetPath = $Uv
    $Link.Arguments = "run --project `"$Root`" pythonw -m interp.app"
    $Link.WorkingDirectory = "$Root"
    $Link.IconLocation = $Icon
    $Link.WindowStyle = 7  # minimised: uv's own console stays out of sight
    $Link.Save()
}
Write-Host "built: $Menu"
Write-Host "       $Desk"
