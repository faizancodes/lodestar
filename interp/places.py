"""Where things live on each OS: the data folder, and the external drives models can sit on."""
import os
import sys
from pathlib import Path

import psutil

MAC, WINDOWS = sys.platform == "darwin", sys.platform == "win32"


def data_dir():
    """The app's own folder. On a Mac it keeps the app's first name, so models, recipes and directions saved before
    the rename stay put."""
    if "AUTO_INTERP_DATA" in os.environ:
        return Path(os.environ["AUTO_INTERP_DATA"])
    if MAC:
        return Path.home() / "Library" / "Application Support" / "Auto Interp"
    if WINDOWS:
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Lodestar"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "lodestar"


def log_file():
    """Where the app writes its output when no console shows it: ~/Library/Logs on a Mac (scripts/build_app.sh sends it
    there), %LOCALAPPDATA%\\Lodestar on Windows."""
    if MAC:
        d = Path.home() / "Library" / "Logs"
    elif WINDOWS:
        d = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Lodestar"
    else:
        d = data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "Lodestar.log"


def drives():
    """The roots of external drives: /Volumes/<drive> on a Mac; on Windows, every drive letter but the system one;
    elsewhere, what's mounted under /media and /mnt."""
    if MAC:  # "Macintosh HD" there is a link back to the system disk
        return sorted(p for p in Path("/Volumes").glob("*") if p.is_dir() and p.resolve() != Path("/"))
    if WINDOWS:
        system = Path(os.environ.get("SystemDrive", "C:") + "\\")
        try:
            parts = psutil.disk_partitions(all=False)
        except OSError:
            return []
        return [Path(p.mountpoint) for p in parts if Path(p.mountpoint) != system and "cdrom" not in p.opts]
    roots = []
    for base in (Path("/media"), Path("/mnt")):
        roots += [p for p in base.glob("*") if p.is_dir()] + [p for p in base.glob("*/*") if p.is_dir()]
    return sorted(set(roots))


def drive_name(root):
    """What to call a drive in the UI: its folder name on a Mac, its letter (D:) on Windows."""
    return (root.drive or root.name) if WINDOWS else root.name


def drive_of(path):
    """The external drive a path is on, or None when it's on the system disk."""
    path = Path(path).resolve()
    return next((r for r in drives() if path.is_relative_to(r.resolve())), None)
