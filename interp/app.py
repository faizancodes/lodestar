"""Lodestar desktop entry point: runs the local API server on a background thread with a native WebKit window."""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote

import uvicorn

from .utils import get_logger


logger = get_logger(__name__)

DEFAULT_PORT = 8792
_READY_TIMEOUT_STEPS = 200
_READY_POLL_SECONDS = 0.05


def free_port(preferred: int = DEFAULT_PORT) -> int:
    """Return ``preferred`` if bindable on localhost, otherwise an OS-assigned ephemeral port."""
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def serve(port: int) -> None:
    """Run the FastAPI server bound to ``127.0.0.1:port``."""
    from .server import app

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _wait_for_server(port: int) -> bool:
    for _ in range(_READY_TIMEOUT_STEPS):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return True
        except OSError:
            time.sleep(_READY_POLL_SECONDS)
    return False


def main(argv: Sequence[str] | None = None) -> None:
    """Launch Lodestar either headless (``--server``) or inside a native macOS WebKit window."""
    parser = argparse.ArgumentParser(description="Run Lodestar locally.")
    parser.add_argument("--server", action="store_true", help="Run headless HTTP API/UI without opening a native window.")
    parser.add_argument("--open", dest="open_target", default=None, help="Experiment name or URL to open on launch.")
    args = parser.parse_args(argv)

    preferred = int(os.environ.get("AUTO_INTERP_PORT", DEFAULT_PORT))
    port = preferred if args.server else free_port(preferred)
    threading.Thread(target=serve, args=(port,), daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    if args.open_target:
        url += "?open=" + quote(args.open_target, safe="")

    if not _wait_for_server(port):
        raise RuntimeError(f"Lodestar server did not start on {url} within {_READY_TIMEOUT_STEPS * _READY_POLL_SECONDS:.0f}s.")

    if args.server:
        print(url, flush=True)
        threading.Event().wait()

    import webview

    window = webview.create_window(
        "Lodestar",
        url,
        width=1440,
        height=920,
        min_size=(1100, 700),
        background_color="#0A0A0A",
        text_select=True,
    )

    def _on_shown() -> None:
        _style_titlebar(window)
        _forward_paste()

    window.events.shown += _on_shown
    webview.start()


def _forward_paste() -> None:
    """Tell the page when Cmd+V is pressed outside a text field so pasted experiment links open cleanly."""
    try:
        import AppKit
        from webview.platforms.cocoa import BrowserView
    except Exception:
        return

    def _on_key(ev: Any) -> Any:
        flags = ev.modifierFlags() & AppKit.NSEventModifierFlagDeviceIndependentFlagsMask
        if ev.charactersIgnoringModifiers() == "v" and flags == AppKit.NSEventModifierFlagCommand:
            for view in list(BrowserView.instances.values()):
                view.webview.evaluateJavaScript_completionHandler_(
                    'document.dispatchEvent(new Event("lodestar-paste"))', None
                )
        return ev

    def _install() -> None:
        AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(AppKit.NSEventMaskKeyDown, _on_key)

    AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(_install)


def _style_titlebar(window: Any) -> None:
    """Match macOS's native title bar to Lodestar's dark canvas and hide the title text on the main thread."""
    try:
        import AppKit
    except Exception:
        return

    def _apply() -> None:
        icon = os.environ.get("LODESTAR_ICON")
        if icon:
            AppKit.NSApp.setApplicationIconImage_(AppKit.NSImage.alloc().initWithContentsOfFile_(icon))

        win = getattr(window, "native", None)
        if win is None:
            return

        dark = AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameDarkAqua)
        win.setAppearance_(dark)
        win.setTitlebarAppearsTransparent_(True)
        win.setTitleVisibility_(AppKit.NSWindowTitleHidden)
        win.setBackgroundColor_(AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(0x0A / 255, 0x0A / 255, 0x0A / 255, 1.0))

        def _hide_backdrops(view: Any, in_titlebar: bool = False) -> None:
            name = view.className()
            here = in_titlebar or "Titlebar" in name
            if here and (name == "NSVisualEffectView" or "Decoration" in name):
                view.setHidden_(True)
            for sub in view.subviews():
                _hide_backdrops(sub, here)

        content_view = win.contentView()
        root = content_view.superview() if content_view else None
        if root is not None:
            _hide_backdrops(root)

    AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(_apply)


if __name__ == "__main__":
    main()
