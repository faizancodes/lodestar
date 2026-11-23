"""Lodestar as a Mac app: the API server on a background thread, the UI in a native WebKit window."""
import os
import socket
import sys
import threading
import time

import uvicorn


def free_port(preferred=8792):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as uvicorn binds, so a restart's TIME_WAIT doesn't push it off
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def serve(port):
    from .server import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main():
    preferred = int(os.environ.get("AUTO_INTERP_PORT", 8792))

    # the window can live on any port; headless mode keeps the one asked for so bookmarks and tools find it
    port = preferred if "--server" in sys.argv else free_port(preferred)
    threading.Thread(target=serve, args=(port,), daemon=True).start()
    url = f"http://127.0.0.1:{port}/"

    for _ in range(200):  # wait for the server before showing the window
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.05)

    if "--server" in sys.argv:  # headless: just the API and UI on localhost
        print(url, flush=True)
        threading.Event().wait()

    import webview
    window = webview.create_window("Lodestar", url, width=1440, height=920, min_size=(1100, 700),
                                   background_color="#0A0A0A", text_select=True)
    window.events.shown += lambda: (_style_titlebar(window), _forward_paste())
    webview.start()


def _forward_paste():
    """Tell the page when ⌘V is pressed. The Edit menu takes ⌘V before WebKit sees it, and outside a text field WebKit
    drops the paste, so without this the page can't open an experiment link pasted there (web/src/App.tsx)."""
    try:
        import AppKit
        from webview.platforms.cocoa import BrowserView
    except Exception:
        return

    def on_key(ev):
        if ev.charactersIgnoringModifiers() == "v" and (ev.modifierFlags() & AppKit.NSEventModifierFlagDeviceIndependentFlagsMask) == AppKit.NSEventModifierFlagCommand:
            for bv in list(BrowserView.instances.values()):
                bv.webview.evaluateJavaScript_completionHandler_('document.dispatchEvent(new Event("lodestar-paste"))', None)
        return ev  # the paste itself still happens, in a text field

    def install():
        AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(AppKit.NSEventMaskKeyDown, on_key)

    AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(install)


def _style_titlebar(window):
    """Give the window a near-black title bar that matches the app instead of the default grey one, and hide the
    title text. pywebview's WebKit window won't let the page draw under the bar, so the UI keeps a dark strip at the top
    (the sidebar and content start below it) and this only has to colour the bar to match. Main thread required."""
    try:
        import AppKit
    except Exception:
        return

    def apply():
        icon = os.environ.get("LODESTAR_ICON")  # set by the app bundle, whose icon Python can't otherwise know
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

        # the title bar still shows a vibrancy material over the window colour; hide that backdrop so the
        # window's own background shows there instead. The traffic-light buttons (_NSThemeWidget) are untouched.
        def hide_backdrops(view, in_titlebar=False):
            name = view.className()
            here = in_titlebar or "Titlebar" in name
            if here and (name == "NSVisualEffectView" or "Decoration" in name):
                view.setHidden_(True)
            for sub in view.subviews():
                hide_backdrops(sub, here)

        cv = win.contentView()
        root = cv.superview() if cv else None
        if root is not None:
            hide_backdrops(root)

    AppKit.NSOperationQueue.mainQueue().addOperationWithBlock_(apply)


if __name__ == "__main__":
    main()
