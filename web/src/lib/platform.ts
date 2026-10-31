// The window is WebKit on a Mac and Edge's WebView2 on Windows; shortcuts and wording follow the machine it's on.
export const MAC = /Mac|iPhone|iPad/.test(navigator.userAgent);
export const MOD = MAC ? "⌘" : "Ctrl+"; // as shortcuts are written: ⌘1, Ctrl+1
export const HERE = MAC ? "this Mac" : "this PC";

/** The shortcut modifier is down: Command on a Mac, Ctrl elsewhere. */
export const modDown = (e: KeyboardEvent) => (MAC ? e.metaKey : e.ctrlKey);
