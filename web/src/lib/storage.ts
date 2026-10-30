// What the app remembers between launches, under versioned keys so a later shape can't misread an older one. Storage
// can throw (private browsing, quota, disabled); then the choice just isn't remembered.
const PREFIX = "lodestar:v1:";

export function load(key: string): string | null {
  try {
    const v = localStorage.getItem(PREFIX + key);
    if (v !== null) return v;

    const old = localStorage.getItem(key); // from before the keys were versioned
    if (old !== null) (localStorage.setItem(PREFIX + key, old), localStorage.removeItem(key));
    return old;
  } catch {
    return null;
  }
}

export function save(key: string, value: string) {
  try { localStorage.setItem(PREFIX + key, value); } catch {}
}
