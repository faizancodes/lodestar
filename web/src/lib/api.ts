/** Extract a human-readable message from an unknown thrown value. */
export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** A JSON request to the engine. GETs when `body` is omitted; a non-2xx answer throws with the server's message. */
export async function api<T = unknown>(path: string, body?: unknown, method?: string): Promise<T> {
  const hasBody = body !== undefined;
  const r = await fetch(path, {
    method: method || (hasBody ? "POST" : "GET"),
    headers: hasBody ? { "Content-Type": "application/json" } : {},
    body: hasBody ? JSON.stringify(body) : undefined,
  });

  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || j.detail || r.statusText || `HTTP ${r.status}`);
  return j as T;
}

/** Request cancellation of the currently running engine stream, swallowing transport errors. */
export function cancelRun(): void {
  void api("/api/cancel", {}).catch(() => {});
}
