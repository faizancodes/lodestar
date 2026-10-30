/** A JSON request to the engine. GETs without a body; a non-2xx answer throws with the server's message. */
export async function api<T = unknown>(path: string, body?: unknown, method?: string): Promise<T> {
  const r = await fetch(path, {
    method: method || (body ? "POST" : "GET"),
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });

  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || j.detail || r.statusText);
  return j as T;
}
