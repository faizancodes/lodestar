/** Deterministic pseudo-random float in [0, 1) from numeric seeds, shared by canvas scenes. */
export function hash(...v: number[]): number {
  let h = 0;
  for (const x of v) h = Math.sin(h * 1.7 + x * 12.9898 + 78.233) * 43758.5453;
  return h - Math.floor(h);
}
