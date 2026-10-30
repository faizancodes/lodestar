// Lodestar's mark: a compass star drawn in the app's ASCII marks. Its north ray breaks out through a dotted ring
// that holds ticks at the other three points. scripts/make_icon.py draws the same star for the app icon.
const SPAN = 31; // the star is designed on a 31×31 grid, centre at 15

const ray = (along: number, across: number, len: number, w: number) => (along < 0 || along > len ? 0 : Math.exp(-Math.abs(across) / w) * (1 - along / len) ** 0.8);

function star(x: number, y: number, up: number, side: number, down: number, diag: number) {
  const r = Math.hypot(x, y);
  return Math.max(
    r < 1.5 ? 1 : 0, 0.55 * Math.exp(-r / 2.4),
    ray(-y, x, up, 0.55), ray(y, x, down, 0.55), ray(x, y, side, 0.55), ray(-x, y, side, 0.55),
    ...[[1, 1], [1, -1], [-1, 1], [-1, -1]].map(([a, b]) => 0.72 * ray((a * x + b * y) / Math.SQRT2, (a * x - b * y) / Math.SQRT2, diag, 0.45)));
}

/** Brightness 0..1 of the mark at (x, y), both in -15..15. */
export function compassStar(x: number, y: number) {
  const r = Math.hypot(x, y);
  const ring = Math.abs(r - 12.5) < 0.5 && Math.round(Math.atan2(y, x) * 12) % 2 === 0 ? 0.24 : 0;
  const ticks = Math.abs(r - 12.5) < 1.2 && (Math.abs(y) < 0.6 || (Math.abs(x) < 0.6 && y > 0)) ? 0.85 : 0; // none at north
  return Math.max(star(x, y, 15.5, 8, 8, 6.5), ring, ticks);
}

/** The mark as an inline SVG string, px wide, on an n×n grid (fewer cells at small sizes keep it legible). */
export function markSVG(px: number, n = SPAN, color = "currentColor") {
  const S = px / n, k = SPAN / n;
  let out = "";

  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      let b = 0; // a coarse cell takes the brightest point it covers
      for (let v = 0; v < 3; v++)
        for (let u = 0; u < 3; u++) b = Math.max(b, compassStar((i + (u + 0.5) / 3) * k - SPAN / 2, (j + (v + 0.5) / 3) * k - SPAN / 2));
      if ((b = Math.min(1, b)) < 0.07) continue;

      const cx = (i + 0.5) * S, cy = (j + 0.5) * S, a = S * 0.42, w = Math.max(0.8, S * 0.2);
      if (b < 0.3) {
        const s = S * (0.18 + 0.5 * b);
        out += `<rect x="${cx - s / 2}" y="${cy - s / 2}" width="${s}" height="${s}" stroke="none"/>`;
      } else if (b < 0.55) out += `<path d="M${cx - a} ${cy}h${2 * a}M${cx} ${cy - a}v${2 * a}" stroke-width="${w}" opacity=".8"/>`;
      else if (b < 0.8) out += `<path d="M${cx - a * 0.75} ${cy - a * 0.75}l${a * 1.5} ${a * 1.5}M${cx - a * 0.75} ${cy + a * 0.75}l${a * 1.5} ${-a * 1.5}" stroke-width="${w * 1.15}"/>`;
      else {
        const s = S * 0.84;
        out += `<rect x="${cx - s / 2}" y="${cy - s / 2}" width="${s}" height="${s}" stroke="none"/>`;
      }
    }

  return `<svg width="${px}" height="${px}" viewBox="0 0 ${px} ${px}" fill="${color}" stroke="${color}" aria-hidden="true">${out}</svg>`;
}
