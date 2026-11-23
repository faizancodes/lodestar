// The state map: every prompt of a probe at one layer (interp/probe.py state_map), A in red and B in green, and a
// prompt being tested as a bright dot. Positions between layers are interpolated, so the layer can animate.

export type Pt = [number, number];
type Draw = { map: Pt[][]; sides: number[]; labels: string[]; layer: number; path: Pt[] | null; hot: boolean | null };

const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// a point between two layers: `get(k)` is its place at layer k
function at(get: (k: number) => Pt, n: number, l: number): Pt {
  const a = Math.floor(l), b = Math.min(a + 1, n - 1), t = l - a, p = get(a), q = get(b);
  return [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
}

export function drawMap(cv: HTMLCanvasElement, d: Draw, now: number) {
  const dpr = devicePixelRatio || 1, w = (cv.width = cv.clientWidth * dpr), h = (cv.height = cv.clientHeight * dpr);
  const g = cv.getContext("2d")!;
  const R = 2.8; // standard deviations shown either side of the middle
  const X = (v: number) => w / 2 + (Math.max(-R, Math.min(R, v)) / R) * (w / 2 - 36 * dpr);
  const Y = (v: number) => h / 2 - (Math.max(-R, Math.min(R, v)) / R) * (h / 2 - 34 * dpr);
  const l = Math.max(0, Math.min(d.layer, d.map.length - 1));
  const red = css("--bad"), green = css("--good");

  g.globalAlpha = 0.7;
  d.sides.forEach((s, i) => {
    const [x, y] = at((k) => d.map[k][i], d.map.length, l), z = 4.2 * dpr;
    g.fillStyle = s ? red : green;
    g.fillRect(X(x) - z / 2, Y(y) - z / 2, z, z);
  });

  g.globalAlpha = 1;
  g.font = `500 ${12 * dpr}px "Geist Mono", ui-monospace, monospace`;
  g.textBaseline = "top";
  g.fillStyle = css("--muted");
  g.fillText(`layer ${Math.round(l)}`, 16 * dpr, 14 * dpr);

  g.font = `500 ${12.5 * dpr}px Geist, system-ui, sans-serif`;
  g.textBaseline = "bottom";
  g.fillStyle = green;
  g.fillText(d.labels[1], 16 * dpr, h - 12 * dpr);
  g.fillStyle = red;
  g.textAlign = "right";
  g.fillText(d.labels[0], w - 16 * dpr, h - 12 * dpr);
  g.textAlign = "left";

  if (!d.path) return;
  const path = d.path, [x, y] = at((k) => path[k], path.length, l), cx = X(x), cy = Y(y);
  const col = d.hot == null ? "#ffffff" : d.hot ? red : green;
  g.strokeStyle = col;
  g.globalAlpha = 0.5;
  g.lineWidth = 2 * dpr;
  g.beginPath();
  g.arc(cx, cy, 13 * dpr * (1 + 0.22 * Math.sin(now / 120)), 0, 7);
  g.stroke();
  g.globalAlpha = 1;
  g.fillStyle = col;
  g.beginPath();
  g.arc(cx, cy, 6 * dpr, 0, 7);
  g.fill();
}
