"""Draws Lodestar's app icon, the compass star from ui/mark.js in ASCII marks on a near-black tile, saved as .icns.
Each size is drawn on its own grid (coarser when small) rather than shrunk from the largest, so the marks stay
marks in the Dock and the menu bar."""
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

TILE, EDGE, INK = (11, 11, 11), (255, 255, 255, 26), (195, 207, 245)
SPAN = 31  # the star's design grid, as in ui/mark.js


def ray(along, across, length, w):
    return 0 if along < 0 or along > length else math.exp(-abs(across) / w) * (1 - along / length) ** 0.8


def star(x, y, up, side, down, diag):
    r = math.hypot(x, y)
    diags = [0.72 * ray((a * x + b * y) / math.sqrt(2), (a * x - b * y) / math.sqrt(2), diag, 0.45)
             for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    return max(1 if r < 1.5 else 0, 0.55 * math.exp(-r / 2.4), ray(-y, x, up, 0.55), ray(y, x, down, 0.55),
               ray(x, y, side, 0.55), ray(-x, y, side, 0.55), *diags)


def compass_star(x, y):
    r = math.hypot(x, y)
    ring = 0.24 if abs(r - 12.5) < 0.5 and math.floor(math.atan2(y, x) * 12 + 0.5) % 2 == 0 else 0
    ticks = 0.85 if abs(r - 12.5) < 1.2 and (abs(y) < 0.6 or (abs(x) < 0.6 and y > 0)) else 0  # none at north
    return max(star(x, y, 15.5, 8, 8, 6.5), ring, ticks)


def draw(px):
    ss = 4  # draw large, then shrink, for clean edges
    s = px * ss
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    m, rad = s * 0.1, s * 0.18  # the macOS icon grid: a rounded tile inset from the canvas
    d.rounded_rectangle([m, m, s - m, s - m], radius=rad, fill=TILE + (255,), outline=EDGE, width=max(1, s // 256))
    n = 15 if px <= 64 else 23 if px <= 128 else SPAN
    size = s * 0.56
    S, k, ox = size / n, SPAN / n, (s - size) / 2
    for j in range(n):
        for i in range(n):
            b = min(1, max(compass_star((i + (u + 0.5) / 3) * k - SPAN / 2, (j + (v + 0.5) / 3) * k - SPAN / 2)
                           for u in range(3) for v in range(3)))
            if b < 0.07:
                continue
            cx, cy, a, w = ox + (i + 0.5) * S, ox + (j + 0.5) * S, S * 0.42, max(ss, S * 0.2)
            if b < 0.3:
                h = S * (0.18 + 0.5 * b) / 2
                d.rectangle([cx - h, cy - h, cx + h, cy + h], fill=INK)
            elif b < 0.55:
                c = INK + (204,)
                d.line([cx - a, cy, cx + a, cy], fill=c, width=round(w))
                d.line([cx, cy - a, cx, cy + a], fill=c, width=round(w))
            elif b < 0.8:
                q = a * 0.75
                d.line([cx - q, cy - q, cx + q, cy + q], fill=INK, width=round(w * 1.15))
                d.line([cx - q, cy + q, cx + q, cy - q], fill=INK, width=round(w * 1.15))
            else:
                h = S * 0.42
                d.rectangle([cx - h, cy - h, cx + h, cy + h], fill=INK)
    return im.resize((px, px), Image.LANCZOS)


def main(out):
    out = Path(out)
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for px in (16, 32, 64, 128, 256, 512, 1024):
            img = draw(px)
            if px <= 512:
                img.save(iconset / f"icon_{px}x{px}.png")
            if px >= 32:
                img.save(iconset / f"icon_{px // 2}x{px // 2}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True)


if __name__ == "__main__":
    if len(sys.argv) > 2:  # preview: write one size as a png
        draw(int(sys.argv[2])).save(sys.argv[1])
    else:
        main(sys.argv[1])
