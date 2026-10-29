"""Draws the app icon: a cream sun on halftone cobalt with engraved hills, saved as an .icns."""
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

COBALT, DEEP, CREAM = (31, 73, 163), (24, 60, 140), (248, 239, 224)


def draw(size=1024):
    s = size
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([s * 0.1, s * 0.1, s * 0.9, s * 0.9], radius=s * 0.18, fill=255)
    tile = Image.new("RGBA", (s, s), COBALT + (255,))
    d = ImageDraw.Draw(tile)
    step = max(4, s // 128)
    for y in range(0, s, step):  # halftone dots
        for x in range(0, s, step):
            d.ellipse([x, y, x + step * 0.22, y + step * 0.22], fill=(248, 239, 224, 46))
    for k in range(18):  # engraved hills
        pts = []
        for x in range(0, s + 8, 8):
            u = x / s
            y = 0.70 - 0.13 * math.exp(-((u - 0.68) ** 2) / 0.02) - 0.07 * math.exp(-((u - 0.25) ** 2) / 0.012)
            pts.append((x, s * (y + k * 0.017 + 0.004 * math.sin(u * 30 + k))))
        d.line(pts, fill=CREAM + (int(120 + k * 7),), width=max(1, s // 300))
    cx, cy, r = s * 0.5, s * 0.40, s * 0.15  # a cream sun over the hills, ringed like a firing unit
    for k, a in ((2.0, 40), (1.55, 80)):
        d.ellipse([cx - r * k, cy - r * k, cx + r * k, cy + r * k], outline=CREAM + (a,), width=max(2, s // 120))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=CREAM)
    im.paste(tile, (0, 0), mask)
    return im


def main(out):
    out = Path(out)
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        base = draw(1024)
        for px in (16, 32, 64, 128, 256, 512, 1024):
            img = base.resize((px, px), Image.LANCZOS)
            if px <= 512:
                img.save(iconset / f"icon_{px}x{px}.png")
            if px >= 32:
                img.save(iconset / f"icon_{px // 2}x{px // 2}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True)


if __name__ == "__main__":
    main(sys.argv[1])
