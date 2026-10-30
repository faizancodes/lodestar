"""Draws Lodestar's app icon and saves it as .icns: a long exposure of the night sky. Star trails, drawn in the app's
ASCII marks, wheel around the lodestar: a crisp four-point star (its north ray longest) with a soft periwinkle glow,
on a deep blue-black tile cut to macOS's rounded-square shape. Small sizes get their own simpler drawing (the star and a trail or two) so
they stay crisp in the menu bar and Finder rather than turning into a grey smudge.

  uv run --with pillow python scripts/make_icon.py out.icns
  uv run --with pillow python scripts/make_icon.py preview.png 512     # one size, as a png
"""
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

INK = (195, 207, 245)  # --ascii
TOP, BOTTOM, CORE = (16, 18, 30), (6, 6, 8), (28, 33, 58)  # the tile's sky: lighter near the star


def hash01(*v):
    h = 0.0
    for x in v:
        h = math.sin(h * 1.7 + x * 12.9898 + 78.233) * 43758.5453
    return h - math.floor(h)


def squircle(size, inset):
    """macOS's tile as a mask: a rounded square inset from the canvas edge, corners about 22% of its side."""
    m = Image.new("L", (size, size), 0)
    side = size - 2 * inset
    ImageDraw.Draw(m).rounded_rectangle([inset, inset, size - inset, size - inset], radius=side * 0.225, fill=255)
    return m


def ray(along, across, length, w):
    return 0.0 if along < 0 or along > length else math.exp(-abs(across) / w) * (1 - along / length) ** 0.8


def star(x, y, up, side, down, diag, w=0.6):
    """The lodestar: a bright core, four rays with the north one longest, glints on the diagonals."""
    r = math.hypot(x, y)
    diags = [0.6 * ray((a * x + b * y) / math.sqrt(2), (a * x - b * y) / math.sqrt(2), diag, w * 0.8)
             for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
    return max(1.0 if r < 1.2 else 0.0, 0.5 * math.exp(-r / 1.6), ray(-y, x, up, w), ray(y, x, down, w),
               ray(x, y, side, w), ray(-x, y, side, w), *diags)


def marks(draw, grid, n, cell, ox, oy):
    """Each cell as one of the app's marks by brightness: a dot, a plus, an ×, a block."""
    for j in range(n):
        for i in range(n):
            b = min(1.0, grid[j][i])
            if b < 0.06:
                continue
            cx, cy = ox + (i + 0.5) * cell, oy + (j + 0.5) * cell
            a, w = cell * 0.4, max(1, round(cell * 0.2))
            if b < 0.3:
                h = cell * (0.16 + 0.5 * b) / 2
                draw.rectangle([cx - h, cy - h, cx + h, cy + h], fill=INK + (int(255 * min(1, 0.45 + b)),))
            elif b < 0.55:
                c = INK + (215,)
                draw.line([cx - a, cy, cx + a, cy], fill=c, width=w)
                draw.line([cx, cy - a, cx, cy + a], fill=c, width=w)
            elif b < 0.8:
                q = a * 0.78
                draw.line([cx - q, cy - q, cx + q, cy + q], fill=INK + (255,), width=round(w * 1.2))
                draw.line([cx - q, cy + q, cx + q, cy - q], fill=INK + (255,), width=round(w * 1.2))
            else:
                h = cell * 0.43
                draw.rectangle([cx - h, cy - h, cx + h, cy + h], fill=INK + (255,))


def draw(px):
    ss = 2 if px >= 256 else 4
    s = px * ss
    small = px <= 64
    inset = s * 0.1  # Apple's grid: the tile sits inside a 10% margin
    tile = s - 2 * inset

    # the sky: a vertical fall from blue-black to black, lifted around the star
    sky = Image.new("RGB", (s, s))
    sd = ImageDraw.Draw(sky)
    for y in range(s):
        k = y / s
        sd.line([(0, y), (s, y)], fill=tuple(round(TOP[c] + (BOTTOM[c] - TOP[c]) * k) for c in range(3)))
    cx, cy = s / 2, s * 0.47
    halo = Image.new("L", (s, s), 0)
    ImageDraw.Draw(halo).ellipse([cx - tile * 0.42, cy - tile * 0.42, cx + tile * 0.42, cy + tile * 0.42], fill=255)
    halo = halo.filter(ImageFilter.GaussianBlur(tile * 0.18))
    sky = Image.composite(Image.new("RGB", (s, s), CORE), sky, halo.point(lambda v: v * 0.55))

    # the long exposure, in marks: trails around the lodestar, then the star itself
    n = 15 if small else 37
    cell = tile / n
    grid = [[0.0] * n for _ in range(n)]
    c0, r0 = (n - 1) / 2, (n - 1) / 2 - n * 0.03  # the lodestar sits a touch above centre

    def put(i, j, b):
        i, j = round(i), round(j)
        if 0 <= i < n and 0 <= j < n and b > grid[j][i]:
            grid[j][i] = b

    rings = [4.6, 6.4] if small else [6, 8, 10, 12, 14, 16]
    for k, r in enumerate(rings):
        for seg in range(1 if small else 2):
            head = 0.3 + k * 1.05 + seg * math.pi  # heads spread evenly around, every trail turning the same way
            length = 1.5 if small else 1.25 + hash01(k, seg, 2) * 0.7
            peak = 0.5 if small else 0.34 + hash01(k, seg, 3) * 0.24  # dots and pluses: the star leads
            u = 0.0
            while u <= length:
                put(c0 + math.cos(head - u) * r, r0 + math.sin(head - u) * r, peak * (1 - u / length) ** 1.3)
                u += 0.5 / r

    art = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    marks(ImageDraw.Draw(art), grid, n, cell, inset, inset)
    # the lodestar: a soft glow, then a crisp four-point star with fine diagonal glints
    gx, gy = inset + (c0 + 0.5) * cell, inset + (r0 + 0.5) * cell
    R = tile * (0.3 if small else 0.27)  # north ray length
    glow = Image.new("RGBA", (s, s), INK + (0,))
    gr = R * 0.55
    ImageDraw.Draw(glow).ellipse([gx - gr, gy - gr, gx + gr, gy + gr], fill=INK + (190,))
    glow = glow.filter(ImageFilter.GaussianBlur(gr * 0.7))
    sparkle = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sp = ImageDraw.Draw(sparkle)
    tips = [(0, -R), (R * 0.62, 0), (0, R * 0.7), (-R * 0.62, 0)]  # north longest
    waist = R * 0.075  # how far the concave sides pull toward the centre

    def quad(a, b, ctrl, steps=40):
        return [((1 - t) ** 2 * a[0] + 2 * (1 - t) * t * ctrl[0] + t * t * b[0],
                 (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * ctrl[1] + t * t * b[1]) for t in (i / steps for i in range(steps + 1))]

    pts = []
    for k in range(4):
        a, b = tips[k], tips[(k + 1) % 4]
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        m = math.hypot(*mid) or 1
        pts += quad(a, b, (mid[0] / m * waist, mid[1] / m * waist))
    if not small:
        for a in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4):  # glints between the rays
            L = R * 0.42
            sp.line([gx, gy, gx + math.cos(a) * L, gy + math.sin(a) * L], fill=INK + (190,), width=max(1, round(s / 400)))
    sp.polygon([(gx + x, gy + y) for x, y in pts], fill=(236, 240, 255, 255))
    core = Image.new("RGBA", (s, s), (255, 255, 255, 0))
    cr = R * 0.1
    ImageDraw.Draw(core).ellipse([gx - cr, gy - cr, gx + cr, gy + cr], fill=(255, 255, 255, 255))
    core = core.filter(ImageFilter.GaussianBlur(cr * 0.6))

    out = sky.convert("RGBA")
    out.alpha_composite(art)
    out.alpha_composite(glow)
    out.alpha_composite(sparkle)
    out.alpha_composite(core)
    # a fine highlight along the tile's top edge, as on a lit glass rim
    mask = squircle(s, inset)
    rim = ImageChops.subtract(mask, squircle(s, inset + max(1, s // 400)))
    edge = Image.new("RGBA", (s, s), (255, 255, 255, 0))
    fade = Image.linear_gradient("L").resize((s, s)).point(lambda v: int(70 * (1 - v / 255) ** 2))
    edge.putalpha(ImageChops.multiply(rim, fade))
    out.alpha_composite(edge)
    out.putalpha(mask)
    return out.resize((px, px), Image.Resampling.LANCZOS)


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
    if len(sys.argv) > 2:
        draw(int(sys.argv[2])).save(sys.argv[1])
    else:
        main(sys.argv[1])
