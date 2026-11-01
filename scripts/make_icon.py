"""Draws Lodestar's app icon and saves it as .icns: the app's mark, as the sidebar shows it. A compass star drawn in
the app's ASCII marks, its north ray breaking out through a dotted ring that holds ticks at the other three points,
glowing softly on a deep blue-black tile cut to macOS's rounded-square shape. The star is web/src/art/mark.ts's
compassStar, on the same 15-cell grid the sidebar draws it on, so it reads as one bold mark from the Dock down to the
menu bar.

  uv run --with pillow python scripts/make_icon.py out.icns
  uv run --with pillow python scripts/make_icon.py preview.png 512     # one size, as a png
  uv run --with pillow python scripts/make_icon.py --favicons site/public   # the site's favicons
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


SPAN = 31  # the mark is designed on a 31×31 grid, centre at 15 (web/src/art/mark.ts)


def _ray(along, across, length, w):
    return 0 if along < 0 or along > length else math.exp(-abs(across) / w) * (1 - along / length) ** 0.8


def _star(x, y, up, side, down, diag):
    r = math.hypot(x, y)
    return max(
        1 if r < 1.5 else 0, 0.55 * math.exp(-r / 2.4),
        _ray(-y, x, up, 0.55), _ray(y, x, down, 0.55), _ray(x, y, side, 0.55), _ray(-x, y, side, 0.55),
        *(0.72 * _ray((a * x + b * y) / math.sqrt(2), (a * x - b * y) / math.sqrt(2), diag, 0.45)
          for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))))


def compass_star(x, y):
    """Brightness 0..1 of the mark at (x, y), both in -15..15: the same function as compassStar in mark.ts."""
    r = math.hypot(x, y)
    ring = 0.24 if abs(r - 12.5) < 0.5 and round(math.atan2(y, x) * 12) % 2 == 0 else 0
    ticks = 0.85 if abs(r - 12.5) < 1.2 and (abs(y) < 0.6 or (abs(x) < 0.6 and y > 0)) else 0  # none at north
    return max(_star(x, y, 15.5, 8, 8, 6.5), ring, ticks)


def draw(px, margin=0.1, corner=True):
    """The icon at px square. margin: the empty edge round the tile (Apple's grid is 10%); corner=False gives a square
    tile to the edge, for platforms that cut their own shape (iOS home screens)."""
    ss = 2 if px >= 256 else 4
    s = px * ss
    small = px <= 64
    inset = s * margin
    tile = s - 2 * inset

    # the sky: a vertical fall from blue-black to black, lifted around the star
    sky = Image.new("RGB", (s, s))
    sd = ImageDraw.Draw(sky)
    for y in range(s):
        k = y / s
        sd.line([(0, y), (s, y)], fill=tuple(round(TOP[c] + (BOTTOM[c] - TOP[c]) * k) for c in range(3)))

    cx, cy = s / 2, s / 2
    halo = Image.new("L", (s, s), 0)
    ImageDraw.Draw(halo).ellipse([cx - tile * 0.4, cy - tile * 0.4, cx + tile * 0.4, cy + tile * 0.4], fill=255)
    halo = halo.filter(ImageFilter.GaussianBlur(tile * 0.18))
    sky = Image.composite(Image.new("RGB", (s, s), CORE), sky, halo.point(lambda v: v * 0.55))

    # the mark, in cells: the 31-unit design across most of the tile, each cell the brightest point it covers
    nd = 15  # cells across the design, as the sidebar's mark
    design = tile * 0.8
    cell = design / nd
    n = math.ceil(tile / cell) | 1  # cells across the tile, odd so one sits on the centre
    ox, oy = cx - n * cell / 2, cy - n * cell / 2
    k = SPAN / nd
    grid = [[0.0] * n for _ in range(n)]
    for j in range(n):
        for i in range(n):
            b = 0.0
            for v in range(3):
                for u in range(3):
                    b = max(b, compass_star((i - n / 2 + (u + 0.5) / 3) * k, (j - n / 2 + (v + 0.5) / 3) * k))
            grid[j][i] = b

    if not small:  # a few faint stars in the sky beyond the ring
        for q in range(40):
            i, j = int(hash01(q, 1) * n), int(hash01(q, 2) * n)
            if math.hypot(i - n / 2, j - n / 2) * k > 15.5 and grid[j][i] == 0:
                grid[j][i] = 0.08 + 0.14 * hash01(q, 3)

    art = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    marks(ImageDraw.Draw(art), grid, n, cell, ox, oy)

    # a soft glow behind the centre
    glow = Image.new("RGBA", (s, s), INK + (0,))
    gr = design * 0.14
    ImageDraw.Draw(glow).ellipse([cx - gr, cy - gr, cx + gr, cy + gr], fill=INK + (120,))
    glow = glow.filter(ImageFilter.GaussianBlur(gr * 0.9))

    out = sky.convert("RGBA")
    out.alpha_composite(glow)
    out.alpha_composite(art)

    # a fine highlight along the tile's top edge, as on a lit glass rim
    mask = squircle(s, inset)
    rim = ImageChops.subtract(mask, squircle(s, inset + max(1, s // 400)))
    edge = Image.new("RGBA", (s, s), (255, 255, 255, 0))
    fade = Image.linear_gradient("L").resize((s, s)).point(lambda v: int(70 * (1 - v / 255) ** 2))
    edge.putalpha(ImageChops.multiply(rim, fade))
    out.alpha_composite(edge)

    if corner:
        out.putalpha(mask)
    return out.resize((px, px), Image.Resampling.LANCZOS)


def favicon_svg(n=11):
    """The tab icon as SVG: the mark on a dark rounded tile, each cell one of the app's marks, so browsers draw it sharp
    at any size. The cells are markSVG's (web/src/art/mark.ts), on a coarser grid and without the faintest dots, which
    at 16px would only blur the star."""
    k, out = SPAN / n, []
    for j in range(n):
        for i in range(n):
            b = min(1.0, max(compass_star((i + (u + 0.5) / 3) * k - SPAN / 2, (j + (v + 0.5) / 3) * k - SPAN / 2)
                             for u in range(3) for v in range(3)))
            if b < 0.2:
                continue
            cx, cy, a = i + 0.5, j + 0.5, 0.42
            if b < 0.3:
                h = (0.18 + 0.5 * b) / 2
                out.append(f'<rect x="{cx - h:.3f}" y="{cy - h:.3f}" width="{2 * h:.3f}" height="{2 * h:.3f}" stroke="none"/>')
            elif b < 0.55:
                out.append(f'<path d="M{cx - a:.2f} {cy}h{2 * a}M{cx} {cy - a:.2f}v{2 * a}" stroke-width=".26" opacity=".85"/>')
            elif b < 0.8:
                q = a * 0.75
                out.append(f'<path d="M{cx - q:.3f} {cy - q:.3f}l{2 * q:.3f} {2 * q:.3f}M{cx - q:.3f} {cy + q:.3f}l{2 * q:.3f} {-2 * q:.3f}" stroke-width=".3"/>')
            else:
                out.append(f'<rect x="{cx - 0.42:.2f}" y="{cy - 0.42:.2f}" width=".84" height=".84" stroke="none"/>')
    ink = "#%02x%02x%02x" % INK
    pad = 0.9  # tile edge to the ring's outer cells
    w = n + 2 * pad
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-pad} {-pad} {w} {w}">'
            f'<rect x="{-pad}" y="{-pad}" width="{w}" height="{w}" rx="{w * 0.22:.2f}" fill="#0e1018"/>'
            f'<g fill="{ink}" stroke="{ink}" stroke-linecap="square">{"".join(out)}</g></svg>\n')


def favicons(d):
    """The site's icons: an SVG for browsers that take one, a .ico for those that don't, a full-bleed tile for iOS home
    screens, and two sizes for the web manifest."""
    d = Path(d)
    (d / "favicon.svg").write_text(favicon_svg())
    ico = [draw(px, margin=0.02) for px in (48, 32, 16)]  # each size drawn for itself, not shrunk from the largest
    ico[0].save(d / "favicon.ico", sizes=[im.size for im in ico], append_images=ico[1:])
    draw(180, margin=0, corner=False).convert("RGB").save(d / "apple-touch-icon.png")
    for px in (192, 512):
        draw(px, margin=0, corner=False).convert("RGB").save(d / f"icon-{px}.png")


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
    if sys.argv[1] == "--favicons":
        favicons(sys.argv[2])
    elif len(sys.argv) > 2:
        draw(int(sys.argv[2])).save(sys.argv[1])
    else:
        main(sys.argv[1])
