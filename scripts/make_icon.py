"""Render Lodestar's compass-star ASCII mark as macOS ``.icns``, PNG previews, and site favicons.

Usage::

    uv run --with pillow python scripts/make_icon.py out.icns
    uv run --with pillow python scripts/make_icon.py preview.png 512
    uv run --with pillow python scripts/make_icon.py --favicons site/public
"""

from __future__ import annotations

import math
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter


INK = (195, 207, 245)  # --ascii
TOP, BOTTOM, CORE = (16, 18, 30), (6, 6, 8), (28, 33, 58)
SPAN = 31  # 31x31 design grid matching web/src/art/mark.ts


def hash01(*values: float) -> float:
    """Deterministic pseudo-random float in ``[0, 1)`` from numeric inputs."""
    acc = 0.0
    for val in values:
        acc = math.sin(acc * 1.7 + val * 12.9898 + 78.233) * 43758.5453
    return acc - math.floor(acc)


def squircle(size: int, inset: float) -> Image.Image:
    """Create a macOS rounded-square tile mask inset from the canvas edges."""
    mask = Image.new("L", (size, size), 0)
    side = size - 2 * inset
    ImageDraw.Draw(mask).rounded_rectangle([inset, inset, size - inset, size - inset], radius=side * 0.225, fill=255)
    return mask


def marks(draw_ctx: ImageDraw.ImageDraw, grid: list[list[float]], n: int, cell: float, ox: float, oy: float) -> None:
    """Draw each grid cell as a brightness-mapped Lodestar mark (dot, plus, cross, or block)."""
    for j in range(n):
        for i in range(n):
            brightness = min(1.0, grid[j][i])
            if brightness < 0.06:
                continue

            cx, cy = ox + (i + 0.5) * cell, oy + (j + 0.5) * cell
            arm, width = cell * 0.4, max(1, round(cell * 0.2))

            if brightness < 0.3:
                half = cell * (0.16 + 0.5 * brightness) / 2
                alpha = int(255 * min(1.0, 0.45 + brightness))
                draw_ctx.rectangle([cx - half, cy - half, cx + half, cy + half], fill=(*INK, alpha))
            elif brightness < 0.55:
                color = (*INK, 215)
                draw_ctx.line([cx - arm, cy, cx + arm, cy], fill=color, width=width)
                draw_ctx.line([cx, cy - arm, cx, cy + arm], fill=color, width=width)
            elif brightness < 0.8:
                diag = arm * 0.78
                stroke = round(width * 1.2)
                draw_ctx.line([cx - diag, cy - diag, cx + diag, cy + diag], fill=(*INK, 255), width=stroke)
                draw_ctx.line([cx - diag, cy + diag, cx + diag, cy - diag], fill=(*INK, 255), width=stroke)
            else:
                half = cell * 0.43
                draw_ctx.rectangle([cx - half, cy - half, cx + half, cy + half], fill=(*INK, 255))


def _ray(along: float, across: float, length: float, width: float) -> float:
    if along < 0 or along > length:
        return 0.0
    return math.exp(-abs(across) / width) * (1 - along / length) ** 0.8


def _star(x: float, y: float, up: float, side: float, down: float, diag: float) -> float:
    radius = math.hypot(x, y)
    inv_sqrt2 = 1.0 / math.sqrt(2)
    return max(
        1.0 if radius < 1.5 else 0.0,
        0.55 * math.exp(-radius / 2.4),
        _ray(-y, x, up, 0.55),
        _ray(y, x, down, 0.55),
        _ray(x, y, side, 0.55),
        _ray(-x, y, side, 0.55),
        *(
            0.72 * _ray((a * x + b * y) * inv_sqrt2, (a * x - b * y) * inv_sqrt2, diag, 0.45)
            for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))
        ),
    )


def compass_star(x: float, y: float) -> float:
    """Return brightness in ``[0, 1]`` at coordinates ``(x, y)`` in ``[-15, 15]`` (matches ``mark.ts``)."""
    radius = math.hypot(x, y)
    ring = 0.24 if abs(radius - 12.5) < 0.5 and round(math.atan2(y, x) * 12) % 2 == 0 else 0.0
    ticks = 0.85 if abs(radius - 12.5) < 1.2 and (abs(y) < 0.6 or (abs(x) < 0.6 and y > 0)) else 0.0
    return max(_star(x, y, 15.5, 8, 8, 6.5), ring, ticks)


def draw(px: int, margin: float = 0.1, corner: bool = True) -> Image.Image:
    """Render the icon at ``px x px`` resolution."""
    supersample = 2 if px >= 256 else 4
    size = px * supersample
    small = px <= 64
    inset = size * margin
    tile = size - 2 * inset

    sky = Image.new("RGB", (size, size))
    sky_draw = ImageDraw.Draw(sky)
    for y in range(size):
        ratio = y / size
        color = tuple(round(TOP[c] + (BOTTOM[c] - TOP[c]) * ratio) for c in range(3))
        sky_draw.line([(0, y), (size, y)], fill=color)

    cx = cy = size / 2
    halo = Image.new("L", (size, size), 0)
    ImageDraw.Draw(halo).ellipse([cx - tile * 0.4, cy - tile * 0.4, cx + tile * 0.4, cy + tile * 0.4], fill=255)
    halo = halo.filter(ImageFilter.GaussianBlur(tile * 0.18))
    sky = Image.composite(Image.new("RGB", (size, size), CORE), sky, halo.point(lambda v: v * 0.55))

    nd = 15
    design = tile * 0.8
    cell = design / nd
    n = math.ceil(tile / cell) | 1
    ox, oy = cx - n * cell / 2, cy - n * cell / 2
    scale = SPAN / nd
    grid = [[0.0] * n for _ in range(n)]
    for j in range(n):
        for i in range(n):
            brightness = 0.0
            for v in range(3):
                for u in range(3):
                    brightness = max(
                        brightness,
                        compass_star((i - n / 2 + (u + 0.5) / 3) * scale, (j - n / 2 + (v + 0.5) / 3) * scale),
                    )
            grid[j][i] = brightness

    if not small:
        for q in range(40):
            i, j = int(hash01(q, 1) * n), int(hash01(q, 2) * n)
            if math.hypot(i - n / 2, j - n / 2) * scale > 15.5 and grid[j][i] == 0:
                grid[j][i] = 0.08 + 0.14 * hash01(q, 3)

    art = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    marks(ImageDraw.Draw(art), grid, n, cell, ox, oy)

    glow = Image.new("RGBA", (size, size), (*INK, 0))
    glow_r = design * 0.14
    ImageDraw.Draw(glow).ellipse([cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r], fill=(*INK, 120))
    glow = glow.filter(ImageFilter.GaussianBlur(glow_r * 0.9))

    out = sky.convert("RGBA")
    out.alpha_composite(glow)
    out.alpha_composite(art)

    mask = squircle(size, inset)
    rim = ImageChops.subtract(mask, squircle(size, inset + max(1, size // 400)))
    edge = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    fade = Image.linear_gradient("L").resize((size, size)).point(lambda v: int(70 * (1 - v / 255) ** 2))
    edge.putalpha(ImageChops.multiply(rim, fade))
    out.alpha_composite(edge)

    if corner:
        out.putalpha(mask)
    return out.resize((px, px), Image.Resampling.LANCZOS)


def favicon_svg(n: int = 11) -> str:
    """Render the crisp SVG tab favicon on an ``n x n`` grid."""
    scale = SPAN / n
    out: list[str] = []
    for j in range(n):
        for i in range(n):
            brightness = min(
                1.0,
                max(
                    compass_star((i + (u + 0.5) / 3) * scale - SPAN / 2, (j + (v + 0.5) / 3) * scale - SPAN / 2)
                    for u in range(3)
                    for v in range(3)
                ),
            )
            if brightness < 0.2:
                continue
            cx, cy, arm = i + 0.5, j + 0.5, 0.42
            if brightness < 0.3:
                half = (0.18 + 0.5 * brightness) / 2
                out.append(
                    f'<rect x="{cx - half:.3f}" y="{cy - half:.3f}" width="{2 * half:.3f}" height="{2 * half:.3f}" stroke="none"/>'
                )
            elif brightness < 0.55:
                out.append(f'<path d="M{cx - arm:.2f} {cy}h{2 * arm}M{cx} {cy - arm:.2f}v{2 * arm}" stroke-width=".26" opacity=".85"/>')
            elif brightness < 0.8:
                diag = arm * 0.75
                out.append(
                    f'<path d="M{cx - diag:.3f} {cy - diag:.3f}l{2 * diag:.3f} {2 * diag:.3f}'
                    f'M{cx - diag:.3f} {cy + diag:.3f}l{2 * diag:.3f} {-2 * diag:.3f}" stroke-width=".3"/>'
                )
            else:
                out.append(f'<rect x="{cx - 0.42:.2f}" y="{cy - 0.42:.2f}" width=".84" height=".84" stroke="none"/>')

    ink = "#{:02x}{:02x}{:02x}".format(*INK)
    pad = 0.9
    width = n + 2 * pad
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-pad} {-pad} {width} {width}">'
        f'<rect x="{-pad}" y="{-pad}" width="{width}" height="{width}" rx="{width * 0.22:.2f}" fill="#0e1018"/>'
        f'<g fill="{ink}" stroke="{ink}" stroke-linecap="square">{"".join(out)}</g></svg>\n'
    )


def favicons(target_dir: str | Path) -> None:
    """Write ``favicon.svg``, ``favicon.ico``, ``apple-touch-icon.png``, and manifest PNGs to ``target_dir``."""
    dest = Path(target_dir)
    (dest / "favicon.svg").write_text(favicon_svg())
    ico = [draw(px, margin=0.02) for px in (48, 32, 16)]
    ico[0].save(dest / "favicon.ico", sizes=[im.size for im in ico], append_images=ico[1:])
    draw(180, margin=0, corner=False).convert("RGB").save(dest / "apple-touch-icon.png")
    for px in (192, 512):
        draw(px, margin=0, corner=False).convert("RGB").save(dest / f"icon-{px}.png")


def main(out_path: str | Path) -> None:
    """Build a multi-resolution macOS ``.icns`` icon bundle via ``iconutil``."""
    out = Path(out_path)
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
    if len(sys.argv) < 2:
        sys.exit("Usage: make_icon.py <out.icns | out.png size | --favicons dir>")
    if sys.argv[1] == "--favicons":
        if len(sys.argv) < 3:
            sys.exit("Usage: make_icon.py --favicons <dir>")
        favicons(sys.argv[2])
    elif len(sys.argv) > 2:
        draw(int(sys.argv[2])).save(sys.argv[1])
    else:
        main(sys.argv[1])
