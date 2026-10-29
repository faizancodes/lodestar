"""Builds Interp Matrix, the app's display face: every glyph drawn by hand on a 5×7 dot grid (two more rows below
the baseline for descenders), with round dots, so headings are set in the same dot matrix as the ASCII scenes.
Capitals only; lowercase maps onto them.

  uvx --with fonttools --with brotli python scripts/build_matrix_font.py ui/fonts/InterpMatrix.woff2
"""
import sys

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

GLYPHS = {
    "A": ".###. #...# #...# ##### #...# #...# #...#",
    "B": "####. #...# #...# ####. #...# #...# ####.",
    "C": ".###. #...# #.... #.... #.... #...# .###.",
    "D": "####. #...# #...# #...# #...# #...# ####.",
    "E": "##### #.... #.... ####. #.... #.... #####",
    "F": "##### #.... #.... ####. #.... #.... #....",
    "G": ".###. #...# #.... #.### #...# #...# .####",
    "H": "#...# #...# #...# ##### #...# #...# #...#",
    "I": "### .#. .#. .#. .#. .#. ###",
    "J": "..### ...#. ...#. ...#. ...#. #..#. .##..",
    "K": "#...# #..#. #.#.. ##... #.#.. #..#. #...#",
    "L": "#.... #.... #.... #.... #.... #.... #####",
    "M": "#...# ##.## #.#.# #.#.# #...# #...# #...#",
    "N": "#...# #...# ##..# #.#.# #..## #...# #...#",
    "O": ".###. #...# #...# #...# #...# #...# .###.",
    "P": "####. #...# #...# ####. #.... #.... #....",
    "Q": ".###. #...# #...# #...# #.#.# #..#. .##.#",
    "R": "####. #...# #...# ####. #.#.. #..#. #...#",
    "S": ".#### #.... #.... .###. ....# ....# ####.",
    "T": "##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..",
    "U": "#...# #...# #...# #...# #...# #...# .###.",
    "V": "#...# #...# #...# #...# #...# .#.#. ..#..",
    "W": "#...# #...# #...# #.#.# #.#.# #.#.# .#.#.",
    "X": "#...# #...# .#.#. ..#.. .#.#. #...# #...#",
    "Y": "#...# #...# .#.#. ..#.. ..#.. ..#.. ..#..",
    "Z": "##### ....# ...#. ..#.. .#... #.... #####",
    "0": ".###. #...# #..## #.#.# ##..# #...# .###.",
    "1": ".#. ##. .#. .#. .#. .#. ###",
    "2": ".###. #...# ....# ...#. ..#.. .#... #####",
    "3": "##### ...#. ..#.. ...#. ....# #...# .###.",
    "4": "...#. ..##. .#.#. #..#. ##### ...#. ...#.",
    "5": "##### #.... ####. ....# ....# #...# .###.",
    "6": "..##. .#... #.... ####. #...# #...# .###.",
    "7": "##### ....# ...#. ..#.. .#... .#... .#...",
    "8": ".###. #...# #...# .###. #...# #...# .###.",
    "9": ".###. #...# #...# .#### ....# ...#. .##..",
    ".": ". . . . . . #",
    ",": ".. .. .. .. .. .# .# #.",
    ":": ". . # . . # .",
    ";": ".. .. .# .. .. .# .# #.",
    "!": "# # # # # . #",
    "?": ".###. #...# ....# ...#. ..#.. ..... ..#..",
    "'": "# # . . . . .",
    "\"": "#.# #.# ... ... ... ... ...",
    "-": ".... .... .... #### .... .... ....",
    "–": "..... ..... ..... ##### ..... ..... .....",
    "—": "....... ....... ....... ####### ....... ....... .......",
    "_": "..... ..... ..... ..... ..... ..... ..... #####",
    "(": "..# .#. #.. #.. #.. .#. ..#",
    ")": "#.. .#. ..# ..# ..# .#. #..",
    "[": "### #.. #.. #.. #.. #.. ###",
    "]": "### ..# ..# ..# ..# ..# ###",
    "{": "..# .#. .#. #.. .#. .#. ..#",
    "}": "#.. .#. .#. ..# .#. .#. #..",
    "/": "....# ....# ...#. ..#.. .#... #.... #....",
    "\\": "#.... #.... .#... ..#.. ...#. ....# ....#",
    "|": "# # # # # # # # #",
    "+": "..... ..#.. ..#.. ##### ..#.. ..#.. .....",
    "=": "..... ..... ##### ..... ##### ..... .....",
    "*": "..... ..#.. #.#.# .###. #.#.# ..#.. .....",
    "<": "...# ..#. .#.. #... .#.. ..#. ...#",
    ">": "#... .#.. ..#. ...# ..#. .#.. #...",
    "^": ".#. #.# ... ... ... ... ...",
    "~": "..... ..... .#... #.#.# ...#. ..... .....",
    "`": "#. .# .. .. .. .. ..",
    "%": "##..# ##..# ...#. ..#.. .#... #..## #..##",
    "&": ".##.. #..#. #.#.. .#... #.#.# #..#. .##.#",
    "@": ".###. #...# #.### #.#.# #.### #.... .###.",
    "#": ".#.#. .#.#. ##### .#.#. ##### .#.#. .#.#.",
    "$": "..#.. .#### #.#.. .###. ..#.# ####. ..#..",
    "×": "..... #...# .#.#. ..#.. .#.#. #...# .....",
    "·": ". . . # . . .",
    "•": "... ... ### ### ### ... ...",
    "…": "..... ..... ..... ..... ..... ..... #.#.#",
    "→": "..... ...#. ....# ##### ....# ...#. .....",
}
ALIAS = {"’": "'", "‘": "'", "“": "\"", "”": "\""}
P = 100            # dot pitch, font units: cap height is 7 dots = 700
R = 0.45 * P       # dot radius: nearly touching, so strokes read as lines of light
SPACE = 4          # dots


def dot(pen, cx, cy, r=R):
    # a circle as eight quadratic arcs: on-curve points on the circle, off-curve points where tangents meet
    import math
    k = r / math.cos(math.pi / 8)
    pts = [(cx + r * math.cos(i * math.pi / 4), cy + r * math.sin(i * math.pi / 4)) for i in range(8)]
    offs = [(cx + k * math.cos((i + 0.5) * math.pi / 4), cy + k * math.sin((i + 0.5) * math.pi / 4)) for i in range(8)]
    pen.moveTo(pts[0])
    for i in range(8):
        pen.qCurveTo(offs[i], pts[(i + 1) % 8])
    pen.closePath()


def build(out):
    rows_of = {ch: spec.split() for ch, spec in GLYPHS.items()}
    order = [".notdef", "space"] + [f"g{ord(c):04X}" for c in rows_of]
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(order)
    cmap = {32: "space"}
    for c in rows_of:
        cmap[ord(c)] = f"g{ord(c):04X}"
        if c.isalpha() and c.isupper():
            cmap[ord(c.lower())] = f"g{ord(c):04X}"
    for a, b in ALIAS.items():
        cmap[ord(a)] = f"g{ord(b):04X}"
    fb.setupCharacterMap(cmap)
    glyphs, metrics = {}, {}
    empty = TTGlyphPen(None)
    glyphs[".notdef"] = glyphs["space"] = empty.glyph()
    metrics[".notdef"] = (6 * P, 0)
    metrics["space"] = (SPACE * P, 0)
    for c, rows in rows_of.items():
        pen = TTGlyphPen(None)
        w = len(rows[0])
        for r, line in enumerate(rows):
            for x, v in enumerate(line):
                if v == "#":
                    dot(pen, (x + 0.5) * P + P / 2, (6 - r) * P + P / 2)  # row 6 sits on the baseline
        glyphs[f"g{ord(c):04X}"] = pen.glyph()
        metrics[f"g{ord(c):04X}"] = ((w + 1) * P, P // 2)  # half a dot either side
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=900, descent=-250)
    fb.setupNameTable({"familyName": "Interp Matrix", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=900, sTypoDescender=-250, sTypoLineGap=0, usWinAscent=900, usWinDescent=250,
                sCapHeight=700, sxHeight=700)
    fb.setupPost()
    fb.font.flavor = "woff2"
    fb.save(out)


if __name__ == "__main__":
    build(sys.argv[1])
