#!/usr/bin/env python3
"""Generate the CUDA guide's SVG figures, tuned to the htmler blue theme.

Same house style as the compiler-optimization guide: because the figures are
inlined as static base64 images (no page CSS reaches them), every colour is
chosen to work on BOTH the dark (#0b0d12) and light (#ffffff) themes at once.
A mid-slate around luminance ~0.2 gives roughly 4.3:1 contrast three ways —
white text on the fill, and the same colour as ink on either background.

  * slate blue  #6B7B94  (neutral boxes, connectors, axes, labels)
  * blue        #3E7CC0  (highlighted / "after" boxes)         + dark #2F5F98
  * teal        #1F918C  (positive "result" accent)
  * amber       #D9922B  (warning / spill; dark text on fill)
  * red         #D65A5F  (problem callouts)
  * muted       #9AA0B4  (captions)
  * white       #FFFFFF  (text inside dark fills)
  * 1.5pt wide rules, Aptos / system sans font stack

Run:  python3 scripts/gen_figures.py
Output: figures/*.svg  (referenced from the chapter markdown at the repo root)
"""
import base64
import io
import math
import os

# ── House-style constants (htmler blue theme, dual light/dark legible) ───────
GREY = "#6B7B94"
GREY_D = "#55637A"
BLUE = "#3E7CC0"
BLUE_D = "#2F5F98"
TEAL = "#1F918C"
AMBER = "#D9922B"
RED = "#D65A5F"
WHITE = "#FFFFFF"
LIGHT = "#9AA0B4"
INK_DARK = "#1F2433"  # text on light (amber) fills
# Hand-drawn Excalidraw look: "Virgil" is embedded per-figure as a subsetted
# woff2 data URI (external font URLs are blocked for base64-inlined <img>).
FONT = ("'Virgil','Segoe Print','Bradley Hand','Comic Sans MS',"
        "'Segoe UI',system-ui,-apple-system,sans-serif")
MONO = ("'Virgil','SFMono-Regular',ui-monospace,'JetBrains Mono',Consolas,"
        "monospace")
RULE = 1.5  # pt wide rules

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "Virgil.woff2")

# Populated by esc() as figures are built; used to subset the embedded font.
USED_CHARS = set()
# The <style> block (with the base64 @font-face) injected into every SVG.
FONT_STYLE = ""


# ── Primitive builders ──────────────────────────────────────────────────────
def esc(s):
    USED_CHARS.update(str(s))
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def defs():
    """Arrowhead markers in each ink colour."""
    marks = []
    for name, col in (("g", GREY), ("p", BLUE), ("t", TEAL),
                      ("r", RED), ("a", AMBER), ("l", LIGHT)):
        marks.append(
            f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="8" refY="5" '
            f'markerWidth="4.5" markerHeight="4.5" '
            f'orient="auto-start-reverse">'
            f'<path d="M0 1L9 5L0 9z" fill="{col}"/></marker>')
    return "<defs>" + "".join(marks) + "</defs>"


def rrect(x, y, w, h, fill, rx=9, stroke=None, sw=RULE, dash=None, opacity=None):
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
         f'fill="{fill}"')
    if stroke:
        s += f' stroke="{stroke}" stroke-width="{sw}"'
    if dash:
        s += f' stroke-dasharray="{dash}"'
    if opacity is not None:
        s += f' opacity="{opacity}"'
    return s + "/>"


def tspan_lines(x, cy, lines, fill, size, weight, lh, mono=False):
    """Vertically centred multiline <text>."""
    fam = MONO if mono else FONT
    n = len(lines)
    y0 = cy - (n - 1) * lh / 2.0
    out = [f'<text x="{x}" y="{y0}" fill="{fill}" font-family="{fam}" '
           f'font-size="{size}" font-weight="{weight}" text-anchor="middle" '
           f'dominant-baseline="central">']
    for i, ln in enumerate(lines):
        dy = 0 if i == 0 else lh
        out.append(f'<tspan x="{x}" dy="{dy}">{esc(ln)}</tspan>')
    out.append("</text>")
    return "".join(out)


def box(x, y, w, h, lines, fill=GREY, tcol=WHITE, size=13, weight=600,
        rx=9, lh=16, stroke=None, sw=RULE, dash=None, mono=False):
    if isinstance(lines, str):
        lines = lines.split("\n")
    r = rrect(x, y, w, h, fill, rx=rx, stroke=stroke, sw=sw, dash=dash)
    t = tspan_lines(x + w / 2.0, y + h / 2.0, lines, tcol, size, weight, lh, mono)
    return r + t


def obox(x, y, w, h, lines, stroke=GREY, tcol=GREY, size=13, weight=600,
         rx=9, lh=16, sw=RULE, dash=None, fill="none", mono=False):
    """Outlined box (transparent fill) with coloured text."""
    r = rrect(x, y, w, h, fill, rx=rx, stroke=stroke, sw=sw, dash=dash)
    t = tspan_lines(x + w / 2.0, y + h / 2.0, lines if isinstance(lines, list)
                    else [lines], tcol, size, weight, lh, mono)
    return r + t


def text(x, y, s, fill=GREY, size=13, weight=600, anchor="middle",
         italic=False, mono=False):
    fam = MONO if mono else FONT
    st = ""  # italics disabled: the hand-drawn font is hard to read slanted
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-family="{fam}" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"'
            f'{st} dominant-baseline="central">{esc(s)}</text>')


def line(x1, y1, x2, y2, col=GREY, sw=RULE, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
            f'stroke-width="{sw}"{d}/>')


def _mk(col):
    return {GREY: "g", BLUE: "p", TEAL: "t", RED: "r", AMBER: "a",
            LIGHT: "l"}.get(col, "g")


def arrow(x1, y1, x2, y2, col=GREY, sw=RULE, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
            f'stroke-width="{sw}" marker-end="url(#ah-{_mk(col)})"{d}/>')


def path(d, col=GREY, sw=RULE, dash=None, arrow_end=False, fill="none"):
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#ah-{_mk(col)})"' if arrow_end else ""
    return (f'<path d="{d}" fill="{fill}" stroke="{col}" stroke-width="{sw}"'
            f'{dd}{m}/>')


def circle(cx, cy, r, fill, stroke=None, sw=RULE):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"{st}/>'


def cylinder(x, y, w, h, fill=GREY, tcol=WHITE, lines=None, size=12,
             stroke=None, sw=RULE):
    """Database / memory cylinder."""
    ry = min(h * 0.16, 14)
    st = (f' stroke="{stroke}" stroke-width="{sw}"') if stroke else ""
    body = (f'<path d="M{x} {y+ry} A{w/2} {ry} 0 0 0 {x+w} {y+ry} '
            f'L{x+w} {y+h-ry} A{w/2} {ry} 0 0 1 {x} {y+h-ry} Z" '
            f'fill="{fill}"{st}/>')
    top = (f'<ellipse cx="{x+w/2}" cy="{y+ry}" rx="{w/2}" ry="{ry}" '
           f'fill="{fill}"{st}/>')
    lip = (f'<path d="M{x} {y+ry} A{w/2} {ry} 0 0 0 {x+w} {y+ry}" '
           f'fill="none" stroke="{WHITE}" stroke-width="1" opacity="0.35"/>')
    t = ""
    if lines:
        t = tspan_lines(x + w / 2.0, y + h / 2.0 + ry / 2, lines, tcol, size,
                        600, 15)
    return body + top + lip + t


def svg(w, h, body, title=""):
    t = f"<title>{esc(title)}</title>" if title else ""
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" font-family="{FONT}">{t}{FONT_STYLE}'
            f'{defs()}{body}</svg>\n')


def write(rel_path, content):
    full = os.path.join(REPO_ROOT, rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    print("wrote", rel_path, f"({len(content)} bytes)")


def core_grid(x, y, cols, rows, cell=13, gap=4, col=BLUE):
    """A small grid of tiny squares (visual shorthand for 'many cores')."""
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(rrect(x + c * (cell + gap), y + r * (cell + gap),
                             cell, cell, col, rx=2))
    return "".join(out)


# ── 00 introduction ─────────────────────────────────────────────────────────
def fig_cpu_vs_gpu():
    W, H = 940, 400
    b = [text(W / 2, 26, "CPU vs GPU: latency cores vs throughput cores",
              GREY, 17, 700)]
    # left CPU panel
    b.append(obox(30, 50, 420, 320, "", GREY, GREY, rx=14))
    b.append(text(240, 74, "CPU  \u2014  a few complex cores", GREY, 14, 700))
    cores = [(60, 100), (245, 100), (60, 195), (245, 195)]
    for (cx, cy) in cores:
        b.append(box(cx, cy, 165, 78, "", BLUE, rx=10))
        b.append(text(cx + 82, cy + 20, "Core", WHITE, 13, 700))
        b.append(box(cx + 20, cy + 36, 125, 28, "control + L1/L2", BLUE_D,
                     size=10, rx=6))
    b.append(box(60, 292, 350, 34, "large shared L3 cache", GREY_D, size=12,
                 rx=8))
    b.append(text(240, 350, "deep caches \u00b7 branch prediction \u00b7 "
                  "fast single thread", LIGHT, 11, 500, italic=True))
    # right GPU panel
    b.append(obox(490, 50, 420, 320, "", GREY, GREY, rx=14))
    b.append(text(700, 74, "GPU  \u2014  thousands of simple cores", GREY, 14,
                  700))
    sm_pos = [(510, 96), (700, 96), (510, 205), (700, 205)]
    for (sx, sy) in sm_pos:
        b.append(box(sx, sy, 190, 96, "", BLUE, rx=10))
        b.append(text(sx + 95, sy + 15, "SM", WHITE, 11, 700))
        b.append(core_grid(sx + 16, sy + 28, 8, 3, cell=13, gap=4,
                           col="#BCD3EE"))
        b.append(box(sx + 16, sy + 75, 158, 14, "shared mem", BLUE_D, size=8,
                     rx=4, lh=10))
    b.append(text(700, 350, "many SMs \u00d7 many cores \u00b7 latency hidden "
                  "by parallelism", LIGHT, 11, 500, italic=True))
    write("figures/cpu-vs-gpu.svg", svg(W, H, "".join(b), "CPU vs GPU"))


def fig_thread_hierarchy():
    W, H = 840, 430
    b = [text(W / 2, 26, "Thread hierarchy: grid \u2283 block \u2283 thread",
              GREY, 17, 700)]
    b.append(obox(30, 48, 780, 300, "", GREY, GREY, rx=14))
    b.append(text(120, 68, "Grid (one kernel launch)", GREY, 12, 700,
                  anchor="start"))
    bw, bh = 230, 118
    for j in range(2):
        for i in range(3):
            bx = 55 + i * 250
            by = 82 + j * 132
            b.append(box(bx, by, bw, bh, "", BLUE, rx=10))
            b.append(text(bx + bw / 2, by + 16, f"Block ({i},{j})", WHITE, 11,
                          700))
            # threads
            for r in range(3):
                for c in range(6):
                    cx = bx + 34 + c * 28
                    cy = by + 42 + r * 24
                    b.append(circle(cx, cy, 7, "#BCD3EE"))
    b.append(text(W / 2, 372, "threads in a block share memory and can "
                  "sync; blocks run independently, in any order",
                  LIGHT, 12, 500, italic=True))
    b.append(text(W / 2, 398, "T = thread \u00b7 threadIdx within block \u00b7 "
                  "blockIdx within grid", LIGHT, 11, 500))
    write("figures/thread-hierarchy.svg",
          svg(W, H, "".join(b), "CUDA thread hierarchy"))


def fig_memory_hierarchy():
    W, H = 760, 440
    b = [text(W / 2, 26, "GPU memory hierarchy: speed \u2194 size", GREY, 16,
              700)]
    rows = [
        ("Registers", "per thread", "~1 cycle", TEAL, 220),
        ("Shared mem / L1", "per block", "~5\u201330 cyc", BLUE, 300),
        ("L2 cache", "device-wide", "~200 cyc", BLUE_D, 420),
        ("Global memory (HBM)", "all threads", "~400 cyc", GREY, 540),
        ("Host RAM (via PCIe)", "CPU side", "\u226b slow", RED, 620),
    ]
    y0, rh = 66, 62
    cx = W / 2
    for i, (name, scope, lat, col, w) in enumerate(rows):
        y = y0 + i * rh
        x = cx - w / 2
        b.append(box(x, y, w, 44, "", col, rx=9))
        b.append(text(cx, y + 22, name, WHITE, 13, 700))
        b.append(text(x - 12, y + 22, scope, LIGHT, 11, 500, anchor="end"))
        b.append(text(x + w + 12, y + 22, lat, LIGHT, 11, 600, anchor="start"))
    # side arrows
    b.append(arrow(40, y0 + 4 * rh + 22, 40, y0 + 22, TEAL, 2))
    b.append(text(28, (y0 + 22 + y0 + 4 * rh + 22) / 2, "faster", TEAL, 11,
                  700, anchor="middle"))
    b.append(arrow(W - 40, y0 + 22, W - 40, y0 + 4 * rh + 22, GREY, 2))
    b.append(text(W - 26, (y0 + 22 + y0 + 4 * rh + 22) / 2, "bigger", GREY, 11,
                  700, anchor="middle"))
    write("figures/memory-hierarchy.svg",
          svg(W, H, "".join(b), "Memory hierarchy"))


def fig_roofline():
    W, H = 760, 400
    b = [text(W / 2, 26, "The roofline model", GREY, 16, 700)]
    ox, oy = 90, 320          # origin
    xr, yt = 690, 70          # axis extents
    b.append(arrow(ox, oy, xr, oy, GREY, RULE))   # x axis
    b.append(arrow(ox, oy, ox, yt, GREY, RULE))   # y axis
    b.append(text(W / 2, 360, "arithmetic intensity  (FLOP / byte)  \u2192",
                  GREY, 12, 600))
    ylab_y = (oy + yt) / 2
    b.append(f'<g transform="rotate(-90 40 {ylab_y})">'
             + text(40, ylab_y, "attainable GFLOP/s", GREY, 12, 600)
             + '</g>')
    # roofline: rising slope to ridge, then flat ceiling
    ridge_x, ceil_y = 360, 110
    b.append(path(f"M{ox} {oy} L{ridge_x} {ceil_y}", BLUE, 3))
    b.append(path(f"M{ridge_x} {ceil_y} L{xr - 10} {ceil_y}", TEAL, 3))
    b.append(line(ridge_x, ceil_y, ridge_x, oy, LIGHT, 1, dash="4 4"))
    b.append(text(ridge_x, oy + 18, "ridge point", LIGHT, 10, 600))
    b.append(text((ox + ridge_x) / 2 - 40, (oy + ceil_y) / 2 - 8,
                  "memory-bound", BLUE, 12, 700, anchor="middle"))
    b.append(text((ox + ridge_x) / 2 - 40, (oy + ceil_y) / 2 + 8,
                  "(bandwidth \u00d7 AI)", BLUE, 10, 500, anchor="middle"))
    b.append(text((ridge_x + xr) / 2, ceil_y - 14, "compute-bound "
                  "(peak FLOP/s)", TEAL, 12, 700))
    # example kernels
    b.append(circle(300, 250, 6, RED))
    b.append(text(300, 268, "naive", RED, 10, 600))
    b.append(circle(520, ceil_y + 24, 6, TEAL))
    b.append(text(520, ceil_y + 44, "tiled / fused", TEAL, 10, 600))
    write("figures/roofline.svg", svg(W, H, "".join(b), "Roofline model"))


# ── 04 thread indexing patterns ─────────────────────────────────────────────
def fig_index_1d():
    W, H = 820, 260
    b = [text(W / 2, 26, "1D global index = blockIdx.x \u00b7 blockDim.x + "
              "threadIdx.x", GREY, 15, 700)]
    bw = 240
    blocks = 3
    perblk = 4
    x0 = 30
    for blk in range(blocks):
        bx = x0 + blk * (bw + 10)
        b.append(box(bx, 70, bw, 70, "", BLUE if blk == 1 else GREY, rx=10))
        b.append(text(bx + bw / 2, 58, f"blockIdx.x = {blk}", GREY, 11, 600))
        for t in range(perblk):
            cw = 52
            cx = bx + 12 + t * (cw + 3)
            gidx = blk * perblk + t
            b.append(box(cx, 88, cw, 36, str(t), BLUE_D if blk == 1 else GREY_D,
                         size=12, rx=6))
            b.append(text(cx + cw / 2, 158, str(gidx), TEAL, 12, 700))
    b.append(text(30, 158, "global idx:", LIGHT, 11, 600, anchor="start"))
    b.append(text(W / 2, 200, "blockDim.x = 4 threads per block", LIGHT, 11,
                  500, italic=True))
    b.append(text(W / 2, 224, "one contiguous, monotonically increasing index "
                  "across the whole grid", LIGHT, 11, 500, italic=True))
    write("figures/index-1d.svg", svg(W, H, "".join(b), "1D indexing"))


def fig_grid_2d():
    W, H = 640, 460
    b = [text(W / 2, 26, "2D indexing maps threads to a matrix", GREY, 16,
              700)]
    b.append(text(W / 2, 50, "col = blockIdx.x\u00b7blockDim.x + threadIdx.x   "
                  "\u00b7   row = blockIdx.y\u00b7blockDim.y + threadIdx.y",
                  LIGHT, 11, 500))
    gx, gy = 120, 80
    cell = 46
    n = 8
    tile = 4
    for r in range(n):
        for c in range(n):
            x = gx + c * cell
            y = gy + r * cell
            hot = (r // tile == 1 and c // tile == 1)
            b.append(rrect(x, y, cell - 3, cell - 3,
                          BLUE if (r == 5 and c == 5) else
                          ("#DCE7F5" if hot else "none"),
                          rx=4, stroke=GREY, sw=1))
    # block boundaries
    for k in range(0, n + 1, tile):
        b.append(line(gx, gy + k * cell, gx + n * cell, gy + k * cell, BLUE_D,
                      2))
        b.append(line(gx + k * cell, gy, gx + k * cell, gy + n * cell, BLUE_D,
                      2))
    b.append(text(gx + 5.5 * cell, gy + 5.5 * cell - 1, "", WHITE, 10, 700))
    b.append(text(gx - 14, gy + n * cell / 2, "row (y)", GREY, 12, 700,
                  anchor="middle"))
    b.append(text(gx + n * cell / 2, gy - 14, "col (x)", GREY, 12, 700))
    b.append(text(W / 2, gy + n * cell + 30, "thick lines = block edges; one "
                  "thread \u2192 one element (row, col)", LIGHT, 11, 500,
                  italic=True))
    write("figures/grid-2d.svg", svg(W, H, "".join(b), "2D grid indexing"))


def fig_coalesced():
    W, H = 820, 340
    b = [text(W / 2, 26, "Coalesced vs strided global memory access", GREY, 15,
              700)]

    def lanes(y, label):
        b.append(text(30, y + 18, label, GREY, 12, 700, anchor="start"))
        xs = []
        for t in range(8):
            x = 150 + t * 40
            b.append(box(x, y, 34, 34, f"t{t}", GREY, size=11, rx=6))
            xs.append(x + 17)
        return xs

    def mem(y):
        xs = []
        for i in range(8):
            x = 150 + i * 40
            b.append(rrect(x, y, 34, 34, "#DCE7F5", rx=6, stroke=GREY, sw=1))
            b.append(text(x + 17, y + 17, str(i), GREY, 10, 600))
            xs.append(x + 17)
        return xs

    # coalesced
    b.append(text(410, 66, "coalesced \u2014 thread t reads address t", TEAL,
                  12, 700))
    tl = lanes(78, "warp")
    ml = mem(150)
    for a, c in zip(tl, ml):
        b.append(arrow(a, 112, c, 150, TEAL, RULE))
    b.append(box(490, 148, 210, 34, "1 memory transaction", TEAL, size=11,
                 rx=8))
    # strided
    b.append(text(410, 216, "strided \u2014 thread t reads address 2t "
                  "(gaps)", RED, 12, 700))
    tl2 = lanes(228, "warp")
    ml2 = mem(298)
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    for i, a in enumerate(tl2):
        tgt = ml2[order[i]] if i < 4 else a
        if i < 4:
            b.append(arrow(a, 262, ml2[i * 2] if i * 2 < 8 else a, 298, RED,
                           RULE))
    b.append(box(490, 296, 210, 34, "many transactions \u2192 wasted BW", RED,
                 size=11, rx=8))
    write("figures/coalesced.svg", svg(W, H, "".join(b), "Coalesced access"))


# ── 05 memory model ─────────────────────────────────────────────────────────
def fig_memory_spaces():
    W, H = 720, 400
    b = [text(W / 2, 26, "Memory scopes follow the thread hierarchy", GREY, 16,
              700)]
    # grid scope (outer)
    b.append(obox(40, 48, 640, 320, "", GREY, GREY, rx=14))
    b.append(text(60, 68, "Grid scope", GREY, 12, 700, anchor="start"))
    b.append(box(440, 92, 210, 40, "global memory", GREY, size=12, rx=8))
    b.append(box(440, 146, 210, 40, "constant / texture", GREY_D, size=12,
                 rx=8))
    b.append(text(545, 214, "visible to every thread,", LIGHT, 10, 500))
    b.append(text(545, 230, "persists across kernels", LIGHT, 10, 500))
    # block scope
    b.append(obox(60, 88, 340, 262, "", BLUE, BLUE, rx=12))
    b.append(text(78, 106, "Block scope", BLUE, 12, 700, anchor="start"))
    # thread scope (two threads) sit above the shared-memory bar
    for i, tx in enumerate((76, 236)):
        b.append(obox(tx, 120, 148, 150, "", TEAL, TEAL, rx=10))
        b.append(text(tx + 74, 138, f"Thread {i}", TEAL, 11, 700))
        b.append(box(tx + 16, 152, 116, 34, "registers", TEAL, size=10, rx=6))
        b.append(box(tx + 16, 192, 116, 34, "local mem", GREY_D, size=10,
                     rx=6))
    # shared memory is per-block: one bar spanning both threads
    b.append(box(76, 286, 308, 40, "shared memory  (shared by the block)",
                 BLUE, size=11, rx=8))
    b.append(text(W / 2, 386, "registers/local = private \u00b7 shared = "
                  "per-block \u00b7 global/constant = whole grid", LIGHT, 11,
                  500, italic=True))
    write("figures/memory-spaces.svg",
          svg(W, H, "".join(b), "CUDA memory spaces"))


# ── 07 shared memory ────────────────────────────────────────────────────────
def fig_reduction():
    W, H = 780, 400
    b = [text(W / 2, 26, "Parallel tree reduction: log\u2082(n) steps",
              GREY, 15, 700)]
    # Left-packed reduction: at each step only lanes t < stride stay active,
    # and each shows the running partial sum. Boxes stay in their column so the
    # "active lanes packed to the left" idea is visible.
    strides = [4, 2, 1]
    cur = [3, 1, 7, 0, 4, 1, 6, 3]
    levels = [list(cur)]          # (values, active_count) via len tracking
    counts = [8]
    active = 8
    for st in strides:
        nxt = list(cur)
        for t in range(st):
            nxt[t] = cur[t] + cur[t + st]
        active = st
        cur = nxt
        levels.append(list(nxt))
        counts.append(active)
    cw, gap = 56, 12
    y0, rh = 66, 84
    n0 = 8
    total_w = n0 * cw + (n0 - 1) * gap
    x0 = (W - total_w) / 2

    def colx(i):
        return x0 + i * (cw + gap)

    for lvl, arr in enumerate(levels):
        y = y0 + lvl * rh
        act = counts[lvl]
        # arrows into this level's active lanes
        if lvl > 0:
            st = strides[lvl - 1]
            for i in range(act):
                py = y0 + (lvl - 1) * rh + 40
                b.append(arrow(colx(i) + cw / 2, py, colx(i) + cw / 2, y,
                               GREY, RULE))
                b.append(arrow(colx(i + st) + cw / 2, py,
                               colx(i) + cw / 2 + 6, y, TEAL, RULE))
        for i in range(n0):
            act_here = i < act
            col = (BLUE if lvl > 0 else GREY) if act_here else "#E4E7EF"
            tc = WHITE if act_here else LIGHT
            b.append(box(colx(i), y, cw, 40, str(levels[lvl][i]), col,
                         tcol=tc, size=14, rx=8))
        lbl = "input" if lvl == 0 else f"stride {strides[lvl-1]}"
        b.append(text(x0 - 16, y + 20, lbl, LIGHT, 10, 600, anchor="end"))
    b.append(text(W / 2, H - 22, "if (t < stride) s[t] += s[t+stride]  "
                  "\u2014  active lanes stay packed \u2192 no warp divergence",
                  LIGHT, 11, 500, italic=True))
    write("figures/reduction.svg", svg(W, H, "".join(b), "Tree reduction"))


# ── 08 execution model & occupancy ──────────────────────────────────────────
def fig_blocks_to_sm():
    W, H = 820, 340
    b = [text(W / 2, 26, "Blocks are distributed to SMs in waves", GREY, 16,
              700)]
    b.append(text(W / 2, 50, "kernel<<<12 blocks, 256 threads>>> on a 4-SM "
                  "GPU", LIGHT, 12, 500, italic=True))
    waves = [["B0", "B4", "B8"], ["B1", "B5", "B9"], ["B2", "B6", "B10"],
             ["B3", "B7", "B11"]]
    sx0 = 60
    smw = 170
    for s in range(4):
        sx = sx0 + s * (smw + 15)
        b.append(obox(sx, 78, smw, 180, "", GREY, GREY, rx=12))
        b.append(text(sx + smw / 2, 96, f"SM {s}", GREY, 13, 700))
        for k, blk in enumerate(waves[s]):
            col = BLUE if k == 0 else (BLUE_D if k == 1 else GREY_D)
            b.append(box(sx + 25, 112 + k * 44, smw - 50, 36, blk, col,
                         size=13, rx=8))
        b.append(text(sx + smw / 2, 246, "8 warps each", LIGHT, 9, 500))
    b.append(text(60, 292, "wave 1 (resident):", BLUE, 11, 700, anchor="start"))
    b.append(box(200, 278, 24, 24, "", BLUE, rx=5))
    b.append(text(360, 292, "later waves wait for regs/smem/warp slots to free",
                  LIGHT, 11, 500, italic=True, anchor="start"))
    b.append(box(330, 278, 24, 24, "", GREY_D, rx=5))
    write("figures/blocks-to-sm.svg",
          svg(W, H, "".join(b), "Blocks to SMs"))


def fig_latency_hiding():
    W, H = 820, 320
    b = [text(W / 2, 26, "Latency hiding: the scheduler swaps stalled warps",
              GREY, 15, 700)]
    y0 = 70
    rh = 46
    x0 = 120
    unit = 46
    # each warp: compute (blue) then stall (light) then compute...
    plan = [
        [("c", 1), ("s", 3), ("c", 1), ("s", 3)],
        [("s", 1), ("c", 1), ("s", 3), ("c", 1), ("s", 2)],
        [("s", 2), ("c", 1), ("s", 3), ("c", 1)],
        [("s", 3), ("c", 1), ("s", 3), ("c", 1)],
    ]
    for w, segs in enumerate(plan):
        y = y0 + w * rh
        b.append(text(x0 - 14, y + 16, f"W{w}", GREY, 12, 700, anchor="end"))
        x = x0
        for kind, dur in segs:
            ww = dur * unit
            if kind == "c":
                b.append(box(x, y, ww, 32, "compute", BLUE, size=10, rx=6))
            else:
                b.append(rrect(x, y, ww, 32, "#E4E7EF", rx=6, stroke=LIGHT,
                              sw=1))
                b.append(text(x + ww / 2, y + 16, "stall", LIGHT, 9, 500))
            x += ww
    # highlight that some warp is always computing
    b.append(text(x0, y0 + 4 * rh + 6, "time \u2192", GREY, 11, 600,
                  anchor="start"))
    b.append(text(W / 2, H - 24, "while one warp waits on memory (~400 cyc), "
                  "ready warps keep the cores busy \u2014 throughput, not "
                  "latency", LIGHT, 11, 500, italic=True))
    write("figures/latency-hiding.svg",
          svg(W, H, "".join(b), "Latency hiding"))


# ── 09 work allocation ──────────────────────────────────────────────────────
def fig_occupancy():
    W, H = 760, 380
    b = [text(W / 2, 26, "Occupancy = the most constraining resource wins",
              GREY, 15, 700)]
    # three resource columns; each caps how many warps can be resident
    cols = [
        ("Warp/block\nslots", 0.95, BLUE, "hardware limit"),
        ("Registers\nper thread", 0.55, AMBER, "high reg use \u2192 fewer warps"),
        ("Shared mem\nper block", 0.75, TEAL, "big smem \u2192 fewer blocks"),
    ]
    bx0, bw, gap = 110, 130, 90
    base = 300
    barmax = 210
    minfrac = min(c[1] for c in cols)
    for i, (name, frac, col, note) in enumerate(cols):
        x = bx0 + i * (bw + gap)
        b.append(rrect(x, base - barmax, bw, barmax, "#E4E7EF", rx=10))
        h = barmax * frac
        b.append(rrect(x, base - h, bw, h, col, rx=10))
        b.append(text(x + bw / 2, base + 18, name.split("\n")[0], GREY, 11,
                      700))
        b.append(text(x + bw / 2, base + 33, name.split("\n")[1], GREY, 11,
                      700))
        b.append(text(x + bw / 2, base + 52, note, LIGHT, 9, 500))
    # min line
    ymin = base - barmax * minfrac
    b.append(line(90, ymin, W - 40, ymin, RED, 2, dash="6 4"))
    b.append(text(W - 44, ymin - 12, "achieved occupancy", RED, 11, 700,
                  anchor="end"))
    write("figures/occupancy.svg", svg(W, H, "".join(b), "Occupancy limiters"))


# ── 10 gpu architecture ─────────────────────────────────────────────────────
def fig_gpu_chip():
    W, H = 820, 460
    b = [text(W / 2, 26, "GPU chip layout: SMs around a shared L2 + HBM",
              GREY, 16, 700)]
    # HBM stacks left/right
    for side, sx in (("HBM", 30), ("HBM", W - 90)):
        for k in range(3):
            b.append(box(sx, 90 + k * 90, 60, 74, "HBM", AMBER, tcol=INK_DARK,
                         size=11, rx=8))
    # memory controllers
    b.append(text(W / 2, 52, "GPCs (SM clusters)", LIGHT, 11, 500))
    # SM grid
    gx, gy = 130, 74
    smw, smh = 74, 44
    for r in range(6):
        for c in range(6):
            x = gx + c * (smw + 8)
            y = gy + r * (smh + 7)
            if r in (2, 3):   # middle band is L2
                continue
            b.append(box(x, y, smw, smh, "SM", BLUE, size=11, rx=7))
    # L2 band in middle
    b.append(box(gx, gy + 2 * (smh + 7), 6 * smw + 5 * 8, 2 * smh + 7,
                 "L2 cache (shared by all SMs)", GREY_D, size=13, rx=10))
    # bottom I/O
    b.append(box(W / 2 - 220, H - 60, 200, 38, "PCIe / host", GREY, size=12,
                 rx=8))
    b.append(box(W / 2 + 20, H - 60, 200, 38, "NVLink / other GPUs", GREY_D,
                 size=12, rx=8))
    b.append(text(W / 2, H - 78, "off-chip HBM feeds L2 \u2192 SMs; SMs never "
                  "touch DRAM directly", LIGHT, 11, 500, italic=True))
    write("figures/gpu-chip.svg", svg(W, H, "".join(b), "GPU chip layout"))


def fig_sm_block():
    W, H = 860, 440
    b = [text(W / 2, 26, "Inside a Streaming Multiprocessor (SM)", GREY, 16,
              700)]
    b.append(obox(30, 46, 800, 360, "", GREY, GREY, rx=14))
    # L1/shared at bottom, register-file top; 4 processing blocks
    b.append(box(50, 64, 760, 36, "L1 instruction cache \u00b7 4 warp "
                 "schedulers + dispatch", GREY_D, size=12, rx=8))
    pbw = 182
    for i in range(4):
        x = 50 + i * (pbw + 8)
        b.append(obox(x, 112, pbw, 210, "", BLUE, BLUE, rx=10))
        b.append(text(x + pbw / 2, 128, f"Processing block {i}", BLUE, 11,
                      700))
        b.append(box(x + 16, 142, pbw - 32, 30, "warp scheduler", BLUE, size=10,
                     rx=6))
        b.append(box(x + 16, 178, pbw - 32, 30, "32 FP32 cores", BLUE_D,
                     size=10, rx=6))
        b.append(box(x + 16, 214, pbw - 32, 30, "16 INT32 \u00b7 SFU", BLUE_D,
                     size=10, rx=6))
        b.append(box(x + 16, 250, pbw - 32, 30, "Tensor Core", TEAL, size=10,
                     rx=6))
        b.append(box(x + 16, 286, pbw - 32, 28, "register file", GREY_D,
                     size=10, rx=6))
    b.append(box(50, 334, 760, 34, "shared memory / L1 data cache "
                 "(configurable split, up to 228 KB)", GREY, size=12, rx=8))
    b.append(text(W / 2, 392, "warps from resident blocks are issued by the "
                  "schedulers every cycle", LIGHT, 11, 500, italic=True))
    write("figures/sm-block.svg", svg(W, H, "".join(b), "SM block diagram"))


def fig_warp_divergence():
    W, H = 780, 360
    b = [text(W / 2, 26, "Warp divergence: branches serialize within a warp",
              GREY, 15, 700)]
    # 8 lanes hit if/else
    lane_x = [90 + i * 76 for i in range(8)]
    b.append(text(W / 2, 58, "if (threadIdx & 1) A();  else B();", GREY, 12,
                  700, mono=True))
    for i, x in enumerate(lane_x):
        b.append(box(x, 74, 60, 30, f"t{i}", GREY, size=11, rx=6))
    yA, yB = 150, 240
    b.append(text(30, yA + 17, "pass 1", LIGHT, 10, 600, anchor="start"))
    b.append(text(30, yB + 17, "pass 2", LIGHT, 10, 600, anchor="start"))
    for i, x in enumerate(lane_x):
        odd = i % 2 == 1
        # pass 1 executes A() on odd lanes, B lanes idle
        b.append(box(x, yA, 60, 34, "A" if odd else "\u00b7",
                     BLUE if odd else "#E4E7EF",
                     tcol=WHITE if odd else LIGHT, size=12, rx=6))
        b.append(box(x, yB, 60, 34, "B" if not odd else "\u00b7",
                     TEAL if not odd else "#E4E7EF",
                     tcol=WHITE if not odd else LIGHT, size=12, rx=6))
        b.append(line(x + 30, 104, x + 30, yA, LIGHT, 1))
    b.append(text(W - 40, yA + 17, "half idle", RED, 10, 700, anchor="end"))
    b.append(text(W - 40, yB + 17, "half idle", RED, 10, 700, anchor="end"))
    b.append(text(W / 2, 320, "both sides run one after another; masked lanes "
                  "waste cycles. Reconverge after.", LIGHT, 11, 500,
                  italic=True))
    write("figures/warp-divergence.svg",
          svg(W, H, "".join(b), "Warp divergence"))


# ── 11 matrix multiplication ────────────────────────────────────────────────
def fig_tiling():
    W, H = 820, 380
    b = [text(W / 2, 26, "Shared-memory tiling reuses each load TILE times",
              GREY, 15, 700)]

    def matrix(ox, oy, label, hot_col=None, hot_row=None, out_rc=None,
               accent=BLUE):
        cell = 40
        n = 3
        b.append(text(ox + n * cell / 2, oy - 12, label, GREY, 13, 700))
        for r in range(n):
            for c in range(n):
                x = ox + c * cell
                y = oy + r * cell
                fill = "none"
                if hot_col is not None and c == hot_col:
                    fill = "#DCE7F5"
                if hot_row is not None and r == hot_row:
                    fill = "#DCE7F5"
                if out_rc is not None and (r, c) == out_rc:
                    fill = accent
                b.append(rrect(x, y, cell - 3, cell - 3, fill, rx=5,
                              stroke=GREY, sw=1))

    matrix(70, 110, "A  (tiles of a row)", hot_row=1)
    b.append(text(230, 165, "\u00d7", GREY, 22, 800))
    matrix(270, 110, "B  (tiles of a col)", hot_col=1)
    b.append(text(430, 165, "=", GREY, 22, 800))
    matrix(470, 110, "C  (this block's tile)", out_rc=(1, 1), accent=TEAL)
    b.append(box(620, 120, 170, 100,
                 ["for each k-tile:", "load A-tile & B-tile", "\u2192 shared "
                  "mem, then", "accumulate into C"], BLUE_D, size=11, lh=18,
                 rx=10))
    b.append(text(W / 2, 300, "each element loaded once per tile and reused "
                  "TILE times \u2192 global traffic drops ~TILE\u00d7", TEAL,
                  12, 700))
    b.append(text(W / 2, 326, "arithmetic intensity rises \u2192 kernel "
                  "becomes compute-bound (3\u201310\u00d7 faster than naive)",
                  LIGHT, 11, 500, italic=True))
    write("figures/tiling.svg", svg(W, H, "".join(b), "Tiled matmul"))


# ── 12 atomics & synchronization ────────────────────────────────────────────
def fig_race_condition():
    W, H = 780, 360
    b = [text(W / 2, 26, "Race condition: a lost update without atomics",
              GREY, 15, 700)]
    # two thread lanes + counter column
    t0x, t1x, cx = 150, 400, 660
    b.append(text(t0x, 60, "Thread 0", BLUE, 13, 700))
    b.append(text(t1x, 60, "Thread 1", BLUE_D, 13, 700))
    b.append(text(cx, 60, "counter", GREY, 13, 700))
    steps = [
        (0, "read \u2192 5", None, "5"),
        (1, None, "read \u2192 5", "5"),
        (2, "5 + 1 = 6", None, "5"),
        (3, None, "5 + 1 = 6", "5"),
        (4, "write 6", None, "6"),
        (5, None, "write 6", "6"),
    ]
    y0 = 84
    rh = 40
    for (t, a, c, cv) in steps:
        y = y0 + t * rh
        b.append(text(40, y + 15, f"t={t}", LIGHT, 11, 600, anchor="start"))
        if a:
            b.append(box(t0x - 80, y, 160, 30, a, BLUE, size=11, rx=6))
        if c:
            b.append(box(t1x - 80, y, 160, 30, c, BLUE_D, size=11, rx=6))
        b.append(box(cx - 30, y, 60, 30, cv, GREY, size=12, rx=6))
    b.append(line(30, y0 + 6 * rh, W - 30, y0 + 6 * rh, LIGHT, 1))
    b.append(text(cx, y0 + 6 * rh + 22, "final = 6", RED, 14, 800))
    b.append(text(cx, y0 + 6 * rh + 42, "(should be 7!)", RED, 11, 600))
    b.append(text(300, y0 + 6 * rh + 30, "both threads read 5 before either "
                  "wrote \u2192 one increment vanished", LIGHT, 11, 500,
                  italic=True))
    write("figures/race-condition.svg",
          svg(W, H, "".join(b), "Race condition"))


def fig_warp_shuffle():
    W, H = 780, 400
    b = [text(W / 2, 26, "Warp-shuffle reduction: registers, no shared memory",
              GREY, 15, 700)]
    lanes = 8
    lane_x = [110 + i * 78 for i in range(lanes)]
    y0 = 80
    rh = 74
    offs = [4, 2, 1]
    for i, x in enumerate(lane_x):
        b.append(box(x, y0, 58, 32, f"t{i}", GREY, size=11, rx=6))
    for step, off in enumerate(offs):
        y = y0 + (step + 1) * rh
        for i, x in enumerate(lane_x):
            active = i < off * (lanes // (2 * off)) if False else True
            keep = i < (lanes // (2 ** (step + 1))) * 2 ** 0
            col = BLUE if i < lanes // (2 ** (step + 1)) else "#E4E7EF"
            tc = WHITE if i < lanes // (2 ** (step + 1)) else LIGHT
            b.append(box(x, y, 58, 32, "\u03a3" if i < lanes //
                         (2 ** (step + 1)) else "\u00b7", col, tcol=tc,
                         size=12, rx=6))
            if i + off < lanes and i < lanes // (2 ** step):
                b.append(path(f"M{lane_x[i+off]+29} {y-rh+32} "
                              f"C{lane_x[i+off]+29} {y-14} {x+29} {y-14} "
                              f"{x+29} {y}", TEAL, RULE, arrow_end=True))
        b.append(text(70, y + 16, f"shfl_down {off}", LIGHT, 9, 600,
                      anchor="end"))
    b.append(text(W / 2, H - 24, "__shfl_down_sync halves the offset each step; "
                  "lane 0 ends with the warp sum", LIGHT, 11, 500,
                  italic=True))
    write("figures/warp-shuffle.svg",
          svg(W, H, "".join(b), "Warp shuffle reduction"))


# ── 13 streams & concurrency ────────────────────────────────────────────────
def fig_streams_overlap():
    W, H = 820, 360
    b = [text(W / 2, 26, "Streams overlap copy with compute", GREY, 16, 700)]
    unit = 60
    x0 = 130
    # serial
    b.append(text(30, 78, "serial", RED, 12, 700, anchor="start"))
    segs = [("H2D copy", GREY, 3), ("kernel", BLUE, 3), ("D2H copy", GREY, 3)]
    x = x0
    for name, col, d in segs:
        b.append(box(x, 66, d * unit, 34, name, col, size=11, rx=6))
        x += d * unit
    b.append(text(x + 14, 83, "GPU ~1/3 used", RED, 11, 700, anchor="start"))
    # pipelined
    b.append(text(30, 150, "3 streams", TEAL, 12, 700, anchor="start"))
    y0 = 130
    rh = 40
    for s in range(3):
        y = y0 + s * rh
        b.append(text(x0 - 14, y + 15, f"S{s}", GREY, 11, 700, anchor="end"))
        base = x0 + s * unit
        parts = [("H2D", GREY, 1), ("kernel", BLUE, 1), ("D2H", GREY, 1)]
        xx = base
        for name, col, d in parts:
            b.append(box(xx, y, d * unit - 4, 30, name, col, size=10, rx=5))
            xx += d * unit
    b.append(text(x0 + 6 * unit + 20, y0 + rh, "copy of one chunk overlaps",
                  TEAL, 11, 600, anchor="start"))
    b.append(text(x0 + 6 * unit + 20, y0 + rh + 16, "compute of another",
                  TEAL, 11, 600, anchor="start"))
    b.append(text(W / 2, H - 30, "same work, shorter wall-clock \u2014 copy "
                  "engines and SMs stay busy together", LIGHT, 11, 500,
                  italic=True))
    write("figures/streams-overlap.svg",
          svg(W, H, "".join(b), "Streams overlap"))


# ── batch 2 ═════════════════════════════════════════════════════════════════
# ── 00 introduction ─────────────────────────────────────────────────────────
def fig_prog_structure():
    W, H = 540, 430
    b = [text(W / 2, 26, "Typical CUDA program flow", GREY, 16, 700)]
    steps = [
        ("1. initialize data on the host", GREY),
        ("2. cudaMalloc on the device", BLUE),
        ("3. copy data  host \u2192 device", TEAL),
        ("4. launch kernel <<<grid, block>>>", BLUE),
        ("5. copy results  device \u2192 host", TEAL),
        ("6. cudaFree device memory", BLUE),
        ("7. use results on the host", GREY),
    ]
    y0, rh, bw, bh = 58, 50, 360, 36
    x = (W - bw) / 2
    for i, (lbl, col) in enumerate(steps):
        y = y0 + i * rh
        b.append(box(x, y, bw, bh, lbl, col, size=12, rx=8))
        if i:
            b.append(arrow(W / 2, y - rh + bh, W / 2, y, GREY, RULE))
    b.append(text(x - 14, y0 + 18, "HOST", LIGHT, 10, 700, anchor="end"))
    b.append(text(x - 14, y0 + rh + 18, "DEVICE", LIGHT, 10, 700, anchor="end"))
    write("figures/prog-structure.svg",
          svg(W, H, "".join(b), "CUDA program flow"))


# ── 02 first kernel ─────────────────────────────────────────────────────────
def fig_kernel_flow():
    W, H = 880, 280
    b = [text(W / 2, 26, "Two memory models, same kernel launch", GREY, 16,
              700)]

    def chain(y, label, steps):
        b.append(text(40, y - 24, label, GREY, 12, 700, anchor="start"))
        x = 40
        bw = 120
        for i, (t, col) in enumerate(steps):
            b.append(box(x, y, bw, 46, t, col, size=10, lh=13, rx=8))
            if i < len(steps) - 1:
                b.append(arrow(x + bw, y + 23, x + bw + 22, y + 23, GREY, RULE))
            x += bw + 22
    chain(84, "Explicit memory", [
        ("host init", GREY), ("cudaMalloc", BLUE), ("copy H2D", TEAL),
        ("launch <<<>>>", BLUE), ("copy D2H", TEAL), ("cudaFree", BLUE)])
    chain(200, "Unified memory", [
        ("cudaMalloc\nManaged", BLUE), ("host init", GREY),
        ("launch <<<>>>", BLUE), ("cudaDevice\nSynchronize", TEAL),
        ("read on host\n(auto-migrated)", GREY), ("cudaFree", BLUE)])
    write("figures/kernel-flow.svg",
          svg(W, H, "".join(b), "CUDA memory models"))


# ── 04 thread indexing ──────────────────────────────────────────────────────
def fig_grid_stride():
    W, H = 840, 320
    b = [text(W / 2, 26, "Grid-stride loop: few threads cover a big array",
              GREY, 15, 700)]
    cols = [BLUE, TEAL, AMBER, GREY_D]
    tcols = [WHITE, WHITE, INK_DARK, WHITE]
    n = 20
    cw, gap = 34, 4
    x0 = (W - (n * (cw + gap) - gap)) / 2
    for i in range(n):
        x = x0 + i * (cw + gap)
        t = i % 4
        b.append(box(x, 66, cw, 34, str(i), cols[t], tcol=tcols[t], size=11,
                     rx=5))
    # stride arrows for T0 (0,4,8,12,16)
    for i in range(0, 16, 4):
        x1 = x0 + i * (cw + gap) + cw / 2
        x2 = x0 + (i + 4) * (cw + gap) + cw / 2
        b.append(path(f"M{x1} 66 C{x1} 44 {x2} 44 {x2} 66", BLUE, RULE,
                      arrow_end=True))
    b.append(text(W / 2, 128, "stride = blockDim.x * gridDim.x  "
                  "(here 4 threads)", LIGHT, 11, 500, italic=True))
    # legend / assignments
    y = 168
    for t in range(4):
        yy = y + t * 30
        b.append(box(x0, yy, 24, 24, "", cols[t], rx=5))
        idxs = ", ".join(str(k) for k in range(t, n, 4))
        b.append(text(x0 + 36, yy + 12, f"Thread {t} processes: {idxs}",
                      GREY, 12, 600, anchor="start"))
    b.append(text(W / 2, H - 20, "for (i = idx; i < N; i += stride)  \u2014  "
                  "each thread strides across the array", LIGHT, 11, 500,
                  italic=True))
    write("figures/grid-stride.svg", svg(W, H, "".join(b), "Grid-stride loop"))


def fig_transpose():
    W, H = 780, 360
    b = [text(W / 2, 26, "Matrix transpose: A[i][j] \u2192 A\u1d40[j][i]",
              GREY, 15, 700)]

    def grid4(ox, oy, label, hot, accent):
        cell = 42
        b.append(text(ox + 2 * cell, oy - 12, label, GREY, 13, 700))
        for r in range(4):
            for c in range(4):
                x = ox + c * cell
                y = oy + r * cell
                fill = accent if (r, c) == hot else "none"
                tc = WHITE if (r, c) == hot else LIGHT
                b.append(rrect(x, y, cell - 3, cell - 3, fill, rx=4,
                              stroke=GREY, sw=1))
                b.append(text(x + (cell - 3) / 2, y + (cell - 3) / 2,
                              f"{r}{c}", tc, 10, 600))
    grid4(120, 90, "A  (read, coalesced)", (1, 3), BLUE)
    grid4(460, 90, "A\u1d40  (write, strided)", (3, 1), TEAL)
    b.append(arrow(300, 175, 450, 175, GREY, 2))
    b.append(text(375, 158, "swap i,j", GREY, 11, 600))
    b.append(text(W / 2, 300, "the naive write is strided (uncoalesced); stage "
                  "the tile in shared memory,", LIGHT, 11, 500, italic=True))
    b.append(text(W / 2, 320, "then write it back coalesced (pad to "
                  "[TILE][TILE+1] to avoid bank conflicts)", LIGHT, 11, 500,
                  italic=True))
    write("figures/transpose.svg", svg(W, H, "".join(b), "Matrix transpose"))


def fig_halo():
    W, H = 620, 480
    b = [text(W / 2, 26, "Halo cells: load a border of neighbours into shared "
              "memory", GREY, 14, 700)]
    n = 8
    cell = 40
    gx = (W - n * cell) / 2
    gy = 60
    for r in range(n):
        for c in range(n):
            x = gx + c * cell
            y = gy + r * cell
            inner = 2 <= r <= 5 and 2 <= c <= 5
            border = (1 <= r <= 6 and 1 <= c <= 6) and not inner
            fill = BLUE if inner else ("#DCE7F5" if border else "none")
            b.append(rrect(x, y, cell - 2, cell - 2, fill, rx=4, stroke=GREY,
                          sw=1))
    b.append(text(gx + 4 * cell, gy + n * cell + 26, "", GREY, 11, 600))
    b.append(box(gx, gy + n * cell + 14, 24, 24, "", BLUE, rx=5))
    b.append(text(gx + 34, gy + n * cell + 26, "output tile (this block "
                  "computes)", GREY, 12, 600, anchor="start"))
    b.append(rrect(gx, gy + n * cell + 46, 24, 24, "#DCE7F5", rx=5,
                  stroke=GREY, sw=1))
    b.append(text(gx + 34, gy + n * cell + 58, "halo (neighbours loaded for "
                  "stencils, read-only)", GREY, 12, 600, anchor="start"))
    write("figures/halo.svg", svg(W, H, "".join(b), "Halo tiling"))


def fig_row_major():
    W, H = 780, 340
    b = [text(W / 2, 26, "Row-major layout: 2D index \u2192 1D address", GREY,
              15, 700)]
    rows, colsn = 3, 4
    cell = 54
    gx, gy = 90, 60
    for r in range(rows):
        for c in range(colsn):
            x = gx + c * cell
            y = gy + r * cell
            b.append(rrect(x, y, cell - 4, cell - 4,
                          BLUE if (r == 1 and c == 2) else "none", rx=5,
                          stroke=GREY, sw=1))
            b.append(text(x + (cell - 4) / 2, y + (cell - 4) / 2, f"{r},{c}",
                          WHITE if (r == 1 and c == 2) else LIGHT, 11, 600))
    b.append(text(gx + colsn * cell / 2, gy - 14, "2D array [row][col]", GREY,
                  12, 700))
    # linear strip
    ly = 250
    lx0 = 60
    lw = 52
    for i in range(rows * colsn):
        x = lx0 + i * (lw + 4)
        hot = i == 1 * colsn + 2
        b.append(box(x, ly, lw, 34, str(i), BLUE if hot else GREY_D, size=11,
                     rx=5))
    b.append(text(W / 2, ly - 16, "1D memory (contiguous rows)", GREY, 12, 700))
    b.append(text(W / 2, ly + 62, "index = row \u00d7 width + col   "
                  "(here 1\u00d74 + 2 = 6)", TEAL, 12, 700))
    write("figures/row-major.svg", svg(W, H, "".join(b), "Row-major layout"))


# ── 06 memory management ────────────────────────────────────────────────────
def fig_mem_landscape():
    W, H = 880, 540
    b = [text(W / 2, 26, "The CUDA memory landscape", GREY, 16, 700)]
    # host
    b.append(obox(30, 48, 820, 120, "", GREY, GREY, rx=12))
    b.append(text(48, 66, "HOST (CPU)", GREY, 11, 700, anchor="start"))
    host = [("Pageable\n(malloc)", GREY_D, "staged via pinned buffer"),
            ("Pinned\n(cudaHostAlloc)", BLUE, "page-locked, direct DMA"),
            ("Managed / Unified\n(cudaMallocManaged)", TEAL,
             "auto-migrates on demand")]
    for i, (t, col, note) in enumerate(host):
        x = 60 + i * 265
        b.append(box(x, 78, 245, 54, t, col, size=11, lh=15, rx=8))
        b.append(text(x + 122, 148, note, LIGHT, 9, 500))
    # bus
    b.append(text(W / 2, 186, "\u2550\u2550\u2550  PCIe Gen5 / NVLink 5  "
                  "\u2550\u2550\u2550", GREY, 12, 700))
    # device
    b.append(obox(30, 206, 820, 310, "", GREY, GREY, rx=12))
    b.append(text(48, 224, "DEVICE (GPU)", GREY, 11, 700, anchor="start"))
    b.append(box(60, 236, 760, 40, "global memory (HBM3e / GDDR6X)  \u00b7  "
                 "~400-800 cyc  \u00b7  1-8 TB/s", GREY, size=12, rx=8))
    b.append(box(60, 284, 760, 34, "L2 cache  (6-96 MB, automatic)", BLUE_D,
                 size=12, rx=8))
    b.append(text(W / 2, 340, "per-SM resources", LIGHT, 11, 600))
    per = [("L1 / shared\n~5-30 cyc", BLUE), ("constant\ncache", GREY_D),
           ("texture\ncache", GREY_D), ("register file\n256 KB ~1 cyc", TEAL),
           ("TMEM\n(Blackwell)", AMBER)]
    pw = 148
    for i, (t, col) in enumerate(per):
        x = 60 + i * (pw + 4)
        tc = INK_DARK if col == AMBER else WHITE
        b.append(box(x, 356, pw, 60, t, col, tcol=tc, size=10, lh=14, rx=8))
    b.append(text(W / 2, 440, "registers/local are per-thread \u00b7 shared is "
                  "per-block \u00b7 L2/global are device-wide", LIGHT, 11, 500,
                  italic=True))
    b.append(text(W / 2, 462, "pick the closest, fastest space your access "
                  "pattern allows", LIGHT, 11, 500, italic=True))
    write("figures/mem-landscape.svg",
          svg(W, H, "".join(b), "Memory landscape"))


def fig_pinned():
    W, H = 780, 320
    b = [text(W / 2, 26, "Pinned host memory transfers faster than pageable",
              GREY, 15, 700)]
    rows = [("Pageable (PCIe 4)", 7, GREY),
            ("Pinned (PCIe 4)", 25, BLUE),
            ("Pinned (PCIe 5)", 50, BLUE_D),
            ("Pinned (NVLink)", 450, TEAL)]
    x0, y0, rh = 200, 70, 46
    barmax = 520
    vmax = 450
    import math as _m
    for i, (lbl, val, col) in enumerate(rows):
        y = y0 + i * rh
        w = barmax * (_m.log10(val + 1) / _m.log10(vmax + 1))
        b.append(text(x0 - 12, y + 16, lbl, GREY, 11, 600, anchor="end"))
        b.append(rrect(x0, y, barmax, 26, "#E4E7EF", rx=8))
        b.append(rrect(x0, y, max(40, w), 26, col, rx=8))
        b.append(text(x0 + max(40, w) + 8, y + 13, f"~{val} GB/s", GREY, 11,
                      700, anchor="start"))
    b.append(text(W / 2, H - 30, "3-4\u00d7 faster with pinned memory; log "
                  "scale (NVLink dwarfs PCIe)", LIGHT, 11, 500, italic=True))
    write("figures/pinned.svg", svg(W, H, "".join(b), "Pinned bandwidth"))


def fig_unified_memory():
    W, H = 780, 320
    b = [text(W / 2, 26, "Unified memory migrates pages on demand", GREY, 15,
              700)]
    b.append(box(70, 100, 200, 120, ["CPU", "", "accesses data"], GREY,
                 size=13, lh=22, rx=12))
    b.append(box(510, 100, 200, 120, ["GPU", "", "accesses data"], BLUE,
                 size=13, lh=22, rx=12))
    b.append(box(320, 120, 140, 80, ["managed", "page"], TEAL, size=12, lh=18,
                 rx=10))
    b.append(path("M320 140 C280 120 300 120 270 130", TEAL, 2, arrow_end=True))
    b.append(path("M460 130 C500 120 520 120 510 135", TEAL, 2,
                  arrow_end=True))
    b.append(text(W / 2, 236, "page fault \u2192 driver migrates the page to "
                  "whoever touched it", RED, 11, 700))
    b.append(text(W / 2, 262, "one pointer for both sides; no explicit "
                  "cudaMemcpy \u2014 but faults cost latency", LIGHT, 11, 500,
                  italic=True))
    b.append(text(W / 2, 284, "use cudaMemPrefetchAsync / cudaMemAdvise to "
                  "avoid fault storms", LIGHT, 11, 500, italic=True))
    write("figures/unified-memory.svg",
          svg(W, H, "".join(b), "Unified memory"))


# ── 09 work allocation ──────────────────────────────────────────────────────
def fig_sw_hw_mapping():
    W, H = 800, 380
    b = [text(W / 2, 26, "Software maps onto hardware", GREY, 16, 700)]
    b.append(text(210, 58, "your code", GREY, 12, 700))
    b.append(text(590, 58, "hardware", GREY, 12, 700))
    pairs = [("Grid", "whole GPU", BLUE),
             ("Block", "an SM", BLUE),
             ("Warp (32 threads)", "warp scheduler", TEAL),
             ("Thread", "CUDA core", GREY)]
    y0, rh = 80, 74
    for i, (sw, hw, col) in enumerate(pairs):
        y = y0 + i * rh
        b.append(box(90, y, 240, 50, sw, col, size=13, rx=9))
        b.append(box(470, y, 240, 50, hw, GREY_D, size=13, rx=9))
        b.append(arrow(330, y + 25, 470, y + 25, GREY, 2))
    b.append(text(W / 2, H - 24, "the runtime assigns blocks to SMs as "
                  "resources free up; you never pick the SM", LIGHT, 11, 500,
                  italic=True))
    write("figures/sw-hw-mapping.svg",
          svg(W, H, "".join(b), "Software to hardware mapping"))


def fig_sm_resources():
    W, H = 820, 360
    b = [text(W / 2, 26, "Per-SM resources are partitioned into whole blocks",
              GREY, 15, 700)]
    # each row: capacity divided into block-sized segments
    rows = [("Threads", 2048, 256, 8, BLUE, "256/block"),
            ("Registers", 65536, 8192, 8, BLUE_D, "8192/block"),
            ("Shared mem", 100, 16, 6, TEAL, "16 KB/block")]
    x0, y0, rh, barw = 150, 70, 78, 560
    for ri, (name, cap, per, fit, col, note) in enumerate(rows):
        y = y0 + ri * rh
        b.append(text(x0 - 14, y + 20, name, GREY, 12, 700, anchor="end"))
        b.append(rrect(x0, y, barw, 40, "#E4E7EF", rx=8))
        seg = barw / (cap / per)
        for k in range(fit):
            b.append(rrect(x0 + k * seg + 2, y + 2, seg - 4, 36, col, rx=5))
            b.append(text(x0 + k * seg + seg / 2, y + 20, str(k + 1), WHITE, 10,
                          700))
        b.append(text(x0 + barw + 10, y + 20, f"{fit} blocks", GREY, 11, 600,
                      anchor="start"))
        b.append(text(x0 + barw + 10, y + 34, note, LIGHT, 9, 500,
                      anchor="start"))
    b.append(text(W / 2, H - 22, "shared memory runs out first \u2192 only 6 "
                  "blocks/SM (the occupancy bottleneck)", RED, 12, 700))
    write("figures/sm-resources.svg",
          svg(W, H, "".join(b), "SM resource partitioning"))


def fig_blocksize_decision():
    W, H = 840, 420
    b = [text(W / 2, 26, "Choosing a block size", GREY, 16, 700)]
    b.append(box(330, 54, 180, 44, "Is the problem 2D/3D?", BLUE, size=12,
                 rx=10))
    # yes branch
    b.append(arrow(420, 98, 200, 140, GREY))
    b.append(text(290, 118, "yes", GREY, 10, 700))
    b.append(box(100, 140, 200, 60, ["2D: 16\u00d716 or 32\u00d732",
                 "3D: 8\u00d78\u00d74 or 4\u00d74\u00d716"], TEAL, size=11,
                 lh=16, rx=10))
    # no branch
    b.append(arrow(440, 98, 560, 140, GREY))
    b.append(text(560, 118, "no", GREY, 10, 700))
    b.append(box(470, 140, 200, 44, "Use shared memory?", BLUE, size=12,
                 rx=10))
    b.append(arrow(520, 184, 430, 240, GREY))
    b.append(text(450, 214, "yes", GREY, 10, 700))
    b.append(box(320, 240, 220, 64, ["< 128 B/thread \u2192 256-512",
                 "> 128 B/thread \u2192 128-256"], GREY_D, size=11, lh=16,
                 rx=10))
    b.append(arrow(610, 184, 690, 240, GREY))
    b.append(text(670, 214, "no", GREY, 10, 700))
    b.append(box(590, 240, 220, 64, ["many registers \u2192 128-256",
                 "otherwise \u2192 256-512"], GREY_D, size=11, lh=16, rx=10))
    b.append(text(W / 2, 350, "then grid size: small N \u2192 ceil(N/block); "
                  "large N \u2192 numSMs \u00d7 8-16 with a grid-stride loop",
                  LIGHT, 11, 500, italic=True))
    b.append(text(W / 2, 374, "256 threads/block is a solid default", LIGHT, 11,
                  600, italic=True))
    write("figures/blocksize-decision.svg",
          svg(W, H, "".join(b), "Block size decision"))


# ── 10 gpu architecture ─────────────────────────────────────────────────────
def fig_tensor_core():
    W, H = 780, 340
    b = [text(W / 2, 26, "Tensor Core: one fused matrix multiply-accumulate",
              GREY, 15, 700)]
    b.append(text(W / 2, 54, "D = A \u00d7 B + C", GREY, 14, 700, mono=True))

    def mat(ox, oy, label, col):
        cell = 26
        b.append(text(ox + cell, oy - 10, label, GREY, 11, 700))
        for r in range(2):
            for c in range(2):
                b.append(rrect(ox + c * cell, oy + r * cell, cell - 3,
                              cell - 3, col, rx=3))
    mat(70, 90, "A", BLUE)
    b.append(text(140, 116, "\u00d7", GREY, 20, 800))
    mat(165, 90, "B", BLUE)
    b.append(text(235, 116, "+", GREY, 20, 800))
    mat(260, 90, "C", GREY)
    b.append(arrow(320, 116, 380, 116, GREY, 2))
    b.append(box(390, 84, 320, 90, ["Tensor Core (4\u00d74\u00d74 MMA)",
                 "16 multiplies \u2192 reduction tree \u2192 + C",
                 "64 FMA in ~8 cycles"], TEAL, size=11, lh=18, rx=10))
    b.append(arrow(550, 174, 586, 200, GREY, RULE))
    mat(560, 200, "D", AMBER)
    b.append(text(W / 2, 250, "per SM (4 Tensor Cores): ~1024 FP16 ops/clock",
                  GREY, 12, 700))
    b.append(text(W / 2, 274, "~16\u00d7 faster than FP32 cores; supports FP16 "
                  "BF16 TF32 FP8 INT8 (and FP4/FP6 on Blackwell)", LIGHT, 11,
                  500, italic=True))
    write("figures/tensor-core.svg", svg(W, H, "".join(b), "Tensor Core MMA"))


def fig_smem_banks():
    W, H = 640, 440
    b = [text(W / 2, 26, "Shared memory: 32 banks, one word each per cycle",
              GREY, 14, 700)]
    tx0, tw = 60, 46

    def panel(oy, title, mapping, cyc, ccol):
        b.append(text(60, oy, title, ccol, 12, 700, anchor="start"))
        for i in range(8):
            x = tx0 + i * tw
            b.append(box(x, oy + 14, tw - 4, 26, f"T{i}", GREY, size=10, rx=5))
            b.append(rrect(x, oy + 66, tw - 4, 26, "#DCE7F5", rx=5,
                          stroke=GREY, sw=1))
            b.append(text(x + (tw - 4) / 2, oy + 79, f"B{i}", GREY, 9, 600))
        for ti, bi in mapping:
            x1 = tx0 + ti * tw + (tw - 4) / 2
            x2 = tx0 + bi * tw + (tw - 4) / 2
            b.append(arrow(x1, oy + 42, x2, oy + 64, ccol, RULE))
        b.append(text(W - 30, oy + 58, cyc, ccol, 12, 700, anchor="end"))

    panel(66, "no conflict \u2014 different banks",
          [(i, i) for i in range(8)], "1 cycle", TEAL)
    panel(184, "2-way conflict \u2014 two threads / bank",
          [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 3), (7, 3)],
          "2 cycles", RED)
    panel(302, "broadcast \u2014 same address",
          [(i, 0) for i in range(8)], "1 cycle", BLUE)
    b.append(text(W / 2, H - 20, "pad arrays ([TILE][TILE+1]) so a column no "
                  "longer lands in one bank", LIGHT, 11, 500, italic=True))
    write("figures/smem-banks.svg", svg(W, H, "".join(b), "Shared memory banks"))


def fig_core_pipeline():
    W, H = 880, 240
    b = [text(W / 2, 26, "FP32 CUDA core pipeline", GREY, 16, 700)]
    stages = [("FETCH", "instr + PC", GREY),
              ("DECODE", "operands", GREY),
              ("READ REGS", "register file", GREY),
              ("EXECUTE", "FMA: a\u00d7b + c", BLUE),
              ("WRITEBACK", "to registers", GREY)]
    bw, bh = 150, 84
    gap = 22
    total = len(stages) * bw + (len(stages) - 1) * gap
    x0 = (W - total) / 2
    y = 74
    for i, (t, sub, col) in enumerate(stages):
        x = x0 + i * (bw + gap)
        b.append(box(x, y, bw, bh, [t, sub], col, size=12, lh=20, rx=10))
        if i < len(stages) - 1:
            b.append(arrow(x + bw, y + bh / 2, x + bw + gap, y + bh / 2, GREY,
                           2))
    b.append(text(W / 2, 190, "pipelined: ~4-cycle latency but ~1 result per "
                  "cycle (2 ops/cycle counting the FMA)", LIGHT, 11, 500,
                  italic=True))
    write("figures/core-pipeline.svg",
          svg(W, H, "".join(b), "CUDA core pipeline"))


def fig_warp_scheduler():
    W, H = 820, 360
    b = [text(W / 2, 26, "A warp scheduler issues one ready warp per cycle",
              GREY, 15, 700)]
    # warp pool
    b.append(text(120, 60, "resident warps", GREY, 12, 700))
    states = [("W0", "ready", TEAL), ("W1", "stalled", "#E4E7EF"),
              ("W2", "ready", TEAL), ("W3", "stalled", "#E4E7EF"),
              ("W4", "ready", TEAL), ("W5", "stalled", "#E4E7EF")]
    for i, (w, st, col) in enumerate(states):
        y = 78 + i * 40
        tc = WHITE if col == TEAL else LIGHT
        b.append(box(60, y, 150, 32, f"{w}  \u00b7  {st}", col, tcol=tc,
                     size=11, rx=6))
    b.append(box(280, 150, 150, 60, ["warp", "scheduler"], BLUE, size=12,
                 lh=18, rx=10))
    b.append(arrow(210, 94, 280, 165, TEAL, RULE))
    b.append(arrow(210, 254, 280, 195, TEAL, RULE))
    # dispatch ports
    ports = [("FP32", BLUE_D), ("INT32", BLUE_D), ("LD/ST", GREY),
             ("Tensor", TEAL)]
    for i, (p, col) in enumerate(ports):
        y = 96 + i * 44
        b.append(box(560, y, 180, 34, p, col, size=11, rx=8))
        b.append(arrow(430, 180, 560, y + 17, GREY, RULE))
    b.append(text(W / 2, H - 22, "stalled warps (waiting on memory) are "
                  "skipped \u2014 zero-cost context switch hides latency",
                  LIGHT, 11, 500, italic=True))
    write("figures/warp-scheduler.svg",
          svg(W, H, "".join(b), "Warp scheduler"))


# ── 12 atomics & synchronization ────────────────────────────────────────────
def fig_atomic_contention():
    W, H = 780, 340
    b = [text(W / 2, 26, "A lock serializes a whole warp", GREY, 16, 700)]
    b.append(text(W / 2, 54, "32 threads in lockstep; only one holds the lock",
                  LIGHT, 12, 500, italic=True))
    cols = 8
    for i in range(16):
        r, c = divmod(i, cols)
        x = 90 + c * 78
        y = 84 + r * 50
        work = i == 0
        b.append(box(x, y, 66, 38, "T0\ncritical" if work else f"T{i}\nspin",
                     TEAL if work else RED, size=9, lh=12, rx=6))
    b.append(text(W / 2, 210, "\u2026 (threads 16-31 also spinning) \u2026",
                  LIGHT, 10, 500, italic=True))
    # efficiency bar
    b.append(text(150, 250, "useful work", GREY, 11, 600, anchor="end"))
    b.append(rrect(160, 238, 520, 24, "#E4E7EF", rx=8))
    b.append(rrect(160, 238, 520 / 32, 24, TEAL, rx=8))
    b.append(text(W / 2, 290, "1/32 = 3.1% efficiency \u2014 96.9% of cycles "
                  "wasted spinning", RED, 12, 700))
    b.append(text(W / 2, 314, "prefer atomics, reductions, or lock-free "
                  "designs over per-thread locks", LIGHT, 11, 500,
                  italic=True))
    write("figures/atomic-contention.svg",
          svg(W, H, "".join(b), "Lock contention"))


def fig_syncthreads():
    W, H = 780, 320
    b = [text(W / 2, 26, "__syncthreads(): a block-wide barrier", GREY, 16,
              700)]
    lanes = 6
    lane_x = [120 + i * 100 for i in range(lanes)]
    for i, x in enumerate(lane_x):
        b.append(box(x - 30, 66, 60, 30, f"T{i}", GREY, size=11, rx=6))
        b.append(box(x - 34, 108, 68, 30, "write", BLUE, size=10, rx=6))
        b.append(arrow(x, 138, x, 168, GREY, RULE))
    b.append(box(70, 170, 640, 40, "__syncthreads()  \u2014  all threads must "
                 "arrive before any continues", BLUE_D, size=12, rx=10))
    for i, x in enumerate(lane_x):
        b.append(arrow(x, 210, x, 240, GREY, RULE))
        b.append(box(x - 34, 240, 68, 30, "read", TEAL, size=10, rx=6))
    b.append(text(W / 2, H - 22, "guarantees shared-memory writes are visible "
                  "before the reads; every thread must reach it", LIGHT, 11,
                  500, italic=True))
    write("figures/syncthreads.svg", svg(W, H, "".join(b), "syncthreads barrier"))


# ── 15 advanced memory ──────────────────────────────────────────────────────
def fig_async_copy():
    W, H = 820, 300
    b = [text(W / 2, 26, "Async copy double-buffers to hide load latency",
              GREY, 15, 700)]
    unit = 70
    x0 = 150
    # sync
    b.append(text(30, 82, "synchronous", RED, 12, 700, anchor="start"))
    segs = [("load 0", GREY, 1), ("stall", "#E4E7EF", 1), ("compute 0", BLUE, 1),
            ("load 1", GREY, 1), ("stall", "#E4E7EF", 1), ("compute 1", BLUE, 1)]
    x = x0
    for name, col, d in segs:
        tc = LIGHT if col == "#E4E7EF" else WHITE
        b.append(box(x, 66, d * unit - 4, 32, name, col, tcol=tc, size=9, rx=5))
        x += d * unit
    # async
    b.append(text(30, 160, "async / double", TEAL, 12, 700, anchor="start"))
    b.append(text(30, 176, "buffered", TEAL, 12, 700, anchor="start"))
    b.append(text(x0 - 14, 160, "copy", GREY, 10, 700, anchor="end"))
    for k in range(3):
        b.append(box(x0 + k * unit, 148, unit - 4, 28, f"cp {k}", GREY,
                     size=9, rx=5))
    b.append(text(x0 - 14, 196, "compute", GREY, 10, 700, anchor="end"))
    for k in range(3):
        b.append(box(x0 + (k + 1) * unit, 184, unit - 4, 28, f"comp {k}", BLUE,
                     size=9, rx=5))
    b.append(text(W / 2, 250, "compute tile k while tile k+1 copies in "
                  "(cp.async / cuda::memcpy_async) \u2014 no stall", LIGHT, 11,
                  500, italic=True))
    write("figures/async-copy.svg", svg(W, H, "".join(b), "Async double buffer"))


# ── 16 cuda graphs ──────────────────────────────────────────────────────────
def fig_cuda_graph():
    W, H = 640, 300
    b = [text(W / 2, 26, "A CUDA graph is a DAG of operations", GREY, 16, 700)]
    b.append(box(60, 120, 100, 50, "A", GREY, size=14, rx=10))
    b.append(box(270, 60, 110, 50, "kernel B", BLUE, size=12, rx=10))
    b.append(box(270, 180, 110, 50, "kernel C", BLUE, size=12, rx=10))
    b.append(box(490, 120, 100, 50, "D", GREY, size=14, rx=10))
    b.append(arrow(160, 135, 270, 90, GREY))
    b.append(arrow(160, 155, 270, 200, GREY))
    b.append(arrow(380, 85, 490, 135, GREY))
    b.append(arrow(380, 205, 490, 155, GREY))
    b.append(text(325, 145, "B, C run", TEAL, 11, 700))
    b.append(text(325, 161, "concurrently", TEAL, 11, 700))
    b.append(text(W / 2, 262, "define once, launch many times: the runtime "
                  "schedules it with almost no per-launch CPU overhead", LIGHT,
                  11, 500, italic=True))
    write("figures/cuda-graph.svg", svg(W, H, "".join(b), "CUDA graph DAG"))


# ── 14 advanced kernel techniques ───────────────────────────────────────────
def fig_dynamic_parallelism():
    W, H = 720, 300
    b = [text(W / 2, 26, "Dynamic parallelism: kernels launch kernels", GREY,
              15, 700)]
    b.append(box(60, 130, 150, 50, "parent kernel", BLUE, size=12, rx=10))
    for i in range(2):
        y = 80 + i * 100
        b.append(box(300, y, 150, 44, f"child {i}", TEAL, size=12, rx=10))
        b.append(arrow(210, 155, 300, y + 22, GREY))
        b.append(box(520, y, 150, 40, "grandchild", GREY_D, size=11, rx=10))
        b.append(arrow(450, y + 22, 520, y + 20, GREY))
    b.append(text(W / 2, 262, "great for irregular/recursive work, but each "
                  "launch has overhead \u2014 measure vs a flat kernel", LIGHT,
                  11, 500, italic=True))
    write("figures/dynamic-parallelism.svg",
          svg(W, H, "".join(b), "Dynamic parallelism"))


# ── 17 multi-GPU ────────────────────────────────────────────────────────────
def fig_interconnect():
    W, H = 800, 340
    b = [text(W / 2, 26, "Interconnect bandwidth hierarchy", GREY, 16, 700)]
    rows = [("on-GPU HBM3e", 4000, TEAL),
            ("NVLink 5 (GPU\u2194GPU)", 1800, BLUE),
            ("PCIe Gen5", 64, AMBER),
            ("network (InfiniBand 400G)", 50, RED)]
    x0, y0, rh = 260, 74, 56
    barmax = 450
    import math as _m
    vmax = 4000
    for i, (lbl, val, col) in enumerate(rows):
        y = y0 + i * rh
        w = barmax * (_m.log10(val + 1) / _m.log10(vmax + 1))
        tc = INK_DARK if col == AMBER else WHITE
        b.append(text(x0 - 12, y + 18, lbl, GREY, 11, 600, anchor="end"))
        b.append(rrect(x0, y, barmax, 34, "#E4E7EF", rx=8))
        b.append(rrect(x0, y, max(50, w), 34, col, rx=8))
        unit = "GB/s" if val < 1000 else "GB/s"
        b.append(text(x0 + max(50, w) + 8, y + 17, f"~{val} {unit}", GREY, 11,
                      700, anchor="start"))
    b.append(text(W / 2, H - 24, "keep chatty GPUs on NVLink; PCIe and the "
                  "network are ~30-80\u00d7 slower (log scale)", LIGHT, 11, 500,
                  italic=True))
    write("figures/interconnect.svg",
          svg(W, H, "".join(b), "Interconnect bandwidth"))


def fig_parallelism():
    W, H = 880, 340
    b = [text(W / 2, 26, "Multi-GPU parallelism strategies", GREY, 16, 700)]

    def panel(ox, title, sub, draw):
        b.append(text(ox + 130, 58, title, BLUE, 13, 700))
        b.append(text(ox + 130, 76, sub, LIGHT, 10, 500, italic=True))
        draw(ox)
    # data parallel
    def d1(ox):
        for i in range(4):
            y = 96 + i * 44
            b.append(box(ox + 40, y, 180, 34,
                         f"GPU{i}: full model \u00b7 batch[{i}]", TEAL,
                         size=9, rx=6))
        b.append(text(ox + 130, 288, "AllReduce gradients", GREY, 10, 700))
    # tensor parallel
    def d2(ox):
        b.append(box(ox + 40, 100, 180, 40, "layer weights", GREY, size=11,
                     rx=8))
        for i in range(4):
            x = ox + 40 + i * 46
            b.append(box(x, 160, 42, 60, f"GPU\n{i}", BLUE, size=10, lh=14,
                         rx=6))
            b.append(arrow(ox + 130, 140, x + 21, 160, GREY, RULE))
        b.append(text(ox + 130, 260, "split each matmul;", GREY, 10, 600))
        b.append(text(ox + 130, 276, "heavy NVLink comm", GREY, 10, 600))
    # pipeline parallel
    def d3(ox):
        for i in range(4):
            y = 100 + i * 40
            b.append(box(ox + 40, y, 180, 30, f"GPU{i}: layers {i*3}-{i*3+2}",
                         GREY_D, size=9, rx=6))
            if i < 3:
                b.append(arrow(ox + 130, y + 30, ox + 130, y + 40, GREY, RULE))
        b.append(text(ox + 130, 280, "micro-batches flow through", GREY, 10,
                      600))
    panel(0, "Data parallel", "split the batch", d1)
    panel(300, "Tensor parallel", "split the model", d2)
    panel(600, "Pipeline parallel", "split the layers", d3)
    write("figures/parallelism.svg",
          svg(W, H, "".join(b), "Parallelism strategies"))


# ── 18 profiling & debugging ────────────────────────────────────────────────
def fig_profiling_stack():
    W, H = 640, 440
    b = [text(W / 2, 26, "The debugging & profiling workflow", GREY, 16, 700)]
    steps = [("1. write code", GREY),
             ("2. error checks: CUDA_CHECK(), assert()", GREY),
             ("3. compute-sanitizer  (memory / races)", BLUE),
             ("4. cuda-gdb  (breakpoints, inspection)", BLUE)]
    y0, rh, bw = 60, 56, 460
    x = (W - bw) / 2
    for i, (lbl, col) in enumerate(steps):
        y = y0 + i * rh
        b.append(box(x, y, bw, 40, lbl, col, size=12, rx=9))
        b.append(arrow(W / 2, y + 40, W / 2, y + rh, GREY, RULE))
    y = y0 + 4 * rh
    b.append(text(W / 2, y + 8, "5. profile", GREY, 12, 700))
    b.append(box(x, y + 20, 220, 54, ["Nsight Systems", "(timeline / overlap)"],
                 TEAL, size=11, lh=16, rx=9))
    b.append(box(x + 240, y + 20, 220, 54, ["Nsight Compute",
                 "(per-kernel detail)"], TEAL, size=11, lh=16, rx=9))
    write("figures/profiling-stack.svg",
          svg(W, H, "".join(b), "Profiling workflow"))


def fig_test_pyramid():
    W, H = 640, 360
    b = [text(W / 2, 26, "The testing pyramid for CUDA", GREY, 16, 700)]
    cx = W / 2
    levels = [("Integration tests", "few, slow", 120, TEAL, 70),
              ("Kernel tests (GPU-specific)", "some", 260, BLUE, 150),
              ("Unit tests (host code, utils)", "many, fast", 420, GREY, 240)]
    for lbl, note, w, col, y in levels:
        b.append(rrect(cx - w / 2, y, w, 70, col, rx=8))
        b.append(text(cx, y + 28, lbl, WHITE, 11, 700))
        b.append(text(cx, y + 48, note, WHITE, 10, 500))
    b.append(text(W / 2, H - 24, "push logic into host-testable units; keep "
                  "slow GPU integration tests few", LIGHT, 11, 500,
                  italic=True))
    write("figures/test-pyramid.svg", svg(W, H, "".join(b), "Testing pyramid"))


def fig_opt_workflow():
    W, H = 820, 320
    b = [text(W / 2, 26, "Optimize in priority order (biggest wins first)",
              GREY, 15, 700)]
    rows = [("1. Algorithm", "10-1000\u00d7", 1000, TEAL),
            ("2. Memory access", "2-10\u00d7", 10, BLUE),
            ("3. Occupancy", "1.5-3\u00d7", 3, BLUE_D),
            ("4. Instructions", "1.2-2\u00d7", 2, GREY)]
    x0, y0, rh = 180, 70, 56
    barmax = 520
    import math as _m
    for i, (lbl, rng, val, col) in enumerate(rows):
        y = y0 + i * rh
        w = barmax * (_m.log10(val) / _m.log10(1000))
        b.append(text(x0 - 12, y + 18, lbl, GREY, 12, 700, anchor="end"))
        b.append(rrect(x0, y, barmax, 34, "#E4E7EF", rx=8))
        b.append(rrect(x0, y, max(60, w), 34, col, rx=8))
        b.append(text(x0 + 10, y + 17, rng, WHITE, 11, 700, anchor="start"))
    b.append(text(W / 2, H - 22, "a better algorithm beats any amount of "
                  "micro-tuning (log scale)", LIGHT, 11, 500, italic=True))
    write("figures/opt-workflow.svg",
          svg(W, H, "".join(b), "Optimization priority"))


# ── 19 optimization case studies ────────────────────────────────────────────
def fig_reduction_evolution():
    W, H = 820, 340
    b = [text(W / 2, 26, "Evolution of a reduction kernel", GREY, 16, 700)]
    rows = [("V1 naive", 1.0, "baseline", GREY),
            ("V2 coalesced", 3.9, "fix memory", BLUE),
            ("V3 no divergence", 11.4, "fix divergence", BLUE),
            ("V4 unrolled", 27.8, "cut loop overhead", BLUE_D),
            ("V5 warp shuffle", 208.3, "registers, no smem", TEAL)]
    x0, y0, rh = 180, 66, 50
    barmax = 380
    import math as _m
    vmax = 208.3
    for i, (lbl, val, note, col) in enumerate(rows):
        y = y0 + i * rh
        w = barmax * (_m.log10(val) / _m.log10(vmax)) if val > 1 else 6
        b.append(text(x0 - 12, y + 16, lbl, GREY, 11, 700, anchor="end"))
        b.append(rrect(x0, y, barmax, 28, "#E4E7EF", rx=8))
        b.append(rrect(x0, y, max(6, w), 28, col, rx=8))
        b.append(text(x0 + max(6, w) + 8, y + 14, f"{val:g}\u00d7  \u00b7  "
                      f"{note}", GREY, 10, 600, anchor="start"))
    b.append(text(W / 2, H - 22, "same output, ~200\u00d7 faster through "
                  "stacked memory + divergence + warp-level fixes (log scale)",
                  LIGHT, 11, 500, italic=True))
    write("figures/reduction-evolution.svg",
          svg(W, H, "".join(b), "Reduction evolution"))


# ── 21 modern CUDA ──────────────────────────────────────────────────────────
def fig_clusters():
    W, H = 780, 340
    b = [text(W / 2, 26, "Thread-block clusters (Hopper sm_90+)", GREY, 15,
              700)]
    chain = ["grid", "cluster", "block", "warp", "thread"]
    x = 70
    for i, c in enumerate(chain):
        col = TEAL if c == "cluster" else GREY
        b.append(box(x, 58, 110, 36, c, col, size=12, rx=8))
        if i < len(chain) - 1:
            b.append(arrow(x + 110, 76, x + 130, 76, GREY, RULE))
        x += 140
    b.append(text(190, 92, "new, optional level", TEAL, 10, 600))
    b.append(obox(160, 130, 460, 150, "", TEAL, TEAL, rx=12))
    b.append(text(180, 150, "cluster", TEAL, 12, 700, anchor="start"))
    for i in range(2):
        bx = 210 + i * 220
        b.append(box(bx, 170, 160, 90, "", BLUE, rx=10))
        b.append(text(bx + 80, 186, f"block {i}", WHITE, 11, 700))
        b.append(box(bx + 20, 202, 120, 40, "shared mem", BLUE_D, size=10,
                     rx=6))
    b.append(path("M370 222 C400 222 420 222 430 222", TEAL, 2,
                  arrow_end=True))
    b.append(path("M430 236 C400 236 380 236 370 236", TEAL, 2,
                  arrow_end=True))
    b.append(text(400, 156, "DSMEM", TEAL, 10, 700))
    b.append(text(W / 2, 306, "blocks in a cluster read each other's shared "
                  "memory (DSMEM) and cluster.sync() together", LIGHT, 11, 500,
                  italic=True))
    write("figures/clusters.svg", svg(W, H, "".join(b), "Thread-block clusters"))


# ── batch 3: remaining diagrams ──────────────────────────────────────────────
def fig_index_3d():
    W, H = 900, 470
    b = [text(W / 2, 26, "3D indexing: a grid of blocks, a block of threads",
              GREY, 16, 700)]
    # grid: two z-layers of 2x2 blocks
    b.append(text(210, 58, "Grid \u2014 2\u00d72\u00d72 blocks", GREY, 12.5, 700))
    for li, (lz, col, lbl) in enumerate([(0, BLUE, "Z=0 (bottom)"),
                                         (1, BLUE_D, "Z=1 (top)")]):
        ox = 60 + li * 210
        b.append(text(ox + 85, 82, lbl, LIGHT, 10, 600))
        for j in range(2):
            for i in range(2):
                x = ox + i * 90
                y = 96 + (1 - j) * 64
                b.append(box(x, y, 84, 58, [f"({i},{j},{lz})"], col, size=10))
    # block: one 4x4 slice of threads
    b.append(text(660, 58, "Block \u2014 4\u00d74\u00d74 threads", GREY, 12.5, 700))
    b.append(text(660, 82, "one Z-slice (16 of 64 threads)", LIGHT, 10, 600))
    for r in range(4):
        for c in range(4):
            b.append(circle(600 + c * 32, 110 + r * 32, 10, "#BCD3EE"))
    # formula card
    b.append(rrect(60, 300, 780, 128, "#232A35", rx=12, stroke=GREY_D,
                   sw=1.75))
    b.append(text(80, 322, "GLOBAL POSITION", "#7FC4FF", 11, 700,
                  anchor="start"))
    for i, s in enumerate(["x = blockIdx.x * blockDim.x + threadIdx.x",
                           "y = blockIdx.y * blockDim.y + threadIdx.y",
                           "z = blockIdx.z * blockDim.z + threadIdx.z"]):
        b.append(text(80, 348 + i * 22, s, "#D7DCE6", 12, 500, anchor="start",
                      mono=True))
    b.append(text(470, 322, "LINEAR INDEX", "#83CEA3", 11, 700, anchor="start"))
    b.append(text(470, 350, "idx = z*(W*H) + y*W + x", "#83CEA3", 12.5, 600,
                  anchor="start", mono=True))
    b.append(text(470, 392, "e.g. Block(1,0,1) Thread(2,3,1)", LIGHT, 10.5,
                  500, anchor="start"))
    b.append(text(470, 410, "\u2192 (6, 3, 5) in the volume", LIGHT, 10.5, 500,
                  anchor="start"))
    write("figures/index-3d.svg", svg(W, H, "".join(b), "3D thread indexing"))


def fig_checkerboard():
    W, H = 760, 470
    b = [text(W / 2, 26, "Checkerboard (red-black) update pattern",
              GREY, 16, 700)]
    n, cell, ox, oy = 8, 40, 220, 56
    for r in range(n):
        for c in range(n):
            red = (r + c) % 2 == 0
            col = RED if red else GREY_D
            b.append(rrect(ox + c * cell, oy + r * cell, cell - 2, cell - 2,
                           col, rx=3))
    b.append(text(ox + n * cell / 2, oy + n * cell + 18,
                  "0   1   2   3   4   5   6   7", LIGHT, 10, 600))
    b.append(rrect(120, 400, 26, 26, RED, rx=4))
    b.append(text(156, 413, "red \u2014 phase 1", GREY, 11, 600, anchor="start"))
    b.append(rrect(430, 400, 26, 26, GREY_D, rx=4))
    b.append(text(466, 413, "black \u2014 phase 2", GREY, 11, 600, anchor="start"))
    b.append(text(W / 2, 448,
                  "update all red cells, sync, then black cells reuse the "
                  "fresh red values \u2014 no races", LIGHT, 11, 500))
    write("figures/checkerboard.svg",
          svg(W, H, "".join(b), "Checkerboard access"))


def fig_smem_reuse():
    W, H = 900, 330
    b = [text(W / 2, 26, "Shared memory: load a tile once, reuse on-chip",
              GREY, 16, 700)]
    # without
    b.append(text(230, 60, "WITHOUT shared memory", RED, 12.5, 700))
    b.append(box(60, 80, 340, 60, ["every thread re-reads overlapping",
                 "data from slow global memory"], GREY, size=11, lh=16))
    for i in range(4):
        b.append(arrow(120 + i * 70, 150, 120 + i * 70, 190, RED, 1.6))
    b.append(box(60, 192, 340, 44, ["global memory (DRAM) \u2014 hammered"],
                 GREY_D, size=11))
    b.append(text(230, 258, "memory-bound, wasteful", RED, 11, 600))
    # with
    b.append(text(680, 60, "WITH shared memory", TEAL, 12.5, 700))
    b.append(box(500, 80, 340, 44, ["1. cooperatively load a tile"], BLUE,
                 size=11))
    b.append(box(500, 130, 340, 40, ["2. __syncthreads()"], GREY, size=11))
    b.append(box(500, 176, 340, 60, ["3. compute, reusing the tile",
                 "from on-chip shared memory"], TEAL, size=11, lh=16))
    b.append(text(680, 258, "one coalesced load \u00b7 ~20\u00d7 faster reuse",
                  TEAL, 11, 600))
    b.append(line(450, 70, 450, 250, LIGHT, 1, dash="4 5"))
    write("figures/smem-reuse.svg",
          svg(W, H, "".join(b), "Shared-memory reuse"))


def fig_matmul_problem():
    W, H = 860, 300
    b = [text(W / 2, 26, "Matrix multiply: C[i,j] = row i of A \u00b7 col j of B",
              GREY, 16, 700)]
    y, s = 80, 150
    # A
    b.append(obox(90, y, s, s, "", GREY, GREY, rx=8))
    b.append(arrow(104, y + 44, 90 + s - 14, y + 44, BLUE, 2))
    b.append(text(90 + s / 2, y + s + 18, "A  (row i)", GREY, 11, 700))
    b.append(text(90 + s / 2, y - 6, "\u00d7", GREY, 14, 700))
    # B
    b.append(obox(340, y, s, s, "", GREY, GREY, rx=8))
    b.append(arrow(340 + s / 2, y + 14, 340 + s / 2, y + s - 14, TEAL, 2))
    b.append(text(340 + s / 2, y + s + 18, "B  (col j)", GREY, 11, 700))
    # C
    b.append(obox(590, y, s, s, "", GREY, GREY, rx=8))
    b.append(circle(590 + s * 0.62, y + s * 0.42, 8, RED))
    b.append(text(590 + s * 0.62, y + s * 0.42 - 20, "C[i,j]", RED, 11, 700))
    b.append(text(590 + s / 2, y + s + 18, "C  (result)", GREY, 11, 700))
    b.append(text(265, y + s / 2, "\u00d7", GREY, 20, 700))
    b.append(text(515, y + s / 2, "=", GREY, 20, 700))
    b.append(text(W / 2, H - 20,
                  "FLOPs = 2N\u00b3 with O(1) arithmetic intensity \u2014 the game is "
                  "to REUSE each element O(N) times \u2192 compute-bound",
                  LIGHT, 11, 500))
    write("figures/matmul-problem.svg",
          svg(W, H, "".join(b), "Matrix multiply problem"))


def fig_stream_queue():
    W, H = 860, 320
    b = [text(W / 2, 26, "A stream is an in-order queue; streams run "
              "independently", GREY, 15.5, 700)]
    b.append(text(150, 66, "default stream (serialized)", LIGHT, 11, 600,
                  anchor="start"))
    ops = ["op1", "op2", "op3"]
    for i, o in enumerate(ops):
        b.append(box(60 + i * 120, 80, 100, 40, [o], GREY, size=11))
        if i < 2:
            b.append(arrow(160 + i * 120, 100, 180 + i * 120, 100, GREY, 1.7))
    rows = [("stream A", "H2D(a)", "kernel(a)", "D2H(a)", BLUE),
            ("stream B", "H2D(b)", "kernel(b)", "D2H(b)", TEAL),
            ("stream C", "H2D(c)", "kernel(c)", "D2H(c)", BLUE_D)]
    for r, (nm, o1, o2, o3, col) in enumerate(rows):
        y = 160 + r * 46
        b.append(text(120, y + 18, nm, col, 11, 700, anchor="end"))
        for i, o in enumerate([o1, o2, o3]):
            b.append(box(140 + i * 150, y, 138, 36, [o], col, size=10))
            if i < 2:
                b.append(arrow(278 + i * 150, y + 18, 290 + i * 150, y + 18,
                               GREY, 1.4))
    b.append(text(W - 40, 182, "these", LIGHT, 10, 600, anchor="end"))
    b.append(text(W - 40, 198, "overlap", LIGHT, 10, 600, anchor="end"))
    b.append(text(W / 2, H - 16,
                  "copy of one stream can run while another computes \u2014 "
                  "overlap needs pinned memory + async APIs", LIGHT, 11, 500))
    write("figures/stream-queue.svg", svg(W, H, "".join(b), "Stream queues"))


def fig_exec_timeline():
    W, H = 900, 320
    b = [text(W / 2, 26, "Blocks are distributed across SMs over time",
              GREY, 16, 700)]
    b.append(arrow(70, 66, 840, 66, GREY, 1.6))
    b.append(text(845, 66, "time", LIGHT, 10, 600, anchor="start"))
    sms = [("SM 0", ["Block 0", "Block 2", "Block 5"], BLUE),
           ("SM 1", ["Block 1", "Block 3", "Block 6"], TEAL),
           ("SM 2", ["Block 4", "Block 7", "Block 9"], BLUE_D)]
    for r, (nm, blocks, col) in enumerate(sms):
        y = 86 + r * 54
        b.append(text(62, y + 18, nm, GREY, 11, 700, anchor="end"))
        for i, bl in enumerate(blocks):
            b.append(box(80 + i * 240, y, 210, 40, [bl], col, size=11))
    b.append(rrect(80, 254, 760, 48, "#232A35", rx=10, stroke=GREY_D, sw=1.5))
    b.append(text(W / 2, 278,
                  "within a block, warps interleave: while one warp waits on "
                  "memory, the scheduler runs another", "#D7DCE6", 11, 500))
    write("figures/exec-timeline.svg",
          svg(W, H, "".join(b), "Execution timeline"))


def fig_gpu_hierarchy():
    W, H = 720, 480
    b = [text(W / 2, 26, "GPU hardware hierarchy", GREY, 16, 700)]
    rows = [("GPU Die", "+ Gigathread global scheduler", BLUE_D, 0),
            ("GPC \u00b7 Graphics Processing Cluster", "several per die", GREY, 1),
            ("TPC \u00b7 Texture Processing Cluster", "a few per GPC", GREY, 2),
            ("SM \u00b7 Streaming Multiprocessor", "several per TPC", BLUE, 3),
            ("Processing block \u00d74", "CUDA + Tensor + SFU cores", TEAL, 4)]
    for nm, sub, col, d in rows:
        x = 60 + d * 40
        y = 70 + d * 72
        w = W - x - 60
        b.append(box(x, y, w, 54, [nm, sub], col, size=11.5, lh=16))
        if d < 4:
            b.append(line(x + 20, y + 54, x + 20, y + 72, GREY, 1.6))
    b.append(text(W / 2, H - 40,
                  "off to the side of every SM: warp schedulers, register "
                  "file, shared mem / L1, load-store units", LIGHT, 10.5, 500))
    b.append(text(W / 2, H - 20,
                  "shared by all SMs: L2 cache \u00b7 memory controllers \u00b7 "
                  "HBM/GDDR interface", LIGHT, 10.5, 500))
    write("figures/gpu-hierarchy.svg",
          svg(W, H, "".join(b), "GPU hardware hierarchy"))


def fig_latency_ladder():
    W, H = 820, 400
    b = [text(W / 2, 26, "The memory latency ladder (approximate)",
              GREY, 16, 700)]
    rows = [("register", "~1 cycle", 60, TEAL),
            ("shared memory", "~20-30 cycles", 120, BLUE),
            ("L1 cache hit", "~30 cycles", 150, BLUE),
            ("L2 cache hit", "~200 cycles", 260, GREY),
            ("global memory (HBM)", "~400-800 cycles", 380, GREY_D),
            ("host over PCIe", "~microseconds", 460, RED)]
    y = 70
    for nm, lat, w, col in rows:
        tc = WHITE
        b.append(box(60, y, w, 40, [nm], col, tcol=tc, size=11))
        b.append(text(60 + w + 14, y + 20, lat, LIGHT, 11, 600, anchor="start"))
        y += 50
    b.append(text(W / 2, H - 16,
                  "each step down is roughly an order of magnitude \u2014 keep the "
                  "working set as high on the ladder as you can",
                  LIGHT, 11, 500))
    write("figures/latency-ladder.svg",
          svg(W, H, "".join(b), "Memory latency ladder"))


def fig_atomic_decision():
    W, H = 820, 440
    b = [text(W / 2, 26, "Which synchronization primitive?", GREY, 16, 700)]
    b.append(box(300, 56, 220, 44, ["Need to synchronize?"], BLUE_D, size=12))
    nodes = [("within a warp (32)?", "__shfl_sync", TEAL, 120),
             ("within a block (\u22641024)?", "__syncthreads()", TEAL, 184),
             ("across blocks, simple?", "atomicAdd / Max / Min", BLUE, 248),
             ("across blocks, complex?", "multi-launch / coop groups", BLUE,
              312),
             ("really need a lock?", "last resort \u2014 expect slowdown", RED,
              376)]
    b.append(line(410, 100, 410, 398, GREY, 1.6))
    for q, ans, col, y in nodes:
        b.append(line(410, y + 18, 150, y + 18, GREY, 1.4))
        b.append(box(60, y, 240, 40, [q], GREY, size=10.5))
        b.append(arrow(410, y + 18, 470, y + 18, col, 1.6))
        b.append(box(475, y, 300, 40, [ans], col, size=10.5))
    write("figures/atomic-decision.svg",
          svg(W, H, "".join(b), "Sync decision tree"))


def fig_optimization_hierarchy():
    W, H = 780, 400
    b = [text(W / 2, 26, "Atomics optimization: work top-down", GREY, 16, 700)]
    levels = [("Level 1 \u00b7 Algorithm design", "lock-free, parallel, "
               "minimal shared state", BLUE_D, 520),
              ("Level 2 \u00b7 Primitive choice", "warp > block > grid \u00b7 "
               "atomics > locks", BLUE, 440),
              ("Level 3 \u00b7 Contention reduction", "stage in shared mem \u00b7 "
               "privatize copies", TEAL, 360),
              ("Level 4 \u00b7 Low-level tuning", "scope, data types, backoff",
               GREY, 280)]
    y = 74
    for nm, sub, col, w in levels:
        b.append(box((W - w) / 2, y, w, 62, [nm, sub], col, size=11.5, lh=17))
        y += 74
    ax = W - 80
    b.append(arrow(ax, 80, ax, H - 44, LIGHT, 1.4, dash="4 4"))
    b.append(text(ax + 12, 96, "biggest", LIGHT, 10, 500, anchor="start"))
    b.append(text(ax + 12, 110, "wins", LIGHT, 10, 500, anchor="start"))
    b.append(text(ax + 12, H - 56, "smallest", LIGHT, 10, 500, anchor="start"))
    b.append(text(ax + 12, H - 42, "wins", LIGHT, 10, 500, anchor="start"))
    write("figures/optimization-hierarchy.svg",
          svg(W, H, "".join(b), "Optimization hierarchy"))


def fig_host_mem_decision():
    W, H = 800, 430
    b = [text(W / 2, 26, "Choosing host memory", GREY, 16, 700)]
    b.append(box(280, 56, 240, 48, ["Need fast GPU", "transfers?"], BLUE_D,
                 size=11, lh=15))
    b.append(arrow(300, 104, 160, 150, GREY, 1.5))
    b.append(text(210, 122, "no", LIGHT, 10, 600))
    b.append(box(60, 150, 220, 42, ["malloc() \u2014 pageable"], GREY, size=11))
    b.append(arrow(430, 104, 430, 150, GREY, 1.5))
    b.append(text(444, 126, "yes", LIGHT, 10, 600, anchor="start"))
    b.append(box(320, 150, 240, 48, ["GPU accesses host", "memory directly?"],
                 BLUE_D, size=11, lh=15))
    b.append(arrow(430, 198, 430, 244, GREY, 1.5))
    b.append(text(444, 220, "no", LIGHT, 10, 600, anchor="start"))
    b.append(box(320, 244, 240, 48, ["CPU writes only,", "GPU reads?"],
                 BLUE_D, size=11, lh=15))
    b.append(arrow(560, 174, 660, 174, TEAL, 1.5))
    b.append(box(610, 260, 170, 44, ["cudaHostAlloc", "(Mapped) \u2192 zero-copy"],
                 TEAL, size=10, lh=14))
    b.append(arrow(430, 292, 250, 338, GREY, 1.5))
    b.append(text(320, 316, "yes", LIGHT, 10, 600))
    b.append(box(90, 338, 260, 44, ["cudaHostAlloc", "(WriteCombined)"], BLUE,
                 size=10, lh=14))
    b.append(arrow(490, 292, 560, 338, GREY, 1.5))
    b.append(text(540, 316, "no", LIGHT, 10, 600))
    b.append(box(430, 338, 200, 44, ["cudaHostAlloc", "(default pinned)"], BLUE,
                 size=10, lh=14))
    write("figures/host-mem-decision.svg",
          svg(W, H, "".join(b), "Host memory decision"))


def fig_tma():
    W, H = 900, 300
    b = [text(W / 2, 26, "The Tensor Memory Accelerator (TMA)", GREY, 16, 700)]
    b.append(text(230, 60, "WITHOUT TMA", RED, 12.5, 700))
    for i in range(4):
        b.append(box(70 + i * 82, 84, 74, 50, [f"thread {i}", "addr+load"],
                     GREY, size=9, lh=13))
        b.append(arrow(107 + i * 82, 134, 107 + i * 82, 176, GREY, 1.3))
    b.append(box(70, 178, 320, 44, ["each thread computes addresses"], GREY_D,
                 size=10.5))
    b.append(text(230, 250, "address math burns registers & issue slots", RED,
                  10.5, 600))
    b.append(text(680, 60, "WITH TMA", TEAL, 12.5, 700))
    b.append(box(520, 84, 150, 50, ["one thread", "describes a tile"], BLUE,
                 size=10, lh=14))
    b.append(box(720, 84, 150, 50, ["TMA engine", "bulk async copy"], TEAL,
                 size=10, lh=14))
    b.append(arrow(670, 109, 720, 109, GREY, 1.6))
    b.append(box(520, 178, 350, 44, ["global \u2194 shared, hardware-driven"], TEAL,
                 size=10.5))
    b.append(arrow(695, 134, 695, 178, GREY, 1.5))
    b.append(text(680, 250, "one descriptor moves the whole tile", TEAL, 10.5,
                  600))
    b.append(line(455, 54, 455, 262, LIGHT, 1, dash="4 5"))
    write("figures/tma.svg", svg(W, H, "".join(b), "Tensor Memory Accelerator"))


def fig_tile_programming():
    W, H = 860, 300
    b = [text(W / 2, 26, "Thread-centric vs tile-centric programming",
              GREY, 16, 700)]
    b.append(text(220, 62, "THREAD-CENTRIC", GREY, 12.5, 700))
    b.append(box(70, 82, 300, 44, ["you index every thread by hand"], GREY,
                 size=11))
    for i in range(6):
        b.append(circle(110 + i * 42, 158, 11, "#BCD3EE"))
    b.append(text(220, 210, "you manage threadIdx, loops, bounds", LIGHT,
                  10.5, 600))
    b.append(text(650, 62, "TILE-CENTRIC", TEAL, 12.5, 700))
    b.append(box(490, 82, 320, 44, ["you operate on whole tiles"], TEAL,
                 size=11))
    b.append(box(540, 140, 220, 40, ["tile op (e.g. matmul)"], BLUE, size=11))
    b.append(text(650, 210, "the compiler maps tiles onto threads", LIGHT,
                  10.5, 600))
    b.append(line(430, 56, 430, 250, LIGHT, 1, dash="4 5"))
    b.append(text(W / 2, H - 16,
                  "CUDA Tile C++ raises the abstraction \u2014 fewer index bugs, "
                  "hardware-tuned lowering", LIGHT, 11, 500))
    write("figures/tile-programming.svg",
          svg(W, H, "".join(b), "Tile programming"))


ALL = [
    # 00 introduction
    fig_cpu_vs_gpu, fig_thread_hierarchy, fig_memory_hierarchy, fig_roofline,
    # 04 thread indexing
    fig_index_1d, fig_grid_2d, fig_coalesced,
    # 05 memory model
    fig_memory_spaces,
    # 07 shared memory
    fig_reduction,
    # 08 execution model & occupancy
    fig_blocks_to_sm, fig_latency_hiding,
    # 09 work allocation
    fig_occupancy,
    # 10 gpu architecture
    fig_gpu_chip, fig_sm_block, fig_warp_divergence,
    # 11 matrix multiplication
    fig_tiling,
    # 12 atomics & synchronization
    fig_race_condition, fig_warp_shuffle,
    # 13 streams & concurrency
    fig_streams_overlap,
    # ── batch 2 ──
    # 00 introduction
    fig_prog_structure,
    # 02 first kernel
    fig_kernel_flow,
    # 04 thread indexing
    fig_grid_stride, fig_transpose, fig_halo, fig_row_major,
    # 06 memory management
    fig_mem_landscape, fig_pinned, fig_unified_memory,
    # 09 work allocation
    fig_sw_hw_mapping, fig_sm_resources, fig_blocksize_decision,
    # 10 gpu architecture
    fig_tensor_core, fig_smem_banks, fig_core_pipeline, fig_warp_scheduler,
    # 12 atomics & synchronization
    fig_atomic_contention, fig_syncthreads,
    # 14 advanced kernel techniques
    fig_dynamic_parallelism,
    # 15 advanced memory techniques
    fig_async_copy,
    # 16 cuda graphs
    fig_cuda_graph,
    # 17 multi-GPU
    fig_interconnect, fig_parallelism,
    # 18 profiling & debugging
    fig_profiling_stack, fig_test_pyramid, fig_opt_workflow,
    # 19 optimization case studies
    fig_reduction_evolution,
    # 21 modern CUDA
    fig_clusters,
    # ── batch 3: remaining diagrams ──
    # 04 thread indexing
    fig_index_3d, fig_checkerboard,
    # 06 memory management
    fig_host_mem_decision,
    # 07 shared memory
    fig_smem_reuse,
    # 09 work allocation
    fig_exec_timeline,
    # 10 gpu architecture
    fig_gpu_hierarchy, fig_latency_ladder,
    # 11 matrix multiplication
    fig_matmul_problem,
    # 12 atomics & synchronization
    fig_atomic_decision, fig_optimization_hierarchy,
    # 13 streams & concurrency
    fig_stream_queue,
    # 15 advanced memory techniques
    fig_tma,
    # 21 modern CUDA
    fig_tile_programming,
]

def build_font_style(chars):
    """Subset Virgil to the glyphs actually used and return an SVG <style>
    block with the font embedded as a woff2 data URI."""
    if not os.path.exists(FONT_PATH):
        print("WARNING: Virgil.woff2 not found; figures will fall back to a "
              "system handwriting font.")
        return ""
    from fontTools.subset import Options, Subsetter
    from fontTools.ttLib import TTFont
    text = "".join(sorted(chars))
    opts = Options()
    opts.flavor = "woff2"
    opts.desubroutinize = True
    opts.notdef_outline = True
    opts.recalc_bounds = True
    font = TTFont(FONT_PATH)
    ss = Subsetter(options=opts)
    ss.populate(text=text)
    ss.subset(font)
    buf = io.BytesIO()
    font.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    print(f"embedded font: {len(chars)} glyphs, "
          f"{len(buf.getvalue())} bytes woff2")
    return ('<style>@font-face{font-family:"Virgil";font-style:normal;'
            'font-weight:400 700;src:url("data:font/woff2;base64,'
            f'{b64}") format("woff2");}}</style>')


if __name__ == "__main__":
    # Pass 1: build every figure once to discover which glyphs are used.
    for fn in ALL:
        fn()
    # Subset + embed the font, then Pass 2: rewrite the figures with it.
    FONT_STYLE = build_font_style(USED_CHARS)
    for fn in ALL:
        fn()
    print(f"\nDone: {len(ALL)} figures generated.")
