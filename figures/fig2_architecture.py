#!/usr/bin/env python3
"""
Figure 2 — Model architecture.
  Panel a: ESM3 as a frozen per-residue feature extractor (sequence -> L x 1536).
  Panel b: task transformer + attention pooling, shown accurately & intuitively:
           learned query . residues -> softmax over positions -> weighted sum.

Same design language / palette as fig1_workflow.py so the figures read as a set.

Run:  python3 fig2_architecture.py   ->   fig2_architecture.svg
Then rasterize with rsvg-convert (see build block at bottom of this repo's md).
"""

from pathlib import Path

W, H = 1680, 1160
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

# palette (shared with fig1)
INK, BODY, MUTE = "#1E2A35", "#45525E", "#8593A0"
CARD_B = "#D5DEE6"
BLUE_D, BLUE_M, BLUE_L = "#2F5F8A", "#5E8CB8", "#E6EEF6"
PANEL_F, PANEL_B = "#FAFBFC", "#E4E9EE"
AMB_B, AMB_F, AMB_D = "#D2963F", "#FDF7EF", "#B87A2A"
ARROW = "#6B7783"
VEC = ["#DCE7F2", "#C3D6E9", "#A9C3DD", "#8BAAD0", "#6B8FC0", "#4E76AC"]

svg = []
def e(s): svg.append(s)
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def text(x, y, s, size=13, fill=BODY, weight="normal", anchor="start",
         ls=None, style=""):
    extra = f' letter-spacing="{ls}"' if ls else ""
    e(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
      f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{extra} '
      f'style="{style}">{esc(s)}</text>')

def formula(x, y, parts, size=13.5, fill=INK, weight="normal", anchor="start"):
    """parts: list of (txt, 'n'|'s'|'p') normal/subscript/superscript."""
    ss, cur, spans = 4, 0, []
    for txt, kind in parts:
        tgt = ss if kind == 's' else (-ss if kind == 'p' else 0)
        dy = tgt - cur; cur = tgt
        fs = '70%' if kind in ('s', 'p') else '100%'
        spans.append(f'<tspan dy="{dy}" font-size="{fs}">{esc(txt)}</tspan>')
    e(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
      f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
      + "".join(spans) + '</text>')

def rrect(x, y, w, h, rx, fill, stroke=None, sw=1.2, dash=None, op=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    o = f' opacity="{op}"' if op is not None else ""
    e(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
      f'fill="{fill}"{st}{d}{o}/>')

def circle(cx, cy, r, fill, stroke=None, sw=1.2):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    e(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"{st}/>')

def line(x1, y1, x2, y2, color, sw=2, dash=None, op=1.0, cap="round"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    e(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
      f'stroke-width="{sw}" stroke-linecap="{cap}" opacity="{op}"{d}/>')

def arrow(x1, y1, x2, y2, color=ARROW, sw=2.4, op=1.0):
    e(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
      f'stroke-width="{sw}" marker-end="url(#ah)" stroke-linecap="round" '
      f'opacity="{op}"/>')

def lock(x, y, col):
    rrect(x - 8, y, 16, 12, 2, col)
    e(f'<path d="M{x-5},{y} v-4 a5,5 0 0 1 10,0 v4" fill="none" '
      f'stroke="{col}" stroke-width="2"/>')

def vec_col(cx, top, w, h, highlight=False):
    """A residue feature-vector glyph: a thin column split into VEC-ramp cells."""
    n = len(VEC); ch = h / n
    for i, c in enumerate(VEC):
        rrect(cx - w / 2, top + i * ch, w, ch, 0, c)
    stroke = AMB_D if highlight else "#9DB2C7"
    sw = 2.2 if highlight else 1
    rrect(cx - w / 2, top, w, h, 3, "none", stroke, sw)

# document
mm_w = 180.0
e(f'<svg xmlns="http://www.w3.org/2000/svg" width="{mm_w}mm" '
  f'height="{mm_w*H/W:.2f}mm" viewBox="0 0 {W} {H}">')
e('<defs>')
e(f'<marker id="ah" markerWidth="9" markerHeight="9" refX="7" refY="4.5" '
  f'orient="auto" markerUnits="userSpaceOnUse">'
  f'<path d="M0,0 L9,4.5 L0,9 L2.4,4.5 Z" fill="{ARROW}"/></marker>')
e('<linearGradient id="mg" x1="0" y1="0" x2="1" y2="0">'
  '<stop offset="0" stop-color="#EAF1F8"/><stop offset="1" stop-color="#B7CDE5"/>'
  '</linearGradient>')
e('</defs>')
rrect(0, 0, W, H, 0, "#FFFFFF")

# panel containers
rrect(18, 70, 1644, 300, 16, PANEL_F, PANEL_B, 1.4)     # a
rrect(18, 400, 1644, 720, 16, PANEL_F, PANEL_B, 1.4)    # b
text(34, 104, "a", 24, INK, "bold")
text(34, 434, "b", 24, INK, "bold")

#
# PANEL a — ESM3 embedding
#
text(70, 104, "ESM3 per-residue embedding", 16, INK, "bold")

# sequence tiles
seq = list("MKTAYIAKQR")
tx, ty, tw, tg = 78, 196, 30, 6
for i, aa in enumerate(seq):
    x = tx + i * (tw + tg)
    rrect(x, ty, tw, 30, 5, BLUE_L, BLUE_M, 1)
    text(x + tw / 2, ty + 20, aa, 13, BLUE_D, "bold", "middle")
dots_x = tx + len(seq) * (tw + tg)
text(dots_x + 8, ty + 22, "…", 18, MUTE, "bold")
text(tx + 150, 258, "amino-acid sequence  (L residues)", 12, BODY, anchor="middle")

# arrow -> ESM3
arrow(dots_x + 30, 211, 520, 211)

# ESM3 box (frozen pLM)
rrect(520, 150, 270, 122, 12, BLUE_L, BLUE_D, 1.6)
for i in range(4):                                       # faint 'deep model' lines
    line(548, 246 + i * 6, 762, 246 + i * 6, "#9DB6D0", 2, op=0.5)
lock(560, 172, BLUE_D)
text(600, 190, "ESM3", 22, INK, "bold")
text(655, 214, "pretrained protein LM · frozen", 11.5, BODY, anchor="middle")
text(655, 233, "esm3_sm_open_v1", 10.5, MUTE, anchor="middle")

# arrow -> matrix
arrow(790, 211, 862, 211)

# embedding matrix L x 1536
mx, my, mw, mh = 862, 146, 300, 130
rrect(mx, my, mw, mh, 6, "url(#mg)", "#8FA9C4", 1.4)
for r in range(1, 10):                                   # residue rows
    line(mx, my + r * mh / 10, mx + mw, my + r * mh / 10, "#FFFFFF", 1, op=0.5)
for c in range(1, 12):                                   # feature columns (faint)
    line(mx + c * mw / 12, my, mx + c * mw / 12, my + mh, "#FFFFFF", 0.8, op=0.25)
text(mx + mw / 2, my - 12, "1536 features", 11.5, BODY, anchor="middle")
e(f'<text x="{mx-14}" y="{my+mh/2}" font-family="{FONT}" font-size="11.5" '
  f'fill="{BODY}" text-anchor="middle" transform="rotate(-90 {mx-14} {my+mh/2})">'
  f'L residues</text>')
text(mx + mw / 2, 300, "per-residue embeddings   (L × 1536)", 12.5, INK, "bold",
     "middle")
text(mx + mw / 2, 322, "special tokens (BOS / EOS) removed", 10.5, MUTE, "middle")

# tie to panel b
arrow(mx + mw + 8, 211, mx + mw + 70, 211)
text(mx + mw + 78, 208, "input", 11.5, BODY)
text(mx + mw + 78, 224, "to (b)", 11.5, BODY)

#
# PANEL b — task transformer + attention pooling
#
text(70, 434, "Task classifier", 16, INK, "bold")

# compact architecture stack (left)
sx, sw_ = 150, 300                # stack x, width
cxs = sx + sw_ / 2
def sbox(y, h, fill, border, sw, lines, highlight=False):
    rrect(sx, y, sw_, h, 10, fill, border, sw)
    n = len(lines)
    for i, (t, sz, w, col) in enumerate(lines):
        yy = y + h / 2 + (i - (n - 1) / 2) * (sz + 4) + sz * 0.35
        text(cxs, yy, t, sz, col, w, "middle")

def vdim(y, label):
    text(cxs + sw_ / 2 + 16, y, label, 11, MUTE)

# B1 input
sbox(452, 60, "#FFFFFF", CARD_B, 1.3,
     [("ESM3 embeddings", 13.5, "bold", INK), ("L × 1536   (from a)", 11, "normal", MUTE)])
arrow(cxs, 512, cxs, 540)
# B2 proj + PE
sbox(540, 78, "#FFFFFF", CARD_B, 1.3,
     [("Linear 1536 → 256  ·  LayerNorm", 12.5, "bold", INK),
      ("+ sinusoidal positional encoding", 11.5, "normal", BODY)])
arrow(cxs, 618, cxs, 648); vdim(636, "L × 256")
# B3 transformer x4 (stacked)
for i in range(3, -1, -1):
    rrect(sx + 10 - i * 3, 662 - i * 4, sw_ - 20, 96, 9,
          BLUE_L if i == 0 else "#EEF4FA", BLUE_D, 1.3)
text(cxs, 700, "Transformer encoder × 4", 13.5, INK, "bold", "middle")
text(cxs, 722, "8 heads · FFN 512 · pre-LN · GELU", 11.5, BODY, anchor="middle")
text(cxs, 740, "self-attention contextualises residues", 10.5, MUTE, "middle",
     style="font-style:italic")
arrow(cxs, 758, cxs, 792); vdim(780, "L × 256")
# B4 attention pooling (highlighted; zoom source)
sbox(792, 62, AMB_F, AMB_B, 2, [("Attention pooling", 14, "bold", INK)], True)
arrow(cxs, 854, cxs, 884); vdim(874, "z (256)")
# B5 head
sbox(884, 62, "#FFFFFF", CARD_B, 1.3,
     [("LayerNorm · Linear 256 → 2", 12.5, "bold", INK),
      ("softmax", 11, "normal", BODY)])
arrow(cxs, 946, cxs, 976)
# B6 output
oy = 976
rrect(sx + 30, oy, 110, 54, 9, "#F5E4E3", "#C0504D", 1.4)
text(sx + 85, oy + 22, "Ferroptosis", 12, INK, "bold", "middle")
text(sx + 85, oy + 40, "positive", 10, MUTE, anchor="middle")
rrect(sx + 160, oy, 110, 54, 9, "#EEF1F4", "#8593A0", 1.4)
text(sx + 215, oy + 22, "Other", 12, INK, "bold", "middle")
text(sx + 215, oy + 40, "negative", 10, MUTE, anchor="middle")
text(cxs, oy + 74, "P(ferroptosis)", 11.5, BODY, anchor="middle")

# zoom callout to the attention-pooling inset
IX, IY, IW, IH = 560, 470, 1082, 548
line(sx + sw_, 794, IX, IY, "#B9C6D2", 1.4, dash="5 5")
line(sx + sw_, 852, IX, IY + IH, "#B9C6D2", 1.4, dash="5 5")
rrect(IX, IY, IW, IH, 14, AMB_F, AMB_B, 1.8)
text(IX + 26, IY + 34, "Attention pooling", 17, INK, "bold")
text(IX + 26, IY + 55, "learned single-query weighting — length-independent & interpretable",
     12, BODY)

# formula (top of inset)
formula(IX + 26, IY + 96,
        [("score  s", 'n'), ("i", 's'), (" = ( q · h", 'n'), ("i", 's'),
         (" ) / √d          a", 'n'), ("i", 's'),
         (" = softmax(s) over positions  ,     Σ a", 'n'),
         ("i", 's'), (" = 1", 'n')],
        14, INK)

# geometry inside inset
base = IY + 300                      # weight-bar baseline
NP = 10
xs = [IX + 250 + i * 66 for i in range(NP)]
w = [0.03, 0.05, 0.04, 0.20, 0.28, 0.12, 0.06, 0.09, 0.05, 0.08]
scale = 92 / max(w)
colw = 34
col_top, col_h = base + 20, 118

# learned query (left)
qx = IX + 70
vec_top = col_top
for i, c in enumerate([AMB_F, "#F2D9B4", "#E9C58C", "#DCAF6A", "#CF9A4B", "#B87A2A"]):
    rrect(qx - 17, vec_top + i * col_h / 6, 34, col_h / 6, 0, c)
rrect(qx - 17, vec_top, 34, col_h, 3, "none", AMB_D, 2)
text(qx, vec_top + col_h + 20, "learned", 11.5, AMB_D, "bold", "middle")
text(qx, vec_top + col_h + 36, "query  q", 11.5, AMB_D, "bold", "middle")
text(qx, vec_top + col_h + 52, "shared · 256-dim", 10, MUTE, anchor="middle")
arrow(qx + 24, col_top + 40, xs[0] - 26, col_top + 40, AMB_D, 2)
text((qx + xs[0]) / 2 + 6, col_top + 30, "compare to every residue", 10.5,
     AMB_D, anchor="middle")

# attention weight bars (above residues)
text(xs[0] - 40, base - 96, "attention weights", 12, AMB_D, "bold")
text(xs[0] - 40, base - 80, "aᵢ  (Σ = 1)", 11, MUTE)
line(xs[0] - 26, base, xs[-1] + 26, base, "#C9B48F", 1.2)
for i, x in enumerate(xs):
    bh = w[i] * scale
    rrect(x - colw / 2, base - bh, colw, bh, 3, AMB_B if w[i] >= 0.15 else "#E4C58F")
    text(x, base - bh - 6, f"{w[i]:.2f}", 9, MUTE, anchor="middle")

# residue vectors (below the bars)
for i, x in enumerate(xs):
    vec_col(x, col_top, colw, col_h, highlight=w[i] >= 0.15)
for i, sb in [(0, "1"), (1, "2"), (NP - 1, "L")]:
    formula(xs[i], col_top + col_h + 20, [("h", 'n'), (sb, 's')], 12, BODY,
            anchor="middle")
text((xs[0] + xs[-1]) / 2, col_top + col_h + 38, "contextualised residues  (each 256-dim)",
     11, MUTE, anchor="middle")

# weighted sum -> pooled z
sigx, sigy = IX + 250 + NP * 66 + 40, col_top + col_h / 2
for i, x in enumerate(xs):
    op = 0.25 + w[i] * 2.4
    line(x, col_top + col_h, sigx - 20, sigy, AMB_B, 0.8 + w[i] * 9,
         op=min(op, 1.0))
circle(sigx, sigy, 22, "#FFFFFF", AMB_D, 2)
text(sigx, sigy + 7, "Σ", 22, AMB_D, "bold", "middle")
text(sigx, sigy + 44, "weighted", 10.5, MUTE, anchor="middle")
text(sigx, sigy + 58, "sum", 10.5, MUTE, anchor="middle")
zx = sigx + 70
arrow(sigx + 24, sigy, zx - 2, sigy, AMB_D, 2.4)
for i, c in enumerate(VEC):
    rrect(zx, sigy - col_h / 2 + i * col_h / 6, 34, col_h / 6, 0, c)
rrect(zx, sigy - col_h / 2, 34, col_h, 3, "none", AMB_D, 2.4)
formula(zx + 17, sigy + col_h / 2 + 22,
        [("z = Σ a", 'n'), ("i", 's'), (" h", 'n'), ("i", 's')], 13.5, INK,
        "bold", "middle")
text(zx + 17, sigy + col_h / 2 + 40, "pooled · 256-dim", 10.5, MUTE, anchor="middle")

# intuition caption
text(IX + IW / 2, IY + IH - 20,
     "high weight → residues the model relies on;  one fixed-size vector regardless of length L",
     11.5, AMB_D, anchor="middle", style="font-style:italic")

e('</svg>')
out = Path(__file__).with_name("fig2_architecture.svg")
out.write_text("\n".join(svg))
print(f"wrote {out}  ({len(svg)} elements)")
