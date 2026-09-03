#!/usr/bin/env python3
"""
Figure 1 — development & validation workflow. Nature idiom (cf. AlphaFold Nature 2021,
DeepFRI Nat Commun 2021): solid gray/blue rounded module blocks with white labels,
one bold accent-blue flow arrow, colored heatmap-strip data icons, restrained palette,
no numbered badges. Run: python3 fig1_workflow.py -> fig1_workflow.svg
"""
import hashlib
from pathlib import Path

W, H = 1720, 384
FONT = "Helvetica, Arial, sans-serif"
INK, BODY, MUTE = "#1b1b1b", "#4a4a4a", "#808080"
BLUE  = "#2f6ca8"; GRAY = "#8a929c"; GRAYL = "#aeb6bf"; PANEL_B = "#d3dae1"
POS, HK, CD = "#C0504D", "#6E8398", "#7E63A8"
HEAT = ["#eaf0f6", "#cfe0ee", "#a9c8e0", "#7ba9cf", "#4c85b8"]
TICK = [POS, HK, CD, "#c9a24b", "#5a9e78", MUTE]

svg = []
def e(s): svg.append(s)
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def txt(x, y, s, size=11, fill=BODY, weight="normal", anchor="start", ital=False, ls=None):
    st = ' font-style="italic"' if ital else ""
    lsp = f' letter-spacing="{ls}"' if ls else ""
    e(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
      f'fill="{fill}" text-anchor="{anchor}"{st}{lsp}>{esc(s)}</text>')
def rrect(x, y, w, h, r, fill, stroke=None, sw=1.1):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    e(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" fill="{fill}"{st}/>')
def block(cx, cy, w, h, label, sub=None, fill=GRAY):
    rrect(cx - w/2, cy - h/2, w, h, 5, fill)
    txt(cx, cy + (-3 if sub else 4), label, 12.5, "#ffffff", "bold", "middle")
    if sub: txt(cx, cy + 13, sub, 9.5, "#e8eef4", anchor="middle")
def circ(cx, cy, r, fill="none", stroke=INK, sw=1.1):
    st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
    e(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"{st}/>')
def ln(x1, y1, x2, y2, color=BLUE, w=1.2):
    e(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>')
def flow(x1, x2, y, w=2.6, hs=9):
    ln(x1, y, x2 - 3, y, BLUE, w)
    e(f'<path d="M{x2},{y} L{x2-hs},{y-hs/2} L{x2-hs},{y+hs/2} Z" fill="{BLUE}"/>')
def heat(x, y, cols, rows, cw=7, seed=0):
    for r in range(rows):
        for c in range(cols):
            hh = int(hashlib.md5(f"{seed}-{r}-{c}".encode()).hexdigest(), 16)
            e(f'<rect x="{x+c*cw}" y="{y+r*cw}" width="{cw-0.6}" height="{cw-0.6}" fill="{HEAT[hh%len(HEAT)]}"/>')
def ticks(x, y, n=9, off=0):
    for t in range(n):
        e(f'<rect x="{x+t*7}" y="{y}" width="5" height="14" rx="1" fill="{TICK[(t+off)%6]}" opacity="0.82"/>')

mm = 180.0
e(f'<svg xmlns="http://www.w3.org/2000/svg" width="{mm}mm" height="{mm*H/W:.2f}mm" '
  f'viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="#ffffff"/>')

ICY = 168
def caption(cx, title, cap):
    txt(cx, 300, title, 14, INK, "bold", "middle")
    txt(cx, 321, cap, 10.5, MUTE, anchor="middle")

# 1 · protein sequences
c1 = 160
rrect(c1 - 96, ICY - 62, 192, 120, 6, "#fbfcfd", PANEL_B, 1)
for i, (name, col) in enumerate([("Ferroptosis", POS), ("Housekeeping", HK), ("Cell death", CD)]):
    y = ICY - 40 + i * 33
    circ(c1 - 80, y + 7, 4.5, fill=col, stroke="none")
    ticks(c1 - 66, y, 9, i); txt(c1 + 26, y + 11, name, 10, BODY)
caption(c1, "Protein sequences", "curated, three classes")

# 2 · redundancy filtering
c2 = 430
for dx, dy in [(-46, -18), (-60, 2), (-40, 14), (-30, -6), (-52, -8), (-36, 24)]:
    circ(c2 + dx, ICY + dy, 6, "none", GRAYL, 1.3)
flow(c2 - 6, c2 + 34, ICY)
circ(c2 + 52, ICY - 9, 6.5, fill=GRAY, stroke="none"); circ(c2 + 52, ICY + 12, 6.5, fill=GRAY, stroke="none")
caption(c2, "Redundancy filtering", "MMseqs2, 30% identity")

# 3 · model training (clean block chain)
c3 = 855
bx, bw, byy = c3 - 150, 300, ICY - 74
xa = bx
for nm, fr, col in [("Train", .64, BLUE), ("Val", .16, "#7ba1c6"), ("Test", .20, GRAYL)]:
    w = bw * fr; rrect(xa + 0.6, byy, w - 1.2, 12, 2, col); txt(xa + w/2, byy - 5, nm, 9, MUTE, anchor="middle"); xa += w
xe, xt, xa2 = 701, 853, 966                # ESM3 / Transformer / Attention centres
block(xe, ICY, 66, 44, "ESM3", "frozen", GRAY)
heat(756, ICY - 17, 4, 5, cw=7, seed=3)    # 756–784  (embedding)
block(xt, ICY, 90, 44, "Transformer", "×4", GRAY)    # 808–898
block(xa2, ICY, 92, 44, "Attention", "pooling", BLUE)  # 920–1012
flow(734, 754, ICY, 2)                     # ESM3 -> embedding
flow(786, 806, ICY, 2)                     # embedding -> transformer
flow(898, 918, ICY)                        # transformer -> attention
flow(1012, 1032, ICY, 2); circ(1038, ICY, 4, fill=INK, stroke="none")
txt(c3, ICY + 40, "L × 1536   →   L × 256   →   256", 9, MUTE, "normal", "middle", ital=True)
caption(c3, "Model training", "ESM3 + attention pooling")

# 4 · external validation
c4 = 1300
circ(c4 - 84, ICY - 12, 4.5, fill=POS, stroke="none"); circ(c4 - 84, ICY + 10, 4.5, fill=CD, stroke="none")
txt(c4 - 72, ICY - 8, "held-out", 9.5, BODY); txt(c4 - 72, ICY + 14, "genes", 9.5, BODY)
flow(c4 - 12, c4 + 14, ICY)
block(c4 + 46, ICY, 56, 44, "", None, GRAY)                 # frozen model = lock only
rrect(c4 + 39, ICY - 5, 14, 11, 2, "none", "#ffffff", 1.3)
e(f'<path d="M{c4+42},{ICY-5} v-4 a4,4 0 0 1 8,0 v4" fill="none" stroke="#ffffff" stroke-width="1.3"/>')
caption(c4, "External validation", "held-out genes")

# 5 · inference
c5 = 1580
ticks(c5 - 60, ICY - 30, 9, 0); txt(c5 + 8, ICY - 19, "?", 12, INK, "bold")
ln(c5, ICY - 12, c5, ICY - 1, BLUE, 2.2)
e(f'<path d="M{c5},{ICY+5} L{c5-4},{ICY-2} L{c5+4},{ICY-2} Z" fill="{BLUE}"/>')
pbx, pbw, pby = c5 - 55, 110, ICY + 14
rrect(pbx, pby, pbw, 9, 4.5, "#eef2f6", PANEL_B, 1); rrect(pbx, pby, pbw * 0.8, 9, 4.5, BLUE)
ln(pbx + pbw * 0.8, pby - 4, pbx + pbw * 0.8, pby + 13, INK, 1.2)
txt(c5, pby + 26, "P(ferroptosis)", 9.5, MUTE, anchor="middle")
caption(c5, "Inference", "new proteins")

# main-flow arrows + development bracket
flow(c1 + 100, c2 - 66, ICY); flow(c2 + 62, xe - 36, ICY)
flow(1046, c4 - 92, ICY); flow(c4 + 78, c5 - 66, ICY)
bx0, bx1, by = c1 - 100, 1030, ICY - 118
ln(bx0, by, bx1, by, MUTE, 1); ln(bx0, by, bx0, by + 9, MUTE, 1); ln(bx1, by, bx1, by + 9, MUTE, 1)
txt((bx0 + bx1) / 2, by - 8, "MODEL DEVELOPMENT", 10.5, MUTE, "bold", "middle", ls="2")

e('</svg>')
Path(__file__).with_name("fig1_workflow.svg").write_text("\n".join(svg))
print(f"wrote fig1_workflow.svg  ({len(svg)} elements)")
