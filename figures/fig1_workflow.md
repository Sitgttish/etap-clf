# Figure 1 — Design Brief & Reproduction Spec

**Deliverable:** a Nature-style schematic of the ferroptosis protein classifier's
development-and-validation workflow, left → right in five stages.

This document is self-contained. An AI agent (or a human) should be able to
**recreate the figure exactly** or **produce a new design of the same content**
from this file alone — without reading the generator source. Where a value matters
scientifically it is stated as fact; where it is an aesthetic choice it is marked
as such so a redesign can override it.

Companion files in this folder:
- `fig1_workflow.py` — reference generator (emits the SVG)
- `fig1_workflow.svg` — editable vector output (text stays as `<text>`)
- `fig1_workflow.pdf` — vector PDF
- `fig1_workflow.png` — 600-dpi raster (4252 × 1484 px)

---

## 1. What the figure must communicate

A single-panel, left-to-right pipeline with **five stages**:

1. **Data input** — protein sequences from three biological classes.
2. **MMseqs2 preprocessing** — sequence clustering / redundancy filtering to
   representative sequences.
3. **Train / validation / test + model training** — the split and the model.
4. **Independent external validation** — held-out genes, frozen model, metrics.
5. **General application** — deploy the finalized model to new (unseen) proteins.

**Two non-negotiable messages** the layout must carry:
- **Development data vs. external-validation data are visually distinct.**
  Stages 1–3 are grouped as "development"; stage 4's dataset is independent and
  never seen in training.
- **The finalized model is deployed for general inference** (stage 5).

Design register: **publication schematic** — clean vector shapes, minimal text,
intuitive icons, consistent arrows, balanced spacing, restrained palette.
Export must be **editable SVG + vector PDF + 600-dpi PNG**.

---

## 2. Scientific content (the facts to depict)

### 2.1 Data classes (shared across the whole figure)
| Class | Role | Meaning |
|---|---|---|
| **Ferroptosis** | positive | proteins in the ferroptosis pathway |
| **Housekeeping** | easy negative | constitutively expressed, unrelated proteins |
| **Cell death** (apoptosis/necroptosis/pyroptosis) | **hard** negative | other death pathways — confusable with ferroptosis |

The three classes reuse the **same color** everywhere they appear (input chips,
cluster representatives, external chips, legend). This color-consistency is what
ties "same categories, different split" together — keep it in any redesign.

### 2.2 Dataset counts (current = "v3" final setup)
- **Development (train/val/test):** 171 ferroptosis · 127 housekeeping · 16 cell-death genes.
- **External validation (held-out):** 23 ferroptosis · 6 cell-death genes. **No housekeeping** in external.
- Split ratio: **Train 64% / Val 16% / Test 20%** (0.20 test, then 0.20 val of the remainder; seed 42).

> These counts are **editable placeholders reflecting the v3 model**. If the final
> dataset changes, update them. Do **not** invent other numbers.

### 2.3 Model architecture (for the stage-3 model glyph)
Per-residue ESM3 embeddings → task transformer → attention pooling → 2-class head:

```
ESM3 per-residue embeddings         (L × 1536)   # esm3_sm_open_v1, bfloat16
  → Linear 1536→256 + LayerNorm
  → + sinusoidal positional encoding
  → 4 × TransformerEncoderLayer  (d=256, heads=8, FFN=512, pre-LN, GELU, dropout=0.1)
  → learned single-query attention pooling  →  (256)
  → LayerNorm + Dropout + Linear 256→2
Total ≈ 2.50M parameters.
```
The **attention-pooling** block is the methodological highlight — give it visual
emphasis (darker fill / heavier stroke) relative to the other model boxes.
Model selection: **best checkpoint by validation AUROC**.

### 2.4 External-validation semantics
- Held-out **genes** (not random sequences) → "never seen in training."
- Model is **frozen** (lock icon) during external eval.
- Reported metrics: **AUROC · Sensitivity · Specificity**.
- The metric bars in the reference figure are **illustrative placeholders**, not
  measured values. Either plug in real numbers once available, or keep schematic.
  **Never render fabricated numeric values as if measured.**

### 2.5 Application semantics
Unseen protein → finalized model → **ferroptosis probability** (0–1) → prioritize
candidate proteins in new proteomes.

---

## 3. Visual design system

### 3.1 Canvas
- Coordinate space (viewBox): **1680 × 586**.
- Physical size: **width 180 mm** (Nature double-column), height ≈ 62.8 mm.
- Aspect ≈ 2.87 : 1 (a wide horizontal strip).
- Background: solid white `#FFFFFF`.
- In the reference SVG all content is wrapped in `translate(0,-52)` purely to
  balance top/bottom whitespace — an implementation detail, not semantic.

### 3.2 Palette (restrained; exact hex)
| Token | Hex | Use |
|---|---|---|
| Ink (titles) | `#1E2A35` | stage titles, bold labels |
| Body | `#45525E` | body text |
| Muted | `#8593A0` | captions, secondary |
| Card border | `#D5DEE6` | dev card outlines |
| Pipeline deep | `#2F5F8A` | badges, model boxes, Train segment |
| Pipeline mid | `#5E8CB8` | Val segment, accents |
| Pipeline light | `#E6EEF6` | model box fills |
| Dev band fill | `#F4F8FB` | development grouping background |
| Dev band border | `#CDDCEA` | development grouping outline |
| **Ferroptosis** | `#C0504D` / light `#F5E4E3` | positive class |
| **Housekeeping** | `#6E8398` / light `#E7ECF1` | easy negative |
| **Cell death** | `#7E63A8` / light `#ECE6F3` | hard negative |
| External border | `#D2963F` (dashed) / fill `#FBF3E7` / deep `#B87A2A` | external-validation card |
| Application border | `#5FA183` / fill `#E9F3EE` / deep `#3E7D60` | application card |
| Arrow | `#6B7783` | connectors |

Palette logic: **one blue family** carries the pipeline; **three category hues**
(red/slate/purple) mark data classes; **amber** = external/held-out; **green** =
deploy/apply. That is the whole palette — do not add more hues in a redesign.

### 3.3 Typography
- Family: `Helvetica Neue, Helvetica, Arial, sans-serif` (Arial is the safe fallback).
- Stage title: **15.5 px bold**, ink. Stage subtitle: 11.5 px muted.
- Chip title: 13.5 px bold. Chip sub: 11 px muted.
- Model-box label: 12 px bold + 10.5 px body sub.
- Captions / notes: 10–11 px muted (italic for "best model…", "never seen…").
- Band label "MODEL DEVELOPMENT": 12.5 px bold, `letter-spacing: 3`, `#7D93A6`.
- Legend: 12 px body.
- Keep text **minimal** — labels are short noun phrases, not sentences.

### 3.4 Shape / mark conventions
- Cards: rounded rect, `rx = 14`.
- Chips / boxes: `rx = 8–10`.
- Strokes: icons 2 px, arrows 2.4 px with a filled arrowhead, card borders 1.4–1.6 px.
- External card border is **dashed** (`5 4`) = "held out / independent."
- 2 px gaps between adjacent fills (e.g. the Train/Val/Test segments each inset 1 px).
- No drop shadows, no gradients (flat, print-safe).

---

## 4. Layout geometry (to recreate exactly)

Five stage cards; x-ranges (left, right) in viewBox units:

| Stage | Card x-range | Accent | Card style |
|---|---|---|---|
| 1 Protein data | 30 – 280 | pipeline blue badge | white + gray border |
| 2 Redundancy filter | 325 – 575 | pipeline blue badge | white + gray border |
| 3 Train/val/test | 620 – 1040 (widest) | pipeline blue badge | white + gray border |
| 4 External validation | 1085 – 1355 | amber badge | amber fill + **dashed** amber border |
| 5 General application | 1400 – 1650 | green badge | green fill + green border |

- Card vertical extent: **y = 152 → 540** (all five aligned).
- Numbered **badge** circle: center `(x_left + 28, 182)`, r = 15, filled with the
  stage accent, white numeral. Title baseline at y = 188, to the right of the badge.
- **Development band** (groups stages 1–3): rounded rect `x=18, y=112, w=1039, h=446`,
  fill `#F4F8FB`, border `#CDDCEA`; label "MODEL DEVELOPMENT" centered at (537, 137).
- **Connector arrows** run horizontally at **y = 346** (card mid-height) in the
  gaps between cards. The 3→4 arrow is labeled "trained model"; the 4→5 arrow "deploy".
- **Legend** row centered near the bottom: three color dots + `Class — role` labels.

### 4.1 Per-stage content
- **① Protein data:** three stacked class chips (color dot + bold name + role +
  a small "beads-on-a-string" protein glyph). Caption: "curated FASTA proteomes".
- **② Redundancy filter:** three faint cluster blobs, each holding grey member dots
  + one colored representative; a collapse-arrow to a single colored representative
  dot on the right. Labels: "clusters" → "representatives". Subtitle "MMseqs2 clustering".
- **③ Train / validate / test:** a segmented bar (Train 64% deep-blue / Val 16% mid /
  Test 20% light) with a per-class count line beneath; then a 4-box model strip
  (ESM3 L×1536 → Transformer ×4 → **Attention pooling** [highlighted] → Classifier
  2-class) joined by arrows; caption "per-residue embeddings → transformer → learned
  attention pooling"; footnote "best model selected on validation AUROC".
- **④ External validation:** two held-out class chips (Ferroptosis 23, Cell death 6);
  italic note "never seen in training"; a white **frozen-model** box with a lock
  glyph; a metrics line "AUROC · Sensitivity · Specificity" over three short bars
  (illustrative); caption "measured on independent data".
- **⑤ General application:** an "unseen protein" glyph (beads + "?" node) → arrow →
  "finalized model" box → arrow → a 0–1 probability gauge with a marker →
  "ferroptosis probability"; closing label "prioritise candidates in new proteomes".

---

## 5. Toolchain (how the outputs are produced)

Renderer: **librsvg** (`rsvg-convert`). Install once:
```bash
brew install librsvg          # macOS; provides rsvg-convert
```
Build:
```bash
python3 fig1_workflow.py                                   # -> fig1_workflow.svg
rsvg-convert -d 600 -p 600 -o fig1_workflow.png fig1_workflow.svg   # 600-dpi PNG
rsvg-convert -f pdf         -o fig1_workflow.pdf fig1_workflow.svg  # vector PDF
```
Notes:
- `-d/-p 600` = 600 dpi; because the SVG declares a physical `width` in mm, the PNG
  comes out at 180 mm × 600 dpi ≈ **4252 px** wide.
- Any SVG renderer works (Inkscape `--export-type=png --export-dpi=600`, or open
  the SVG in Illustrator). librsvg is chosen because it keeps `<text>` crisp and
  produces true-vector PDF.
- The SVG keeps text as text (not outlined), so the `.svg` and `.pdf` remain editable.

---

## 6. What is fixed vs. free (for a redesign)

**Invariants — keep these or the figure loses its meaning:**
- Five stages, left → right, in the stated order.
- Development (1–3) is visually grouped and clearly separated from external
  validation (4). External data reads as "independent / held-out."
- The three data classes use **one consistent color each**, everywhere.
- Attention pooling is visually emphasized as the method's centerpiece.
- Model flows dev → external → application; the deployed model is the finalized one.
- **No fabricated metric numbers.** Bars/values are schematic unless real.
- Minimal text; restrained palette (one pipeline hue + three class hues + two
  status hues).

**Free — a new design may change these:**
- Exact palette hexes (swap for a house/journal palette; keep the *roles*).
- Icon style (line vs. filled; different biological/computational glyphs).
- Orientation of the model sub-schematic, chip styling, badge shape.
- Vertical vs. horizontal grouping of the development band.
- Aspect ratio / physical size (retarget single-column 89 mm if needed).
- Whether metric readouts are bars, a small table, or omitted.

**Accessibility checks for any redesign:**
- Class identity must not be color-alone — every colored mark also carries a text
  label (chips, legend).
- Verify the three class hues remain distinguishable (red/slate/purple are separated
  and always labeled; if you shift them, keep adjacent-pair separation and keep labels).
- Sufficient text contrast on tinted card fills.

---

## 7. One-line change recipes

- **Fix a gene count:** edit the count strings in stage ③ (`171 ferroptosis · 127
  housekeeping · 16 cell-death genes`) and the stage ④ chips (`23`, `6`) — in the
  `.py` or directly in the `.svg` `<text>`.
- **Insert real external metrics:** replace the three illustrative bars in stage ④
  with measured AUROC / Sensitivity / Specificity (add the numeric value as text).
- **Retitle classes / model:** all labels are plain `<text>`; change in place.
- **Recolor to a journal palette:** substitute the hex tokens in §3.2, keeping each
  token's *role*; re-run the build.
