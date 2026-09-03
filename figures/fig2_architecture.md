# Figure 2 — Design Brief & Reproduction Spec

**Deliverable:** a Nature-style, two-panel model-architecture figure.
- **Panel a** — ESM3 used as a *frozen* per-residue feature extractor: sequence → `L × 1536`.
- **Panel b** — the task classifier (transformer + attention pooling), with the
  **attention pooling shown as its true mechanism**, intuitively and accurately.

Self-contained: an agent should be able to **recreate this exactly** or **redesign
the same content** from this file alone. Facts (architecture, shapes, math) are
authoritative; aesthetic choices are marked so a redesign may override them.
Shares the design language of `fig1_workflow.md` — the two figures must read as a set.

Companion files: `fig2_architecture.py` (generator), `.svg` (editable vector),
`.pdf` (vector), `.png` (600 dpi, 4252 × 2936 px).

---

## 1. What the figure must communicate

- ESM3 is a **pretrained protein language model used frozen** (a feature extractor,
  not fine-tuned). It turns a sequence into a **per-residue** embedding matrix.
- The task model is small and sits **on top of** those embeddings.
- The methodological centerpiece is the **attention pooling**: it collapses the
  variable-length residue set into **one fixed-size vector** by a **learned,
  weighted average over positions** — and the weights are **interpretable**.
- Tensor shapes must be correct and visible at each step.

Register: publication schematic — clean vector, minimal text, consistent arrows,
restrained palette, tensor shapes annotated on arrows. Export **SVG + PDF + 600-dpi PNG**.

---

## 2. Scientific content (the facts to depict)

### 2.1 Panel a — ESM3 embedding (must be accurate)
- Model: **ESM3 small open** (`esm3_sm_open_v1`), run in **bfloat16**, **frozen**.
- Tokenization adds special tokens: `BOS + L residues + EOS` → `L+2` tokens.
- Forward pass → last hidden state per token; take `out.embeddings`.
- **Trim BOS/EOS** → keep the `L` real residues → **`L × 1536`** per-residue matrix.
- These embeddings are cached to HDF5 and are the **only** input to panel b.
- Flow: `sequence (L residues) → [ESM3, frozen] → L × 1536`.

### 2.2 Panel b — task classifier (must match `eta_package/etap/model.py`)
```
Input: ESM3 embeddings                       (L × 1536)
  → Linear 1536→256 + LayerNorm
  → + sinusoidal positional encoding
  → 4 × TransformerEncoderLayer  (d=256, heads=8, FFN=512, pre-LN, GELU, dropout=0.1)
                                               (L × 256, contextualised residues)
  → Attention pooling  (learned single query) →  z (256)
  → LayerNorm + Dropout + Linear 256→2
  → softmax                                    →  P(ferroptosis), P(other)
≈ 2.50M trainable parameters.  ESM3 params are NOT counted (frozen).
```

### 2.3 Attention pooling — the exact mechanism (do not hand-wave)
One **learned query vector** `q ∈ ℝ²⁵⁶` (a single shared parameter, same for every
protein). For contextualised residues `h₁ … h_L` (each `∈ ℝ²⁵⁶`):
```
score      sᵢ = ( q · hᵢ ) / √d           # dot product, d = 256; one scalar per residue
weights    aᵢ = softmax(s) over positions # a₁…a_L, and  Σ aᵢ = 1
pooled      z = Σ aᵢ hᵢ                    # weighted average of the residue vectors → 256-dim
```
Key points the visual must preserve:
- The softmax is **over positions (L)**, *not* over the 256 features.
- Weights **sum to 1** and are typically **sparse/peaked** — a few residues dominate.
- The output `z` is **one fixed-size 256-dim vector regardless of L** (this is why
  pooling exists) and the weights `aᵢ` are the **interpretable** readout of "which
  residues the model relied on."
- Mean pooling would be the special case `aᵢ = 1/L`; attention *learns* the weights.

> The specific weight values shown (0.03…0.28 with two peaks) are **illustrative** —
> chosen to convey focus/sparsity. They are **not measured attention**. A redesign
> may change them but must keep `Σ aᵢ = 1` and a peaked profile, and must not label
> them as real data.

---

## 3. Visual design system (shared with Figure 1)

### 3.1 Canvas
- viewBox **1680 × 1160**; physical **width 180 mm** (double-column), height ≈ 124 mm.
- 600-dpi PNG ≈ **4252 × 2936 px**. Background white.
- Two stacked panels, each in a light rounded container; bold lowercase panel letters.

### 3.2 Palette (exact hex)
| Token | Hex | Use |
|---|---|---|
| Ink / Body / Muted | `#1E2A35` / `#45525E` / `#8593A0` | text tiers |
| Pipeline deep / mid / light | `#2F5F8A` / `#5E8CB8` / `#E6EEF6` | ESM3 box, model boxes, tiles |
| Panel fill / border | `#FAFBFC` / `#E4E9EE` | panel containers |
| Card border | `#D5DEE6` | model boxes |
| **Attention accent** deep / border / fill | `#B87A2A` / `#D2963F` / `#FDF7EF` | pooling inset, query, weight bars |
| Residue-vector ramp | `#DCE7F2 · #C3D6E9 · #A9C3DD · #8BAAD0 · #6B8FC0 · #4E76AC` | the 6 cells of a 256-dim vector glyph |
| Embedding-matrix gradient | `#EAF1F8 → #B7CDE5` | the `L × 1536` block |
| Ferroptosis / Other output | `#C0504D` on `#F5E4E3` / `#8593A0` on `#EEF1F4` | 2-class chips |
| Arrow | `#6B7783` | connectors |

Color roles: **blue** = the model/pipeline; **amber** = the attention mechanism
(query, weights, pooled `z`) so the reader's eye locks onto the centerpiece.
Amber here is the same hue Figure 1 uses for "external validation" — that is fine
because figures are read independently; within Figure 2 amber means "attention."

### 3.3 Typography
- Family `Helvetica Neue, Helvetica, Arial, sans-serif`.
- Panel letter 24 px bold; panel title 16 px bold; box titles 12.5–14 px bold;
  sub-lines 10.5–11.5 px; captions 10–11 px muted (italic for intuition notes).
- **Math uses real subscripts** via `<tspan dy=4 font-size=70%>` (see `formula()` in
  the generator). Do not render `h_L` with a literal underscore.

### 3.4 Shapes / marks
- Rounded rects (`rx` 6–14); arrows 2.4 px with a filled arrowhead; tensor-shape
  labels sit beside the arrows (`L × 1536`, `L × 256`, `z (256)`).
- The **frozen** ESM3 carries a small **lock** glyph.
- The attention-pooling box is **amber-highlighted** and expanded via a **dashed
  zoom callout** into the large inset.
- No shadows, no rainbow; flat and print-safe.

---

## 4. Layout geometry (to recreate exactly)

- Panel **a** container: `(18, 70, 1644, 300)`; label "a" at `(34,104)`; title at `(70,104)`.
- Panel **b** container: `(18, 400, 1644, 720)`; label "b" at `(34,434)`; title at `(70,434)`.

### 4.1 Panel a (left → right, centered on y ≈ 211)
1. **Sequence tiles** — ~10 single-letter AA tiles (30 px, blue) + "…", from x≈78, y≈196; caption "amino-acid sequence (L residues)".
2. **ESM3 box** `(520,150,270,122)` — light-blue, lock icon, "ESM3", "pretrained protein LM · frozen", "esm3_sm_open_v1", faint internal layer lines.
3. **Embedding matrix** `(862,146,300,130)` — blue gradient fill, ~9 white row lines (residues) + faint vertical lines (features); "1536 features" above; "L residues" rotated on the left; caption "per-residue embeddings (L × 1536)" + "special tokens (BOS/EOS) removed".
4. Arrow to the right labeled **"input to (b)"**.

### 4.2 Panel b — compact architecture stack (left column, center x = 300, width 300)
Top → bottom, arrows carrying shape labels:
`ESM3 embeddings (L×1536)` → `Linear 1536→256 · LayerNorm + sinusoidal PE` →
`Transformer encoder ×4 (8 heads · FFN 512 · pre-LN · GELU)` [drawn as 4 stacked
layers] → **`Attention pooling`** [amber highlight] → `LayerNorm · Linear 256→2 ·
softmax` → output chips **Ferroptosis** / **Other** + "P(ferroptosis)".

### 4.3 Panel b — attention-pooling zoom inset
- Inset box `(560, 470, 1082, 548)`, amber fill/border; two **dashed** callout lines
  from the compact "Attention pooling" box's right edge to the inset corners.
- Title "Attention pooling" + subtitle "learned single-query weighting — length-independent & interpretable".
- **Formula line** (top): `sᵢ = (q·hᵢ)/√d    aᵢ = softmax(s) over positions ,  Σ aᵢ = 1`.
- **Learned query q** — amber vector glyph on the left; arrow "compare to every residue".
- **Weight profile** — a bar chart of `aᵢ` above the residues; baseline line; values
  labeled; bars with `aᵢ ≥ 0.15` drawn in solid amber (the rest lighter).
- **Residue vectors** — a row of ~10 vector-glyph columns `h₁ … h_L` (6-cell blue
  ramp); the high-weight ones outlined in amber.
- **Weighted sum** — lines from each residue to a `Σ` node with **stroke width ∝ weight**,
  then `Σ → z` where `z` is the pooled amber vector glyph; label `z = Σ aᵢ hᵢ` (256-dim).
- **Intuition caption** (bottom): "high weight → residues the model relies on; one
  fixed-size vector regardless of length L".

Illustrative weight array used: `[0.03, 0.05, 0.04, 0.20, 0.28, 0.12, 0.06, 0.09,
0.05, 0.08]` (Σ = 1.00).

---

## 5. Toolchain
```bash
brew install librsvg     # once; provides rsvg-convert
python3 fig2_architecture.py                                    # -> .svg
rsvg-convert -d 600 -p 600 -o fig2_architecture.png fig2_architecture.svg
rsvg-convert -f pdf         -o fig2_architecture.pdf fig2_architecture.svg
```
Text stays as `<text>` (incl. tspan subscripts), so `.svg`/`.pdf` remain editable.

---

## 6. Fixed vs. free (for a redesign)

**Invariants — keep or the figure becomes wrong/misleading:**
- Two panels: (a) ESM3 **frozen** extractor → `L × 1536`; (b) small task classifier on top.
- ESM3 reads as **pretrained + frozen** (lock), producing **per-residue** embeddings.
- Correct shapes throughout: `L × 1536 → L × 256 → z(256) → 2`.
- Attention pooling is the **accurate** mechanism: single learned query, `sᵢ = q·hᵢ/√d`,
  **softmax over positions**, `Σ aᵢ = 1`, `z = Σ aᵢ hᵢ`. The math must be right.
- Softmax is over **positions**, never over the 256 features (a common miscommunication).
- Weights are **illustrative** and **peaked**; never labeled as measured attention.
- Model hyperparameters accurate (proj 256, 4 layers, 8 heads, FFN 512, pre-LN, GELU,
  dropout 0.1, sinusoidal PE, ≈2.5M params).
- Attention pooling visually emphasized (highlight + zoom).

**Free — a redesign may change:**
- Palette hexes (keep the *roles*: blue = model, one accent = attention).
- Panel arrangement (stacked vs side-by-side), icon style, vector-glyph rendering.
- Number of residue columns shown; the specific illustrative weight values.
- Whether to add a self-attention/head detail; whether to show the output as chips,
  a gauge, or a probability bar.
- Aspect ratio / single-column retarget.

**Accessibility:** amber (attention) vs blue (model) must stay distinguishable and
each carries text; formulas legible at print size; identity never color-alone.

---

## 7. One-line change recipes
- **Change a hyperparameter** (e.g. layers/heads): edit the stack text in the `sbox`
  calls (`Transformer encoder × 4`, `8 heads · FFN 512 …`).
- **Change the illustrative weights:** edit the `w = [...]` list (must sum to 1); bar
  heights and the `Σ`-line thickness update automatically.
- **Insert real attention (optional):** replace `w` with measured pooling weights for
  one example protein and relabel the caption as data (not illustrative).
- **Recolor to a journal palette:** substitute the hex tokens in §3.2, keeping roles.
