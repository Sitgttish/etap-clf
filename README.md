# ETAP-CLF

Code for **“ETAP-CLF: an ESM3-based transformer attention framework for binary protein
classification.”**

Preprint: [bioRxiv 10.64898/2026.08.16.745129](https://doi.org/10.64898/2026.08.16.745129)

ETAP-CLF keeps a pretrained ESM3 backbone (`esm3_sm_open_v1`) frozen, projects its
per-residue embeddings into a 256-dimensional latent space, contextualizes them with four
pre-layer-normalized transformer encoder layers, and aggregates them with a learned
single-query attention-pooling head before a binary classifier — 2,503,682 trainable
parameters. The same architecture and hyperparameters are trained separately for
ferroptosis, senescence, and pyroptosis.

Ferroptosis is the primary benchmark and is evaluated twice: on an internal random
sequence split, and on a **gene-disjoint external cohort** whose genes are entirely absent
from model development.

## Layout

```
etap/            the ETAP-CLF package — model, data, training, evaluation, CLI
ferroptosis/     the ferroptosis model notebook and cohort description
figures/         figure generators and published vector panels
pyroptosis/      pyroptosis split assignment and training history
paper/           the manuscript
*.ipynb          task notebooks (see the map below)
```

## Where each result comes from

| Paper | Produced by |
|---|---|
| **Figure 1A** — architecture | `figures/fig2_architecture.py` |
| **Figure 1B–D** — workflow | `figures/fig1_workflow.py` |
| **Figure 2A–E** — pooling UMAPs | `figures/fig_umap3d_five.py`, vectors from `ESM3_FerroCLF_pooling_ablation.ipynb` |
| **Figure 4A–D** — senescence & pyroptosis | `figures/fig4_portability.py` |
| **Figure 5A–F** — recovery & pathway enrichment | `figures/fig5_pathways.py` |
| **Table 1** — dataset composition | `ferroptosis/ETAP_Ferroptosis.ipynb`, `ETAP_Senescence.ipynb`, `ETAP_Pyroptosis.ipynb` |
| **Table 2** — ferroptosis performance | `ferroptosis/ETAP_Ferroptosis.ipynb`, `Baselines_Published_Predictors.ipynb`, `ETAP_FeroConCap_comparison.ipynb` |
| **Table 3** — senescence & pyroptosis | `ETAP_Senescence.ipynb`, `ETAP_Pyroptosis.ipynb` |

Figure filenames in `figures/` predate the manuscript's final numbering; this table is
authoritative.

## Reproducing

1. `pip install -r requirements.txt`, and install **MMseqs2** for redundancy reduction.
2. Set an `HF_TOKEN` Colab secret (ESM3 is gated). No token is hard-coded.
3. Run `ferroptosis/ETAP_Ferroptosis.ipynb` for the primary result, then the baseline and
   portability notebooks.
4. Generate figures from `figures/`. The Figure 4 and Figure 5 scripts are plotting-only and
   take source-data CSVs as command-line arguments.

Sequence data, ESM3 embeddings, and model weights are not stored here — see
**Data availability** in the manuscript.
