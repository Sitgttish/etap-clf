# Ferroptosis — the model reported in the manuscript

`ETAP_Ferroptosis.ipynb` trains ETAP-CLF on the ferroptosis development cohort and evaluates it
on both the internal test split and the gene-disjoint external cohort. This is the run behind
Table 2 and Figure 3A–C.

## Cohorts

**Development — 116,845 sequences.** ESM3-compatible cluster representatives (MMseqs2 at 0.30
identity; sequences under 10 aa, unsupported residues, and proteins over 2,048 residues
removed): 102,611 ferroptosis-associated and housekeeping sequences, plus 14,234 sequences
from 16 genes associated with other regulated cell-death processes. Split by stratified
random sampling into **74,780 training / 18,696 validation / 23,369 internal test**.

Because the split is at the sequence level, sequences from the same gene appear in more than
one partition. That is deliberate — it is the conventional setting used by existing
ferroptosis predictors, and the external cohort is what measures generalization.

**External — gene-disjoint.** Every sequence from the selected external genes is excluded
from training, validation, model selection, and hyperparameter development. Housekeeping
proteins are not included, so specificity is measured against proteins from biologically
related cell-death pathways rather than unrelated ones.

`external_labeled.fasta` holds the raw external set — **40,871 positive** sequences from
**25 ferroptosis genes** and **14,879 hard-negative** sequences from **6 held-out cell-death
genes** (BCL2, CASP3, CASP4, CASP5, RIPK1, RIPK3): 55,750 sequences over 31 genes. Headers are
`>{gene}|{idx}|label={0/1}`. The notebook applies a 1,500-sequence per-gene cap, giving the
**28,079** sequences (23,621 positive / 4,458 hard-negative) reported in the manuscript.

> **Note.** The manuscript Methods states 23 external ferroptosis genes; this FASTA contains
> 25. Table 1 (194 positive genes) and the Figure 5 caption ("25 of 169") are both consistent
> with 25, so the "23" appears to be an error in that sentence.

## Reported performance

| Split | AUROC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| Internal test (23,369) | 0.980 | 0.931 | 0.946 | 0.893 |
| External (28,079) | 0.801 | 0.765 | 0.791 | 0.629 |

Specificity on the external cohort is measured against the held-out cell-death genes, which
is a stricter test than housekeeping proteins would give.

## Running it

1. Put the ferroptosis data and the ESM3 embedding cache on Drive under `MyDrive/JR_Ferro/`.
2. Runtime → **GPU**; add an `HF_TOKEN` Colab secret (ESM3 is gated).
3. Run all. Outputs go to `MyDrive/JR_Ferro/v4/`.
