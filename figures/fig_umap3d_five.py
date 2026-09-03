#!/usr/bin/env python3
"""Five SEPARATE 3D UMAPs, each a vector PDF (Arial 8pt). 4000/class subsample (identical
indices to fig_pooling_umap_3d.py for the trained three -> same layout/shape), but rendered
OPAQUE (alpha=1.0, s=6), camera elev18/azim-60.
  1 raw mean   2 raw max            (raw ESM3, 1536-D — fig3_umap_vectors.npz, 2500/class avail)
  3 trained mean  4 trained max  5 trained attention  (256-D — pooling_umap_vectors.npz, 4000/class)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa

HERE = os.path.dirname(os.path.abspath(__file__)); JR = os.path.dirname(HERE)

print("Arial resolves to:", os.path.basename(fm.findfont("Arial", fallback_to_default=True)))
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType -> selectable vector text
    "font.family": "Arial", "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8,
})

fig3 = np.load(os.path.join(HERE, "fig3_umap_vectors.npz"), allow_pickle=True)     # raw
pool = np.load(os.path.join(JR, "pooling_umap_vectors.npz"), allow_pickle=True)    # trained

POS, NEG = "#C0504D", "#6E8398"
INK, MUTE = "#1E2A35", "#8593A0"

def subsample(y, k=4000, seed=42):
    """Identical to fig_pooling_umap_3d.py (K=4000, shuffle) -> trained panels reproduce exactly."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    K = min(k, len(pos), len(neg))
    sel = np.concatenate([rng.choice(pos, K, replace=False), rng.choice(neg, K, replace=False)])
    rng.shuffle(sel)
    return sel

def umap3(X):
    import umap
    return umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric="cosine",
                     random_state=42).fit_transform(X.astype("float32"))

def make(X, y, title, fname, auroc=None):
    sel = subsample(y); ys = y[sel]; Z = umap3(X[sel])       # 4000/class -> same layout as 3-panel
    fig = plt.figure(figsize=(4.0, 3.9))
    ax = fig.add_subplot(111, projection="3d")
    order = np.random.default_rng(0).permutation(len(ys))    # fair draw order for opaque points
    cols = np.where(ys == 1, POS, NEG)
    ax.scatter(Z[order, 0], Z[order, 1], Z[order, 2], s=6, c=cols[order], alpha=1.0,   # opaque (0% transp.)
               linewidths=0, depthshade=True)
    ttl = title + (f"\n" + r"$\it{test\ AUROC}$" + f" {auroc:.3f}" if auroc is not None else "")
    ax.set_title(ttl, fontweight="bold", color=INK, pad=2)
    ax.set_xlabel("UMAP-1", color=MUTE, labelpad=-9)
    ax.set_ylabel("UMAP-2", color=MUTE, labelpad=-9)
    ax.set_zlabel("UMAP-3", color=MUTE, labelpad=-9)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(length=0)
    ax.view_init(elev=18, azim=-60); ax.grid(True, alpha=0.25)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_edgecolor(MUTE); a.pane.set_alpha(0.04)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", ls="", mfc=POS, mec="none", ms=5, label="ferroptosis"),
        Line2D([0], [0], marker="o", ls="", mfc=NEG, mec="none", ms=5, label="non-ferroptosis")],
        loc="upper left", frameon=False, handletextpad=0.2, borderpad=0.1)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, fname)); plt.close(fig)
    print("saved", fname, f"(n={len(sel)})", flush=True)

yr = fig3["y"].astype(int); yt = pool["y"].astype(int)
make(fig3["mean"],      yr, "Raw ESM3 mean-pool",     "umap3d_1_raw_mean.pdf")
make(fig3["max"],       yr, "Raw ESM3 max-pool",      "umap3d_2_raw_max.pdf")
make(pool["mean"],      yt, "Trained mean-pool",      "umap3d_3_trained_mean.pdf",      auroc=0.9805)
make(pool["max"],       yt, "Trained max-pool",       "umap3d_4_trained_max.pdf",       auroc=0.9823)
make(pool["attention"], yt, "Trained attention-pool", "umap3d_5_trained_attention.pdf", auroc=0.9794)
print("done — 5 vector PDFs (4000/class trained, opaque alpha 1.0)")
