#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import warnings


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"etap_figure5_matplotlib_{os.getpid()}"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


MM = 1 / 25.4
FONT_SIZE = 8
FIGURE_WIDTH_MM = 178
FIGURE_HEIGHT_MM = 134

INK = "#22272B"
TEXT = "#444C52"
MUTED = "#6E7981"
GRID = "#E5EAEE"
CHANCE = "#A6AFB5"
SENESCENCE = "#2F5F8A"
SENSEQNET = "#D9AC32"
SENSEQNET_EDGE = "#A66F00"
PYROPTOSIS = "#2D8584"
PYROPTOSIS_DARK = "#1E6869"
PYROPTOSIS_PALE = "#DCEFED"

METRIC_ORDER = ["AUROC", "Accuracy", "Sensitivity", "Specificity"]
MODEL_ORDER = ["ETAP-CLF", "SenSeqNet"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 5 from externally stored source-data CSVs."
    )
    parser.add_argument("--senescence-metrics", type=Path, required=True)
    parser.add_argument("--senescence-confusion", type=Path, required=True)
    parser.add_argument("--pyroptosis-predictions", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("Figure5"),
        help="Literal output prefix; file extensions are appended (default: Figure5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Pyroptosis decision threshold (default: 0.50).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=2000,
        help="Class-stratified gene-cluster bootstrap resamples (default: 2000).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260815,
        help="Bootstrap random seed (default: 20260815).",
    )
    parser.add_argument(
        "--font",
        default="Arial",
        help="Preferred sans-serif font (default: Arial).",
    )
    parser.add_argument(
        "--strict-font",
        action="store_true",
        help="Fail instead of using a fallback when the requested font is unavailable.",
    )
    parser.add_argument(
        "--verify-predicted-label",
        action="store_true",
        help="Fail if stored predicted_label values differ from --threshold.",
    )
    return parser.parse_args()


def choose_font(preferred: str, strict: bool = False) -> tuple[str, str]:
    if strict:
        try:
            return preferred, findfont(
                FontProperties(family=preferred), fallback_to_default=False
            )
        except ValueError as error:
            raise RuntimeError(
                f"Requested font {preferred!r} is unavailable. Install it or omit "
                "--strict-font to permit a documented fallback."
            ) from error
    candidates = [preferred, "Arial", "Liberation Sans", "DejaVu Sans"]
    for family in dict.fromkeys(candidates):
        try:
            path = findfont(FontProperties(family=family), fallback_to_default=False)
            if family != preferred:
                warnings.warn(
                    f"Font {preferred!r} was unavailable; using {family!r}. "
                    "Install Arial for typography identical to the manuscript.",
                    stacklevel=2,
                )
            return family, path
        except ValueError:
            continue
    raise RuntimeError("No supported sans-serif font was found")


def configure_matplotlib(font_family: str) -> None:
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": [font_family],
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "ETAP-CLF-Figure5",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
        }
    )


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    return path


def load_senescence_metrics(path: Path) -> pd.DataFrame:
    data = pd.read_csv(require_file(path))
    required = {"metric", "model", "value"}
    if not required.issubset(data.columns):
        raise ValueError(
            f"Senescence metrics must contain {sorted(required)}; "
            f"found {list(data.columns)}"
        )
    data = data.loc[:, ["metric", "model", "value"]].copy()
    data["metric"] = data["metric"].astype(str).str.strip()
    data["model"] = data["model"].astype(str).str.strip()
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    selected = data[
        data["metric"].isin(METRIC_ORDER) & data["model"].isin(MODEL_ORDER)
    ].copy()
    if selected.duplicated(["metric", "model"]).any():
        raise ValueError("Duplicate metric-model rows in senescence metrics CSV")
    expected = {(metric, model) for metric in METRIC_ORDER for model in MODEL_ORDER}
    observed = set(zip(selected["metric"], selected["model"]))
    if observed != expected:
        missing = sorted(expected.difference(observed))
        raise ValueError(f"Missing required senescence metric rows: {missing}")
    if not np.isfinite(selected["value"]).all() or not selected["value"].between(0, 1).all():
        raise ValueError("Senescence metrics must be finite values in [0, 1]")
    return (
        selected.pivot(index="metric", columns="model", values="value")
        .loc[METRIC_ORDER, MODEL_ORDER]
        .astype(float)
    )


def load_senescence_confusion(path: Path) -> np.ndarray:
    data = pd.read_csv(require_file(path))
    lower_columns = {str(column).strip().lower(): column for column in data.columns}
    if {"tn", "fp", "fn", "tp"}.issubset(lower_columns):
        if len(data) != 1:
            raise ValueError("The tn/fp/fn/tp senescence format must contain one row")
        tn, fp, fn, tp = [
            data.loc[data.index[0], lower_columns[name]] for name in ("tn", "fp", "fn", "tp")
        ]
        matrix = np.array([[tn, fp], [fn, tp]], dtype=float)
    else:
        required = {"true_class", "predicted_class", "count"}
        if not required.issubset(data.columns):
            raise ValueError(
                "Senescence confusion CSV must contain true_class, predicted_class, count "
                "or one row with tn, fp, fn, tp"
            )
        tidy = data.loc[:, ["true_class", "predicted_class", "count"]].copy()
        tidy["true_class"] = tidy["true_class"].astype(str).str.strip().str.lower()
        tidy["predicted_class"] = tidy["predicted_class"].astype(str).str.strip().str.lower()
        tidy["count"] = pd.to_numeric(tidy["count"], errors="raise")
        if tidy.duplicated(["true_class", "predicted_class"]).any():
            raise ValueError("Duplicate true/predicted class rows in senescence confusion CSV")
        expected = {(true, pred) for true in ("negative", "positive") for pred in ("negative", "positive")}
        observed = set(zip(tidy["true_class"], tidy["predicted_class"]))
        if observed != expected:
            raise ValueError(f"Unexpected senescence confusion cells: {sorted(observed)}")
        indexed = tidy.set_index(["true_class", "predicted_class"])["count"]
        matrix = np.array(
            [
                [indexed.loc[true, pred] for pred in ("negative", "positive")]
                for true in ("negative", "positive")
            ],
            dtype=float,
        )
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Senescence confusion counts must be finite and non-negative")
    if not np.equal(matrix, np.round(matrix)).all():
        raise ValueError("Senescence confusion counts must be integers")
    matrix = matrix.astype(int)
    if (matrix.sum(axis=1) == 0).any():
        raise ValueError("Each true senescence class must contain at least one record")
    return matrix


def load_pyroptosis_predictions(
    path: Path,
    threshold: float,
    verify_stored_predictions: bool,
) -> pd.DataFrame:
    data = pd.read_csv(require_file(path))
    aliases = {}
    if "true_label" not in data.columns and "y_true" in data.columns:
        aliases["y_true"] = "true_label"
    if "prob_positive" not in data.columns and "y_prob" in data.columns:
        aliases["y_prob"] = "prob_positive"
    data = data.rename(columns=aliases)
    required = {"gene", "true_label", "prob_positive"}
    if not required.issubset(data.columns):
        raise ValueError(
            f"Pyroptosis predictions must contain {sorted(required)}; "
            f"found {list(data.columns)}"
        )
    data = data.copy().reset_index(drop=True)
    if data.empty or data[list(required)].isna().any().any():
        raise ValueError("Pyroptosis predictions are empty or contain missing required values")
    data["gene"] = data["gene"].astype(str).str.strip()
    true_label = pd.to_numeric(data["true_label"], errors="raise")
    if not np.equal(true_label, np.round(true_label)).all():
        raise ValueError("Pyroptosis true_label values must be integers 0 or 1")
    data["true_label"] = true_label.astype(int)
    data["prob_positive"] = pd.to_numeric(data["prob_positive"], errors="raise")
    if (data["gene"] == "").any():
        raise ValueError("Blank pyroptosis gene identifiers found")
    if set(data["true_label"].unique()) != {0, 1}:
        raise ValueError("Pyroptosis true_label must contain both 0 and 1")
    if not np.isfinite(data["prob_positive"]).all() or not data["prob_positive"].between(0, 1).all():
        raise ValueError("Pyroptosis probabilities must be finite values in [0, 1]")
    if (data.groupby("gene")["true_label"].nunique() != 1).any():
        raise ValueError("At least one pyroptosis gene occurs with both true labels")
    if "sequence_id" in data.columns:
        sequence_id = data["sequence_id"].astype(str).str.strip()
        if (sequence_id == "").any() or sequence_id.duplicated().any():
            raise ValueError("Optional pyroptosis sequence_id values must be non-blank and unique")
    duplicate_rows = int(data.duplicated().sum())
    if duplicate_rows:
        warnings.warn(
            f"Retaining {duplicate_rows:,} exact duplicate pyroptosis rows. "
            "Provide a unique sequence_id column if duplicates should be distinguishable.",
            stacklevel=2,
        )
    predicted = (data["prob_positive"].to_numpy() >= threshold).astype(int)
    if "predicted_label" in data.columns:
        stored_values = pd.to_numeric(data["predicted_label"], errors="raise")
        if not np.equal(stored_values, np.round(stored_values)).all():
            raise ValueError("Pyroptosis predicted_label values must be integers 0 or 1")
        stored = stored_values.astype(int).to_numpy()
        if not set(np.unique(stored)).issubset({0, 1}):
            raise ValueError("Pyroptosis predicted_label values must be 0 or 1")
        mismatch_count = int(np.count_nonzero(stored != predicted))
        if mismatch_count and verify_stored_predictions:
            raise ValueError(
                f"{mismatch_count:,} stored pyroptosis predictions do not match "
                "the requested threshold"
            )
        if mismatch_count:
            warnings.warn(
                f"Recomputed predicted_label from prob_positive at threshold {threshold:g}; "
                f"{mismatch_count:,} stored labels differed.",
                stacklevel=2,
            )
    data["predicted_label"] = predicted
    data.attrs["exact_duplicate_rows_retained"] = duplicate_rows
    return data


def check_senescence_consistency(metrics: pd.DataFrame, matrix: np.ndarray) -> None:
    tn, fp, fn, tp = matrix.ravel()
    n = matrix.sum()
    calculated = {
        "Accuracy": (tn + tp) / n,
        "Sensitivity": tp / (tp + fn),
        "Specificity": tn / (tn + fp),
    }
    for metric, expected in calculated.items():
        observed = float(metrics.loc[metric, "ETAP-CLF"])
        if not np.isclose(observed, expected, atol=5e-4, rtol=0):
            raise ValueError(
                f"ETAP-CLF {metric} ({observed:.6f}) conflicts with the "
                f"senescence confusion counts ({expected:.6f})"
            )


def gene_cluster_roc_bootstrap(
    data: pd.DataFrame,
    resamples: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    if resamples < 100:
        warnings.warn(
            "Fewer than 100 bootstrap resamples were requested; use at least 2000 for the final figure.",
            stacklevel=2,
        )
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 301)
    tpr_samples = np.empty((resamples, grid.size), dtype=np.float32)
    auc_samples = np.empty(resamples, dtype=float)
    grouped: dict[int, dict[str, np.ndarray]] = {}
    genes_by_class: dict[int, np.ndarray] = {}
    for label in (0, 1):
        genes = np.sort(data.loc[data["true_label"] == label, "gene"].unique())
        if genes.size == 0:
            raise ValueError(f"No genes found for pyroptosis class {label}")
        genes_by_class[label] = genes
        grouped[label] = {
            gene: data.index[
                (data["true_label"] == label) & (data["gene"] == gene)
            ].to_numpy(dtype=int)
            for gene in genes
        }
    y = data["true_label"].to_numpy(dtype=int)
    score = data["prob_positive"].to_numpy(dtype=float)
    for replicate in range(resamples):
        chunks: list[np.ndarray] = []
        for label in (0, 1):
            genes = genes_by_class[label]
            sampled = rng.choice(genes, size=genes.size, replace=True)
            chunks.extend(grouped[label][gene] for gene in sampled)
        index = np.concatenate(chunks)
        y_rep = y[index]
        score_rep = score[index]
        auc_samples[replicate] = roc_auc_score(y_rep, score_rep)
        fpr_rep, tpr_rep, _ = roc_curve(y_rep, score_rep)
        if np.any(np.diff(fpr_rep) < 0):
            raise RuntimeError("ROC false-positive-rate coordinates are not monotonic")
        tpr_samples[replicate] = np.interp(grid, fpr_rep, tpr_rep)
        tpr_samples[replicate, 0] = 0.0
        tpr_samples[replicate, -1] = 1.0
    return {
        "fpr": grid,
        "tpr_low": np.quantile(tpr_samples, 0.025, axis=0),
        "tpr_high": np.quantile(tpr_samples, 0.975, axis=0),
        "auc_low": float(np.quantile(auc_samples, 0.025)),
        "auc_high": float(np.quantile(auc_samples, 0.975)),
    }


def add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(-24, 14),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=FONT_SIZE,
        fontweight="bold",
        color=INK,
        annotation_clip=False,
    )


def draw_benchmark(ax: mpl.axes.Axes, metrics: pd.DataFrame) -> None:
    y = np.arange(len(metrics))
    y_etap = y - 0.075
    y_senseq = y + 0.075
    ax.scatter(
        metrics["ETAP-CLF"],
        y_etap,
        s=28,
        marker="D",
        color=SENESCENCE,
        edgecolor="white",
        linewidth=0.6,
        label="ETAP-CLF",
        zorder=3,
    )
    ax.scatter(
        metrics["SenSeqNet"],
        y_senseq,
        s=38,
        marker="o",
        color=SENSEQNET,
        edgecolor=SENSEQNET_EDGE,
        linewidth=0.75,
        label="SenSeqNet (reported)",
        zorder=3,
    )
    for index, (_, row) in enumerate(metrics.iterrows()):
        ax.annotate(
            f"{row['ETAP-CLF']:.3f}",
            (row["ETAP-CLF"], y_etap[index]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=SENESCENCE,
            fontweight="bold",
        )
        ax.annotate(
            f"{row['SenSeqNet']:.3f}",
            (row["SenSeqNet"], y_senseq[index]),
            xytext=(0, -7),
            textcoords="offset points",
            ha="center",
            va="top",
            color=SENSEQNET_EDGE,
            fontweight="bold",
        )
    minimum = min(0.70, float(metrics.to_numpy().min()) - 0.03)
    minimum = max(0.0, np.floor(minimum * 20) / 20)
    ax.set_xlim(minimum, 1.00)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]))
    ax.set_ylim(len(metrics) - 0.15, -0.80)
    ax.set_yticks(y, metrics.index)
    ax.set_xlabel("Score")
    ax.set_title("Senescence benchmark comparison", loc="left", fontweight="bold", pad=8)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.00, 0.99),
        ncol=2,
        handletextpad=0.35,
        columnspacing=1.2,
        borderaxespad=0,
    )
    add_panel_label(ax, "A")


def relative_luminance(hex_color: str) -> float:
    rgb = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in rgb
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(color_a: str, color_b: str) -> float:
    high, low = sorted(
        (relative_luminance(color_a), relative_luminance(color_b)), reverse=True
    )
    return (high + 0.05) / (low + 0.05)


def draw_confusion(
    ax: mpl.axes.Axes,
    matrix: np.ndarray,
    color: str,
    task: str,
    panel: str,
    footer: str,
) -> None:
    fractions = matrix / matrix.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list(f"{task}_confusion", ["#F4F7F7", color])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    for row in range(2):
        for column in range(2):
            cell_color = to_hex(cmap(fractions[row, column]))
            ax.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=cell_color,
                    edgecolor="white",
                    linewidth=1.4,
                )
            )
            white_contrast = contrast_ratio(cell_color, "#FFFFFF")
            ink_contrast = contrast_ratio(cell_color, INK)
            text_color = "white" if white_contrast >= ink_contrast else INK
            ax.text(
                column,
                row - 0.10,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
            )
            ax.text(
                column,
                row + 0.18,
                f"{100 * fractions[row, column]:.1f}%",
                ha="center",
                va="center",
                color=text_color,
            )
    ax.set_xticks([0, 1], labels=["Negative", "Positive"])
    ax.set_yticks([0, 1], labels=["Negative", "Positive"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{task}\nConfusion matrix", loc="left", fontweight="bold", pad=8)
    ax.tick_params(length=0, colors=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.50,
        -0.24,
        footer,
        transform=ax.transAxes,
        ha="center",
        va="top",
        color=MUTED,
    )
    add_panel_label(ax, panel)


def draw_roc(
    ax: mpl.axes.Axes,
    y: np.ndarray,
    score: np.ndarray,
    bootstrap: dict[str, np.ndarray | float],
) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    auroc = float(roc_auc_score(y, score))
    ax.fill_between(
        bootstrap["fpr"],
        bootstrap["tpr_low"],
        bootstrap["tpr_high"],
        color=PYROPTOSIS_PALE,
        linewidth=0,
        zorder=1,
    )
    ax.plot([0, 1], [0, 1], color=CHANCE, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    ax.plot(fpr, tpr, color=PYROPTOSIS_DARK, linewidth=1.5, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("W")
    ticks = [0, 0.25, 0.50, 0.75, 1.00]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Pyroptosis\nROC curve", loc="left", fontweight="bold", pad=8)
    ax.text(
        0.96,
        0.08,
        f"AUROC = {auroc:.3f}\n"
        f"95% CI {float(bootstrap['auc_low']):.3f}–{float(bootstrap['auc_high']):.3f}\n"
        f"n = {len(y):,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=INK,
    )
    add_panel_label(ax, "C")
    return auroc


def require_matplotlib_panel_alignment(
    fig: mpl.figure.Figure,
    axes: dict[str, mpl.axes.Axes],
    json_out: Path,
    tolerance_pt: float = 1.5,
) -> None:
    """Write an auditable layout manifest and check comparable B/D panels."""
    fig.canvas.draw()
    figure_width_pt = fig.get_figwidth() * 72
    figure_height_pt = fig.get_figheight() * 72
    rectangles = {}
    panels = []
    for panel, axis in axes.items():
        box = axis.get_position()
        rectangles[panel] = {
            "left_pt": box.x0 * figure_width_pt,
            "bottom_pt": box.y0 * figure_height_pt,
            "width_pt": box.width * figure_width_pt,
            "height_pt": box.height * figure_height_pt,
        }
        panels.append(
            {
                "id": panel,
                "bbox_pt": [
                    box.x0 * figure_width_pt,
                    box.y0 * figure_height_pt,
                    box.x1 * figure_width_pt,
                    box.y1 * figure_height_pt,
                ],
            }
        )
    for key in ("width_pt", "height_pt"):
        difference = abs(rectangles["B"][key] - rectangles["D"][key])
        if difference > tolerance_pt:
            raise RuntimeError(
                f"Comparable confusion-matrix panels B and D differ in {key} "
                f"by {difference:.2f} pt"
            )
    payload = {
        "schema_version": 1,
        "backend": "python-matplotlib",
        "figure": {
            "width_pt": figure_width_pt,
            "height_pt": figure_height_pt,
        },
        "panels": panels,
        "column_groups": [
            {"id": "confusion-matrices", "panels": ["B", "D"]}
        ],
        "exemptions": [],
    }
    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def enforce_typography(fig: mpl.figure.Figure, font_family: str) -> None:
    for text_item in fig.findobj(mpl.text.Text):
        text_item.set_fontfamily(font_family)
        text_item.set_fontsize(FONT_SIZE)
    sizes = {float(item.get_fontsize()) for item in fig.findobj(mpl.text.Text)}
    if sizes != {float(FONT_SIZE)}:
        raise RuntimeError(f"Unexpected rendered font sizes: {sorted(sizes)}")


def output_path(prefix: Path, extension: str) -> Path:
    """Append an extension without stripping dotted version names from the prefix."""
    return Path(f"{prefix}{extension}")


def save_outputs(fig: mpl.figure.Figure, prefix: Path) -> None:
    prefix = prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Creator": "figure5.py",
        "Title": "Figure 5",
        "Subject": "ETAP-CLF portability to senescence and pyroptosis",
    }
    pdf_path = output_path(prefix, ".pdf")
    svg_path = output_path(prefix, ".svg")
    png_path = output_path(prefix, ".png")
    tiff_path = output_path(prefix, ".tiff")
    fig.savefig(pdf_path, metadata=metadata)
    fig.savefig(svg_path, metadata={"Creator": "figure5.py"})
    fig.savefig(png_path, dpi=600, metadata={"Software": "figure5.py"})
    with Image.open(png_path) as rendered:
        rendered.convert("RGB").save(
            tiff_path,
            compression="tiff_lzw",
            dpi=(600, 600),
        )


def format_threshold(value: float) -> str:
    if np.isclose(value, round(value, 2), atol=1e-12, rtol=0):
        return f"{value:.2f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")

    font_family, font_path = choose_font(args.font, strict=args.strict_font)
    configure_matplotlib(font_family)

    senescence_metrics = load_senescence_metrics(args.senescence_metrics)
    senescence_matrix = load_senescence_confusion(args.senescence_confusion)
    check_senescence_consistency(senescence_metrics, senescence_matrix)
    pyroptosis = load_pyroptosis_predictions(
        args.pyroptosis_predictions,
        args.threshold,
        verify_stored_predictions=args.verify_predicted_label,
    )
    y_pyro = pyroptosis["true_label"].to_numpy(dtype=int)
    score_pyro = pyroptosis["prob_positive"].to_numpy(dtype=float)
    pyro_matrix = confusion_matrix(
        y_pyro,
        pyroptosis["predicted_label"].to_numpy(dtype=int),
        labels=[0, 1],
    )
    bootstrap = gene_cluster_roc_bootstrap(
        pyroptosis,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )

    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM * MM, FIGURE_HEIGHT_MM * MM),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.25, 0.88],
        height_ratios=[1, 1],
        left=0.09,
        right=0.97,
        bottom=0.14,
        top=0.93,
        wspace=0.38,
        hspace=0.62,
    )
    axes = {
        "A": fig.add_subplot(grid[0, 0]),
        "B": fig.add_subplot(grid[0, 1]),
        "C": fig.add_subplot(grid[1, 0]),
        "D": fig.add_subplot(grid[1, 1]),
    }

    draw_benchmark(axes["A"], senescence_metrics)
    draw_confusion(
        axes["B"],
        senescence_matrix,
        SENESCENCE,
        "Senescence",
        "B",
        f"n = {senescence_matrix.sum():,} · row-normalized",
    )
    auroc = draw_roc(axes["C"], y_pyro, score_pyro, bootstrap)
    draw_confusion(
        axes["D"],
        pyro_matrix,
        PYROPTOSIS,
        "Pyroptosis",
        "D",
        f"Threshold = {format_threshold(args.threshold)} · n = {pyro_matrix.sum():,}",
    )

    enforce_typography(fig, font_family)
    resolved_prefix = args.output_prefix.expanduser().resolve()
    alignment_path = output_path(resolved_prefix, "_alignment-layout.json")
    alignment_path.parent.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(fig, axes, alignment_path)
    save_outputs(fig, args.output_prefix)
    plt.close(fig)

    tn_sen, fp_sen, fn_sen, tp_sen = senescence_matrix.ravel()
    tn_pyro, fp_pyro, fn_pyro, tp_pyro = pyro_matrix.ravel()
    print(f"Font: {font_family} ({font_path})")
    print(
        "Senescence confusion: "
        f"TN={tn_sen:,}, FP={fp_sen:,}, FN={fn_sen:,}, TP={tp_sen:,}"
    )
    print(
        f"Pyroptosis AUROC={auroc:.6f}; "
        f"95% gene-bootstrap CI [{float(bootstrap['auc_low']):.6f}, "
        f"{float(bootstrap['auc_high']):.6f}]"
    )
    print(
        "Pyroptosis confusion: "
        f"TN={tn_pyro:,}, FP={fp_pyro:,}, FN={fn_pyro:,}, TP={tp_pyro:,}"
    )
    print(
        "Pyroptosis records: "
        f"n={len(pyroptosis):,}, genes={pyroptosis['gene'].nunique():,}, "
        "exact duplicate rows retained="
        f"{pyroptosis.attrs['exact_duplicate_rows_retained']:,}"
    )
    print(f"Outputs: {args.output_prefix.expanduser().resolve()}.[pdf|svg|png|tiff]")


if __name__ == "__main__":
    main()
