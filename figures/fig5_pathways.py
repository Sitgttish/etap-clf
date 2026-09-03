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
    str(Path(tempfile.gettempdir()) / f"etap_figure6_matplotlib_{os.getpid()}"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, NullLocator, ScalarFormatter
import numpy as np
import pandas as pd
from PIL import Image


MM = 1 / 25.4
FONT_SIZE = 8
COMPOSITE_WIDTH_MM = 183
COMPOSITE_HEIGHT_MM = 140
RECOVERY_PANEL_WIDTH_MM = 80
RECOVERY_PANEL_HEIGHT_MM = 70
PATHWAY_PANEL_WIDTH_MM = 92
PATHWAY_PANEL_HEIGHT_MM = 100

INK = "#252525"
MUTED = "#6F777D"
NEUTRAL = "#D9DEE1"
NEUTRAL_EDGE = "#FFFFFF"
GRID = "#ECEFF1"
PALE = "#F3F5F6"

TASK_ORDER = ("ferroptosis", "senescence", "pyroptosis")
TASKS = {
    "ferroptosis": {
        "title": "Ferroptosis",
        "color": "#B84646",
        "recovery_panel": "A",
        "pathway_panel": "D",
    },
    "senescence": {
        "title": "Senescence",
        "color": "#2F6597",
        "recovery_panel": "B",
        "pathway_panel": "E",
    },
    "pyroptosis": {
        "title": "Pyroptosis",
        "color": "#278B8B",
        "recovery_panel": "C",
        "pathway_panel": "F",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the six-panel Figure 6 from external CSV files."
    )
    parser.add_argument("--ferro-recovery", type=Path, required=True)
    parser.add_argument("--senescence-recovery", type=Path, required=True)
    parser.add_argument("--pyroptosis-recovery", type=Path, required=True)
    parser.add_argument("--ferro-pathways", type=Path, required=True)
    parser.add_argument("--senescence-pathways", type=Path, required=True)
    parser.add_argument("--pyroptosis-pathways", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("Figure6_pathways"),
        help="Literal output prefix; extensions are appended.",
    )
    parser.add_argument(
        "--ferro-min-n",
        type=int,
        default=30,
        help="Minimum ferroptosis sequences per gene (default: 30).",
    )
    parser.add_argument(
        "--senescence-min-n",
        type=int,
        default=30,
        help="Minimum senescence sequences per gene (default: 30).",
    )
    parser.add_argument(
        "--pyroptosis-min-n",
        type=int,
        default=10,
        help="Minimum pyroptosis sequences per gene (default: 10).",
    )
    parser.add_argument(
        "--min-recovery",
        type=float,
        default=0.95,
        help="Minimum per-gene recovery rate for all tasks (default: 0.95).",
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
        "--no-separate-panels",
        dest="separate_panels",
        action="store_false",
        help="Generate only the six-panel composite.",
    )
    parser.set_defaults(separate_panels=True)
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
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "ETAP-CLF-Figure6",
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


def choose_single_column(data: pd.DataFrame, names: tuple[str, ...], role: str) -> str:
    present = [name for name in names if name in data.columns]
    if not present:
        raise ValueError(f"Missing {role}; accepted column names are {list(names)}")
    if len(present) > 1:
        reference = pd.to_numeric(data[present[0]], errors="raise").to_numpy(float)
        for candidate in present[1:]:
            values = pd.to_numeric(data[candidate], errors="raise").to_numpy(float)
            if not np.allclose(reference, values, rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError(
                    f"Conflicting columns for {role}: {', '.join(present)}"
                )
    return present[0]


def parse_binary_label(values: pd.Series, role: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or not np.equal(numeric, np.round(numeric)).all():
        raise ValueError(f"{role} must contain integer values 0 or 1")
    numeric = numeric.astype(int)
    if not set(numeric.unique()).issubset({0, 1}):
        raise ValueError(f"{role} must contain only 0 and 1")
    return numeric


def parse_boolean(values: pd.Series, role: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    if not normalized.isin(mapping).all():
        raise ValueError(f"{role} must contain boolean/0/1 values")
    return normalized.map(mapping).astype(bool)


def validate_constant_metadata(
    data: pd.DataFrame,
    column: str,
    expected: float,
    role: str,
) -> None:
    if column not in data.columns:
        return
    values = pd.to_numeric(data[column], errors="raise")
    if values.isna().any() or values.nunique(dropna=False) != 1:
        raise ValueError(f"{column} must contain one constant value per input file")
    observed = float(values.iloc[0])
    if not np.isclose(observed, expected, rtol=0, atol=1e-12):
        raise ValueError(
            f"{role} metadata ({observed:g}) conflicts with the requested value "
            f"({expected:g})"
        )


def load_recovery(
    path: Path,
    task: str,
    minimum_n: int,
    minimum_recovery: float,
) -> pd.DataFrame:
    raw = pd.read_csv(require_file(path))
    if "gene" not in raw.columns:
        raise ValueError(f"{task} recovery CSV must contain a gene column")
    if raw.empty:
        raise ValueError(f"{task} recovery CSV is empty")
    if raw["gene"].isna().any():
        raise ValueError(f"{task} recovery CSV contains missing gene identifiers")
    if "task" in raw.columns:
        observed_tasks = set(raw["task"].astype(str).str.strip().str.lower())
        if observed_tasks != {task}:
            raise ValueError(
                f"{task} recovery file contains unexpected task values: {observed_tasks}"
            )

    before_filter = len(raw)
    if "label" in raw.columns:
        labels = parse_binary_label(raw["label"], f"{task} label")
        raw = raw.loc[labels.eq(1)].copy()
    else:
        raw = raw.copy()
    excluded = before_filter - len(raw)
    if raw.empty:
        raise ValueError(f"No positive {task} genes remain after label filtering")

    count_column = choose_single_column(raw, ("n_test", "n"), "sequence count")
    rate_column = choose_single_column(
        raw,
        ("recovery_rate", "accuracy", "acc"),
        "recovery rate",
    )
    validate_constant_metadata(
        raw,
        "minimum_test_sequences",
        minimum_n,
        f"{task} minimum sequence threshold",
    )
    validate_constant_metadata(
        raw,
        "minimum_recovery_rate",
        minimum_recovery,
        f"{task} minimum recovery threshold",
    )

    normalized = pd.DataFrame(
        {
            "gene": raw["gene"].astype(str).str.strip(),
            "n_test": pd.to_numeric(raw[count_column], errors="raise"),
            "recovery_rate": pd.to_numeric(raw[rate_column], errors="raise"),
        }
    )
    if normalized.isna().any().any() or (normalized["gene"] == "").any():
        raise ValueError(f"{task} recovery data contain missing or blank required values")
    if normalized["gene"].duplicated().any():
        duplicates = sorted(normalized.loc[normalized["gene"].duplicated(), "gene"].unique())
        raise ValueError(f"Duplicate {task} genes found: {duplicates[:5]}")
    if not np.isfinite(normalized["n_test"]).all():
        raise ValueError(f"{task} sequence counts must be finite")
    if not np.equal(normalized["n_test"], np.round(normalized["n_test"])).all():
        raise ValueError(f"{task} sequence counts must be integers")
    normalized["n_test"] = normalized["n_test"].astype(int)
    if (normalized["n_test"] <= 0).any():
        raise ValueError(f"{task} sequence counts must be positive for the log axis")
    rates = normalized["recovery_rate"]
    if not np.isfinite(rates).all() or not rates.between(0, 1).all():
        raise ValueError(f"{task} recovery rates must be finite values in [0, 1]")

    computed_selected = normalized["n_test"].ge(minimum_n) & rates.ge(minimum_recovery)
    if "selected" in raw.columns:
        supplied_selected = parse_boolean(raw["selected"], f"{task} selected").to_numpy()
        if not np.array_equal(supplied_selected, computed_selected.to_numpy()):
            mismatch = int(np.count_nonzero(supplied_selected != computed_selected.to_numpy()))
            raise ValueError(
                f"{mismatch} supplied {task} selected values conflict with the thresholds"
            )
    normalized["selected"] = computed_selected
    if not computed_selected.any():
        raise ValueError(f"No {task} genes satisfy the requested thresholds")
    normalized["task"] = task
    normalized.attrs["rows_before_positive_filter"] = before_filter
    normalized.attrs["rows_excluded_by_positive_filter"] = excluded
    normalized.attrs["minimum_n"] = minimum_n
    normalized.attrs["minimum_recovery"] = minimum_recovery
    return normalized.sort_values(["selected", "n_test", "gene"]).reset_index(drop=True)


def load_pathways(
    path: Path,
    task: str,
    selected_genes: set[str],
) -> pd.DataFrame:
    raw = pd.read_csv(require_file(path))
    required = {"overlap_size", "query_size_mapped", "fdr"}
    if not required.issubset(raw.columns):
        raise ValueError(
            f"{task} pathway CSV must contain {sorted(required)}; "
            f"found {list(raw.columns)}"
        )
    label_column = "display_term" if "display_term" in raw.columns else "term"
    if label_column not in raw.columns:
        raise ValueError(f"{task} pathway CSV must contain display_term or term")
    if raw.empty:
        raise ValueError(f"{task} pathway CSV is empty")
    if "task" in raw.columns:
        observed_tasks = set(raw["task"].astype(str).str.strip().str.lower())
        if observed_tasks != {task}:
            raise ValueError(
                f"{task} pathway file contains unexpected task values: {observed_tasks}"
            )

    data = raw.copy()
    if "display_order" in data.columns:
        display_order = pd.to_numeric(data["display_order"], errors="raise")
        if (
            display_order.isna().any()
            or not np.isfinite(display_order).all()
            or display_order.duplicated().any()
        ):
            raise ValueError(f"{task} display_order values must be unique and non-missing")
        data = data.assign(_display_order=display_order).sort_values(
            "_display_order", kind="stable"
        )

    if data[label_column].isna().any():
        raise ValueError(f"{task} pathway CSV contains missing display terms")
    display_term = data[label_column].astype(str).str.strip()
    overlap = pd.to_numeric(data["overlap_size"], errors="raise")
    mapped = pd.to_numeric(data["query_size_mapped"], errors="raise")
    fdr = pd.to_numeric(data["fdr"], errors="raise")
    if display_term.eq("").any() or display_term.duplicated().any():
        raise ValueError(f"{task} display terms must be non-blank and unique")
    for values, role in ((overlap, "overlap_size"), (mapped, "query_size_mapped")):
        if (
            values.isna().any()
            or not np.isfinite(values).all()
            or not np.equal(values, np.round(values)).all()
        ):
            raise ValueError(f"{task} {role} values must be integers")
    overlap = overlap.astype(int)
    mapped = mapped.astype(int)
    if (overlap < 2).any() or (mapped <= 0).any() or (overlap > mapped).any():
        raise ValueError(
            f"{task} pathway counts require 2 <= overlap_size <= query_size_mapped"
        )
    if (mapped > len(selected_genes)).any():
        raise ValueError(
            f"{task} query_size_mapped cannot exceed the number of selected genes"
        )
    if not np.isfinite(fdr).all() or not ((fdr > 0) & (fdr <= 1)).all():
        raise ValueError(f"{task} FDR values must be finite and in (0, 1]")
    if (fdr >= 0.05).any():
        failing = list(display_term.loc[fdr >= 0.05])
        raise ValueError(
            f"{task} panel input contains pathways with FDR >= 0.05: {failing[:5]}"
        )
    if {"source", "term"}.issubset(data.columns) and data.duplicated(
        ["source", "term"]
    ).any():
        raise ValueError(f"Duplicate source/term rows in the {task} pathway CSV")

    if "overlap_genes" in data.columns:
        for position, (listed, expected_overlap) in enumerate(
            zip(data["overlap_genes"], overlap), start=1
        ):
            if pd.isna(listed):
                raise ValueError(f"Missing {task} overlap_genes at pathway row {position}")
            genes = [
                gene.strip().upper()
                for gene in str(listed).split(";")
                if gene.strip()
            ]
            if len(genes) != len(set(genes)) or len(genes) != int(expected_overlap):
                raise ValueError(
                    f"{task} overlap_genes at row {position} do not match overlap_size"
                )
            unexpected = sorted(set(genes).difference(selected_genes))
            if unexpected:
                raise ValueError(
                    f"{task} pathway row {position} contains genes outside the selected "
                    f"set: {unexpected[:5]}"
                )
    else:
        warnings.warn(
            f"The {task} pathway CSV has no overlap_genes column; membership in the "
            "selected gene set cannot be cross-validated.",
            stacklevel=2,
        )

    ratio = overlap / mapped
    minus_log10_fdr = -np.log10(fdr)
    if "gene_ratio" in data.columns:
        supplied_ratio = pd.to_numeric(data["gene_ratio"], errors="raise")
        if not np.allclose(supplied_ratio, ratio, rtol=0, atol=1e-10):
            raise ValueError(f"Stored {task} gene_ratio values conflict with the counts")
    if "minus_log10_fdr" in data.columns:
        supplied_score = pd.to_numeric(data["minus_log10_fdr"], errors="raise")
        if not np.allclose(supplied_score, minus_log10_fdr, rtol=0, atol=1e-10):
            raise ValueError(f"Stored {task} minus_log10_fdr values conflict with FDR")

    normalized = pd.DataFrame(
        {
            "display_term": display_term.to_numpy(),
            "overlap_size": overlap.to_numpy(),
            "query_size_mapped": mapped.to_numpy(),
            "gene_ratio": ratio.to_numpy(float),
            "fdr": fdr.to_numpy(float),
            "minus_log10_fdr": minus_log10_fdr.to_numpy(float),
            "task": task,
        }
    )
    if len(normalized) > 12:
        warnings.warn(
            f"The {task} pathway panel contains {len(normalized)} rows. All rows will "
            "be retained, but a focused panel table is recommended for readability.",
            stacklevel=2,
        )
    return normalized.reset_index(drop=True)


def recovery_summary(data: pd.DataFrame) -> dict[str, float | int]:
    selected = data.loc[data["selected"]]
    sequence_count = int(selected["n_test"].sum())
    recovered_count = np.rint(
        selected["n_test"].to_numpy(float)
        * selected["recovery_rate"].to_numpy(float)
    ).astype(int)
    if (recovered_count < 0).any() or (
        recovered_count > selected["n_test"].to_numpy(int)
    ).any():
        raise ValueError("Reconstructed recovered-sequence counts are outside [0, n_test]")
    pooled = float(recovered_count.sum() / sequence_count)
    return {
        "selected_genes": int(len(selected)),
        "total_genes": int(len(data)),
        "selected_sequences": sequence_count,
        "pooled_recovery": pooled,
    }


def log_axis_limits_and_ticks(data: pd.DataFrame) -> tuple[tuple[float, float], np.ndarray]:
    minimum = float(data["n_test"].min())
    maximum = float(data["n_test"].max())
    lower_decade = int(np.floor(np.log10(minimum)))
    lower = max(np.nextafter(0.0, 1.0), 10.0**lower_decade * 0.80)
    upper = maximum * 1.18
    upper_decade = int(np.floor(np.log10(upper)))
    ticks = 10.0 ** np.arange(lower_decade, upper_decade + 1)
    return (lower, upper), ticks


def add_panel_heading(
    fig: mpl.figure.Figure,
    cell: mpl.transforms.Bbox,
    panel: str,
    title: str,
    panel_offset: float = 0.020,
) -> None:
    fig.text(
        cell.x0 - panel_offset,
        cell.y1 + 0.020,
        panel,
        ha="left",
        va="bottom",
        fontweight="bold",
        color=INK,
    )
    fig.text(
        cell.x0 + 0.020,
        cell.y1 + 0.020,
        title,
        ha="left",
        va="bottom",
        fontweight="bold",
        color=INK,
    )


def draw_recovery(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    task: str,
    show_legend: bool,
    show_summary: bool,
    threshold_n_label_mode: str = "below",
) -> None:
    meta = TASKS[task]
    color = meta["color"]
    selected = data.loc[data["selected"]]
    other = data.loc[~data["selected"]]
    minimum_n = int(data.attrs["minimum_n"])
    minimum_recovery = float(data.attrs["minimum_recovery"])
    summary = recovery_summary(data)

    ax.scatter(
        other["n_test"],
        other["recovery_rate"],
        s=14,
        facecolor=NEUTRAL,
        edgecolor=NEUTRAL_EDGE,
        linewidth=0.35,
        alpha=0.84,
        label="Other positive genes",
        zorder=2,
    )
    ax.scatter(
        selected["n_test"],
        selected["recovery_rate"],
        s=25,
        facecolor=color,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.95,
        label="Selected genes",
        zorder=4,
    )
    ax.axvline(minimum_n, color=color, linewidth=0.9, linestyle=(0, (4, 2)), zorder=1)
    ax.axhline(
        minimum_recovery,
        color=color,
        linewidth=0.9,
        linestyle=(0, (4, 2)),
        zorder=1,
    )
    limits, ticks = log_axis_limits_and_ticks(data)
    ax.set_xscale("log")
    ax.set_xlim(*limits)
    ax.set_ylim(0, 1.025)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Test sequences per gene")
    ax.set_ylabel("Per-gene recovery rate")
    if not np.any(np.isclose(ticks, minimum_n, rtol=0, atol=1e-12)):
        if threshold_n_label_mode not in {"below", "inside", "none"}:
            raise ValueError(
                "threshold_n_label_mode must be 'below', 'inside' or 'none'"
            )
        if threshold_n_label_mode == "inside":
            label_offset = (6, 6)
            horizontal_alignment = "left"
            vertical_alignment = "bottom"
        else:
            label_offset = (-4, -6)
            horizontal_alignment = "right"
            vertical_alignment = "top"
        if threshold_n_label_mode != "none":
            ax.annotate(
                f"{minimum_n}",
                xy=(minimum_n, 0),
                xycoords="data",
                xytext=label_offset,
                textcoords="offset points",
                color=color,
                ha=horizontal_alignment,
                va=vertical_alignment,
            )
    ax.annotate(
        f"{minimum_recovery:.2f}",
        xy=(1.0, minimum_recovery),
        xycoords=ax.get_yaxis_transform(),
        xytext=(4, 0),
        textcoords="offset points",
        color=color,
        ha="left",
        va="center",
        annotation_clip=False,
    )
    if show_legend:
        ax.legend(
            loc="lower right",
            borderaxespad=0.15,
            handletextpad=0.30,
            labelspacing=0.25,
        )
    if show_summary:
        ax.text(
            0.98,
            0.055,
            f"{summary['selected_genes']}/{summary['total_genes']} genes; "
            f"{summary['selected_sequences']:,} sequences\n"
            f"pooled recovery {summary['pooled_recovery']:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=INK,
        )


def draw_pathway(
    ax: mpl.axes.Axes,
    colorbar_ax: mpl.axes.Axes,
    data: pd.DataFrame,
    task: str,
    norm: Normalize,
    x_max: float,
) -> mpl.cm.ScalarMappable:
    color = TASKS[task]["color"]
    cmap = LinearSegmentedColormap.from_list(f"{task}_fdr", [PALE, color])
    y = np.arange(len(data))
    sizes = 12.0 * data["overlap_size"].to_numpy(float)
    ax.scatter(
        data["gene_ratio"],
        y,
        s=sizes,
        c=data["minus_log10_fdr"],
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    ax.set_yticks(y, labels=data["display_term"])
    ax.invert_yaxis()
    ax.set_xlim(0, x_max)
    tick_stop = np.floor(x_max / 0.25 + 1e-9) * 0.25
    ticks = np.arange(0, tick_stop + 0.001, 0.25)
    if len(ticks) < 2:
        ticks = np.linspace(0, x_max, 3)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2g"))
    ax.set_xlabel("Gene ratio")
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)

    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = mpl.colorbar.ColorbarBase(
        colorbar_ax,
        cmap=cmap,
        norm=norm,
        orientation="horizontal",
    )
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    midpoint = 5.0 if norm.vmin < 5.0 < norm.vmax else (norm.vmin + norm.vmax) / 2
    colorbar.set_ticks([norm.vmin, midpoint, norm.vmax])
    colorbar.ax.set_xticklabels(
        [f"{norm.vmin:.1f}", f"{midpoint:g}", f"{norm.vmax:g}"]
    )
    colorbar.ax.tick_params(axis="x", length=0, pad=6)
    colorbar.outline.set_linewidth(0.6)
    colorbar.ax.set_title("−log10(FDR)", loc="left", pad=3)
    return scalar


def representative_overlap_values(pathways: dict[str, pd.DataFrame]) -> list[int]:
    values = np.concatenate(
        [frame["overlap_size"].to_numpy(dtype=int) for frame in pathways.values()]
    )
    minimum = int(values.min())
    maximum = int(values.max())
    if minimum == maximum:
        return [minimum]
    candidates = np.unique(np.rint(np.linspace(minimum, maximum, 4)).astype(int))
    return [int(value) for value in candidates]


def add_overlap_legend(
    target: mpl.figure.Figure | mpl.axes.Axes,
    values: list[int],
    *,
    location: str,
    bbox: tuple[float, float],
) -> None:
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=float(np.sqrt(12.0 * value)),
            markerfacecolor="#B9BEC2",
            markeredgecolor="white",
            markeredgewidth=0.4,
        )
        for value in values
    ]
    target.legend(
        handles,
        [str(value) for value in values],
        title="Overlap genes",
        loc=location,
        bbox_to_anchor=bbox,
        ncol=len(values),
        borderaxespad=0,
        columnspacing=0.75,
        handletextpad=0.25,
        labelspacing=0.2,
    )


def enforce_typography(fig: mpl.figure.Figure, font_family: str) -> None:
    for item in fig.findobj(mpl.text.Text):
        item.set_fontfamily(font_family)
        item.set_fontsize(FONT_SIZE)
    sizes = {float(item.get_fontsize()) for item in fig.findobj(mpl.text.Text)}
    if sizes != {float(FONT_SIZE)}:
        raise RuntimeError(f"Unexpected rendered font sizes: {sorted(sizes)}")


def output_path(prefix: Path, extension: str) -> Path:
    return Path(f"{prefix}{extension}")


def save_outputs(fig: mpl.figure.Figure, prefix: Path, title: str) -> None:
    prefix = prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path(prefix, ".pdf")
    svg_path = output_path(prefix, ".svg")
    png_path = output_path(prefix, ".png")
    tiff_path = output_path(prefix, ".tiff")
    metadata = {
        "Creator": "figure6_pathways.py",
        "Title": title,
        "Subject": "Per-gene recovery and pathway enrichment",
    }
    fig.savefig(pdf_path, metadata=metadata)
    fig.savefig(svg_path, metadata={"Creator": "figure6_pathways.py"})
    fig.savefig(png_path, dpi=600, metadata={"Software": "figure6_pathways.py"})
    with Image.open(png_path) as rendered:
        rendered.convert("RGB").save(
            tiff_path,
            compression="tiff_lzw",
            dpi=(600, 600),
        )


def require_matplotlib_panel_alignment(
    fig: mpl.figure.Figure,
    axes: dict[str, mpl.axes.Axes],
    path: Path,
    tolerance_pt: float = 1.5,
) -> None:
    fig.canvas.draw()
    width_pt = fig.get_figwidth() * 72
    height_pt = fig.get_figheight() * 72
    panels = []
    rectangles: dict[str, tuple[float, float, float, float]] = {}
    for panel, axis in axes.items():
        box = axis.get_position()
        rectangle = (
            box.x0 * width_pt,
            box.y0 * height_pt,
            box.x1 * width_pt,
            box.y1 * height_pt,
        )
        rectangles[panel] = rectangle
        panels.append({"id": panel, "bbox_pt": list(rectangle)})

    for group in (("A", "B", "C"), ("D", "E", "F")):
        group_boxes = [rectangles[panel] for panel in group]
        for metric_values, metric_name in (
            ([box[1] for box in group_boxes], "bottom"),
            ([box[3] for box in group_boxes], "top"),
            ([box[2] - box[0] for box in group_boxes], "width"),
        ):
            if max(metric_values) - min(metric_values) > tolerance_pt:
                raise RuntimeError(
                    f"Panels {group} differ in {metric_name} by more than "
                    f"{tolerance_pt} pt"
                )
    for first, second in (("A", "B"), ("B", "C"), ("D", "E"), ("E", "F")):
        if rectangles[first][2] >= rectangles[second][0]:
            raise RuntimeError(f"Panel plot areas {first} and {second} overlap")

    manifest = {
        "schema_version": 1,
        "backend": "python-matplotlib",
        "figure": {"width_pt": width_pt, "height_pt": height_pt},
        "panels": panels,
        "row_groups": [
            {"id": "gene-recovery", "panels": ["A", "B", "C"]},
            {"id": "pathway-enrichment", "panels": ["D", "E", "F"]},
        ],
        "exemptions": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def pathway_scale(
    pathways: dict[str, pd.DataFrame],
) -> tuple[Normalize, float]:
    scores = np.concatenate(
        [frame["minus_log10_fdr"].to_numpy(float) for frame in pathways.values()]
    )
    ratios = np.concatenate(
        [frame["gene_ratio"].to_numpy(float) for frame in pathways.values()]
    )
    minimum_score = float(-np.log10(0.05))
    maximum_score = float(max(2.0, np.ceil(scores.max())))
    maximum_ratio = float(max(0.25, np.ceil(ratios.max() / 0.25) * 0.25))
    return Normalize(vmin=minimum_score, vmax=maximum_score, clip=True), maximum_ratio


def create_composite(
    recovery: dict[str, pd.DataFrame],
    pathways: dict[str, pd.DataFrame],
    font_family: str,
    prefix: Path,
) -> None:
    norm, x_max = pathway_scale(pathways)
    overlap_values = representative_overlap_values(pathways)
    fig = plt.figure(
        figsize=(COMPOSITE_WIDTH_MM * MM, COMPOSITE_HEIGHT_MM * MM),
        facecolor="white",
    )
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.92, 1.10],
        left=0.065,
        right=0.95,
        bottom=0.17,
        top=0.91,
        wspace=0.36,
        hspace=0.63,
    )
    axes: dict[str, mpl.axes.Axes] = {}

    for column, task in enumerate(TASK_ORDER):
        meta = TASKS[task]
        top_cell = grid[0, column].get_position(fig)
        top_ax = fig.add_axes([top_cell.x0, top_cell.y0, top_cell.width, top_cell.height])
        draw_recovery(
            top_ax,
            recovery[task],
            task,
            show_legend=False,
            show_summary=False,
            threshold_n_label_mode="inside",
        )
        axes[meta["recovery_panel"]] = top_ax
        add_panel_heading(
            fig,
            top_cell,
            meta["recovery_panel"],
            meta["title"],
        )

        bottom_cell = grid[1, column].get_position(fig)
        plot_left = bottom_cell.x0 + bottom_cell.width * 0.60
        pathway_ax = fig.add_axes(
            [plot_left, bottom_cell.y0, bottom_cell.width * 0.40, bottom_cell.height]
        )
        colorbar_ax = fig.add_axes(
            [
                plot_left + bottom_cell.width * 0.03,
                bottom_cell.y1 + 0.048,
                bottom_cell.width * 0.34,
                0.014,
            ]
        )
        draw_pathway(
            pathway_ax,
            colorbar_ax,
            pathways[task],
            task,
            norm,
            x_max,
        )
        axes[meta["pathway_panel"]] = pathway_ax
        add_panel_heading(
            fig,
            bottom_cell,
            meta["pathway_panel"],
            meta["title"],
        )

    add_overlap_legend(
        fig,
        overlap_values,
        location="lower right",
        bbox=(0.985, 0.012),
    )
    enforce_typography(fig, font_family)
    resolved_prefix = prefix.expanduser().resolve()
    manifest_path = output_path(resolved_prefix, "_alignment-layout.json")
    require_matplotlib_panel_alignment(fig, axes, manifest_path)
    save_outputs(fig, prefix, "Figure 6")
    plt.close(fig)


def create_recovery_panel(
    data: pd.DataFrame,
    task: str,
    font_family: str,
    output_dir: Path,
) -> None:
    meta = TASKS[task]
    fig = plt.figure(
        figsize=(RECOVERY_PANEL_WIDTH_MM * MM, RECOVERY_PANEL_HEIGHT_MM * MM),
        facecolor="white",
    )
    ax = fig.add_axes([0.20, 0.18, 0.70, 0.68])
    draw_recovery(ax, data, task, show_legend=True, show_summary=False)
    cell = mpl.transforms.Bbox.from_extents(0.08, 0.18, 0.96, 0.86)
    add_panel_heading(fig, cell, meta["recovery_panel"], meta["title"], panel_offset=0.055)
    enforce_typography(fig, font_family)
    stem = output_dir / f"Figure6{meta['recovery_panel']}_{task}_gene_recovery"
    save_outputs(fig, stem, f"Figure 6{meta['recovery_panel']}")
    plt.close(fig)


def create_pathway_panel(
    data: pd.DataFrame,
    task: str,
    norm: Normalize,
    x_max: float,
    overlap_values: list[int],
    font_family: str,
    output_dir: Path,
) -> None:
    meta = TASKS[task]
    fig = plt.figure(
        figsize=(PATHWAY_PANEL_WIDTH_MM * MM, PATHWAY_PANEL_HEIGHT_MM * MM),
        facecolor="white",
    )
    ax = fig.add_axes([0.49, 0.27, 0.47, 0.48])
    cax = fig.add_axes([0.61, 0.835, 0.31, 0.027])
    draw_pathway(ax, cax, data, task, norm, x_max)
    add_overlap_legend(
        ax,
        overlap_values,
        location="upper center",
        bbox=(0.5, -0.32),
    )
    cell = mpl.transforms.Bbox.from_extents(0.08, 0.22, 0.96, 0.82)
    add_panel_heading(fig, cell, meta["pathway_panel"], meta["title"], panel_offset=0.055)
    enforce_typography(fig, font_family)
    stem = output_dir / f"Figure6{meta['pathway_panel']}_{task}_pathway_enrichment"
    save_outputs(fig, stem, f"Figure 6{meta['pathway_panel']}")
    plt.close(fig)


def create_separate_panels(
    recovery: dict[str, pd.DataFrame],
    pathways: dict[str, pd.DataFrame],
    font_family: str,
    prefix: Path,
) -> Path:
    output_dir = Path(f"{prefix.expanduser().resolve()}_panels")
    output_dir.mkdir(parents=True, exist_ok=True)
    norm, x_max = pathway_scale(pathways)
    overlap_values = representative_overlap_values(pathways)
    for task in TASK_ORDER:
        create_recovery_panel(recovery[task], task, font_family, output_dir)
    for task in TASK_ORDER:
        create_pathway_panel(
            pathways[task],
            task,
            norm,
            x_max,
            overlap_values,
            font_family,
            output_dir,
        )
    return output_dir


def main() -> None:
    args = parse_args()
    minimum_ns = {
        "ferroptosis": args.ferro_min_n,
        "senescence": args.senescence_min_n,
        "pyroptosis": args.pyroptosis_min_n,
    }
    if any(value <= 0 for value in minimum_ns.values()):
        raise ValueError("All minimum sequence thresholds must be positive integers")
    if not 0 <= args.min_recovery <= 1:
        raise ValueError("--min-recovery must be between 0 and 1")

    font_family, font_path = choose_font(args.font, strict=args.strict_font)
    configure_matplotlib(font_family)
    recovery_paths = {
        "ferroptosis": args.ferro_recovery,
        "senescence": args.senescence_recovery,
        "pyroptosis": args.pyroptosis_recovery,
    }
    pathway_paths = {
        "ferroptosis": args.ferro_pathways,
        "senescence": args.senescence_pathways,
        "pyroptosis": args.pyroptosis_pathways,
    }
    recovery = {
        task: load_recovery(
            recovery_paths[task],
            task,
            minimum_ns[task],
            args.min_recovery,
        )
        for task in TASK_ORDER
    }
    pathways = {
        task: load_pathways(
            pathway_paths[task],
            task,
            set(recovery[task].loc[recovery[task]["selected"], "gene"].str.upper()),
        )
        for task in TASK_ORDER
    }

    create_composite(recovery, pathways, font_family, args.output_prefix)
    panel_directory = None
    if args.separate_panels:
        panel_directory = create_separate_panels(
            recovery,
            pathways,
            font_family,
            args.output_prefix,
        )

    print(f"Font: {font_family} ({font_path})")
    for task in TASK_ORDER:
        summary = recovery_summary(recovery[task])
        excluded = recovery[task].attrs["rows_excluded_by_positive_filter"]
        print(
            f"{TASKS[task]['title']}: selected "
            f"{summary['selected_genes']}/{summary['total_genes']} genes; "
            f"sequences={summary['selected_sequences']:,}; "
            f"pooled recovery={summary['pooled_recovery']:.6f}; "
            f"positive-filter exclusions={excluded}; "
            f"displayed pathways={len(pathways[task])}"
        )
    print(f"Composite: {args.output_prefix.expanduser().resolve()}.[pdf|svg|png|tiff]")
    if panel_directory is not None:
        print(f"Separate panels: {panel_directory}")


if __name__ == "__main__":
    main()
