"""The paper's earlier figure style (Dark2 colours, default font), kept by Figures 3 and 4 and the
ablation figure until they are redrawn in the house style (``style.py``)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from scripts.paper.figures.style import TITLES  # noqa: E402


# ColorBrewer Dark2, its first two colours (teal, orange); as a pair they
# pass the colour-vision-deficiency separation and contrast checks.
COLOR_DISCOVERED = "#1b9e77"
COLOR_SEED = "#d95f02"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
FULL_WIDTH_IN = 7.2  # a two-column paper's text width


# --- statistics -------------------------------------------------------------


def weighted_fit_stats(observed: np.ndarray, predicted: np.ndarray, n: np.ndarray) -> Tuple[float, float]:
    """Trial-weighted Pearson correlation and RMSE between observed and
    predicted proportions. The correlation is NaN when either side is
    constant (a model predicting 0.5 everywhere has no correlation)."""
    w = np.asarray(n, dtype=float) / np.sum(n)
    o = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mo, mp = np.sum(w * o), np.sum(w * p)
    cov = np.sum(w * (o - mo) * (p - mp))
    var_product = np.sum(w * (o - mo) ** 2) * np.sum(w * (p - mp) ** 2)
    r = cov / np.sqrt(var_product) if var_product > 0 else float("nan")
    rmse = np.sqrt(np.sum(w * (o - p) ** 2))
    return float(r), float(rmse)



# --- drawing ----------------------------------------------------------------


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_SECONDARY, labelsize=7, width=0.6)


def _pending(ax, domain: str) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"{TITLES[domain]}\nresults pending", ha="center", va="center",
            fontsize=8, color=INK_SECONDARY, transform=ax.transAxes)


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


