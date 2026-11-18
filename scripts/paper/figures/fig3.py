"""Figure 3 (``fig3_conditions``): per domain, all runs' data pooled, each condition's observed
choice proportion against the prediction of the best starting model (top row) and of the best
discovered model (bottom row), both fitted on the pooled data, with the trial-weighted correlation
and RMSE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from scripts.paper.figures.legacy import (COLOR_DISCOVERED, COLOR_SEED, FULL_WIDTH_IN, GRID, INK, _pending,
                                          _save, _style, weighted_fit_stats)
from scripts.paper.figures.style import panel_title



# Figure 3's point opacity where a domain's points pile up: the number
# game's 840 cells all have 9 responses, so every point is full size and
# they stack on the ten possible observed proportions.
FIG3_ALPHA = {"number_game": 0.1}
FIG3_HEIGHT_IN = 3.7  # panels wider than tall: the axes are not held square


def draw_conditions(domain_data: Dict[str, Optional[Dict[str, Any]]], out_dir: Path,
                    mock_domains: List[str]) -> None:
    rows = [("best_seed", "Best starting model", COLOR_SEED),
            ("best_model", "Best discovered model", COLOR_DISCOVERED)]
    fig, axes = plt.subplots(2, len(domain_data), figsize=(FULL_WIDTH_IN, FIG3_HEIGHT_IN))
    axes = np.asarray(axes).reshape(2, len(domain_data))
    for col, (domain, data) in enumerate(domain_data.items()):
        for row, (key, label, color) in enumerate(rows):
            ax = axes[row, col]
            if data is None:
                _pending(ax, domain)
                continue
            _style(ax)
            conditions = data["conditions"]
            observed = np.array([c["observed"] for c in conditions])
            predicted = np.array([c[key] for c in conditions])
            n = np.array([c["n"] for c in conditions])
            ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.0, zorder=0)
            sizes = 6 + 30 * np.sqrt(n / n.max())
            ax.scatter(predicted, observed, s=sizes, marker="o", color=color,
                       alpha=FIG3_ALPHA.get(domain, 0.25),
                       linewidths=0, zorder=1)
            r, rmse = weighted_fit_stats(observed, predicted, n)
            r_text = "r undefined" if np.isnan(r) else f"r = {r:.2f}"
            ax.text(0.04, 0.96, f"{r_text}\nRMSE = {rmse:.3f}", transform=ax.transAxes,
                    fontsize=6.5, color=INK, va="top", ha="left")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel(f"{label}: predicted", fontsize=7, color=INK)
            if col == 0:
                ax.set_ylabel("Observed proportion", fontsize=7, color=INK)
            if row == 0:
                ax.set_title(panel_title(domain, mock_domains=mock_domains),
                             fontsize=8.5, color=INK, loc="left")
    fig.tight_layout(h_pad=1.5, w_pad=1.5)
    _save(fig, out_dir, "fig3_conditions")
