"""The ablation figure (``fig_ablation``): the backward elimination of each
domain's overall winner, one panel each — the subjective-randomness final best
on all runs pooled (Figure 3's; ``docs/sr_ablation.md``) and the reference-game
consolidation (``docs/auto_rsa/CONSOLIDATION.md``).

Each panel is the path of the model kept after each round: its grouped-CV
ELPD against the starting model's, by its number of mechanisms, each step
named by the mechanism it dropped, and the removal that stopped the
elimination (the round's cheapest, at ≥ 2 clustered SEs). The number game was
not ablated.

Usage:
    uv run python -m scripts.paper.ablation_figure
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tyro  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from pyprojroot import here  # noqa: E402

from scripts.paper.figures import (  # noqa: E402
    COLOR_DISCOVERED,
    FULL_WIDTH_IN,
    INK,
    INK_SECONDARY,
    _save,
    _style,
)

SR_ABLATION = here() / "data" / "results" / "human_experiment_sr-oct26" / "analysis" / "results" / "ablation"
RSA_ELIMINATION = here() / "data" / "rsa" / "consolidate" / "eliminate" / "elimination.json"
PANEL_TITLES = {"subjective_randomness": "Subjective randomness", "rsa": "Reference games"}

COLOR_KEPT = COLOR_DISCOVERED  # Dark2 green
COLOR_TWO_SE = "#9a9892"
SURFACE = "white"
THRESHOLD_SE = 2.0


@dataclass
class Args:
    sr_figure_data: Path = here() / "data" / "paper" / "figure_data" / "subjective_randomness.json"
    """Names the final best on all runs pooled (``pooled.best_model``)."""
    out_dir: Path = here() / "data" / "paper" / "figures"


def best_sr_elimination(figure_data: Dict[str, Any]) -> Tuple[str, Path]:
    """The subjective-randomness final best on all runs pooled, and its run's
    elimination record."""
    best = figure_data["pooled"]["best_model"]
    return best, SR_ABLATION / figure_data["pooled"]["winner_runs"][best] / "elimination.json"


def elimination_path(elimination: Dict[str, Any]) -> Dict[str, Any]:
    """The model kept after each round, ``kept``: (mechanisms, ELPD-CV minus
    the starting model's, converged, the mechanism it dropped or None); and
    ``stop``: (mechanisms, ELPD-CV minus the starting model's, mechanism) of
    the removal that stopped the elimination (the last round's converged
    candidate with the highest ELPD-CV, as the rule takes it), or None when
    no candidate converged; and ``two_se_mark``: the bar that removal fell
    below, 2 clustered SEs of its comparison under the last kept model (None
    without a stop)."""
    rounds = elimination["rounds"]
    start = rounds[0]["elpd_cv"]
    kept = []
    for prev, rd in zip([None] + rounds[:-1], rounds):
        dropped = None if prev is None else (set(prev["mechanisms"]) - set(rd["mechanisms"])).pop()
        kept.append((len(rd["mechanisms"]), rd["elpd_cv"] - start, rd["converged"], dropped))
    converged = [c for c in rounds[-1]["candidates"] if c["converged"]]
    stop = two_se_mark = None
    if converged:
        best = max(converged, key=lambda c: c["elpd_cv"])
        stop = (len(rounds[-1]["mechanisms"]) - 1, best["elpd_cv"] - start, best["mechanism"])
        two_se_mark = kept[-1][1] - THRESHOLD_SE * best["se"]
    return {"kept": kept, "stop": stop, "two_se_mark": two_se_mark}


def _display(mechanism: str) -> str:
    return mechanism.replace("_personal", " (personal)").replace("_", " ")


def _name(ax, x: float, y: float, text: str, y_prev: float, y_next: Optional[float]) -> None:
    """Name a point on the side its line leaves free: above a peak, below a
    trough, beside a point the line passes through (to the right, above a
    falling line, below a rising one), to the left of the path's end."""
    if y_next is None:
        at, ha, va = (-7, 0), "right", "center"
    elif y >= max(y_prev, y_next):
        at, ha, va = (0, 5), "center", "bottom"
    elif y <= min(y_prev, y_next):
        at, ha, va = (0, -6), "center", "top"
    elif y_next < y:
        at, ha, va = (5, 3), "left", "bottom"
    else:
        at, ha, va = (5, -3), "left", "top"
    ax.annotate(text, (x, y), xytext=at, textcoords="offset points", ha=ha, va=va, fontsize=6,
                color=INK_SECONDARY)


def draw_panel(ax, elimination: Dict[str, Any], title: str) -> None:
    path = elimination_path(elimination)
    xs = [k[0] for k in path["kept"]]
    ys = [k[1] for k in path["kept"]]
    ax.plot(xs, ys, color=COLOR_KEPT, lw=1.2, zorder=2)
    ax.scatter(xs, ys, s=16, color=[COLOR_KEPT if k[2] else SURFACE for k in path["kept"]],
               edgecolor=COLOR_KEPT, lw=0.9, zorder=3)
    path_ys = ys + ([path["stop"][1]] if path["stop"] else [])
    for i, (x, y, _, dropped) in enumerate(path["kept"]):
        if dropped is not None:
            _name(ax, x, y, _display(dropped), path_ys[i - 1], path_ys[i + 1] if i + 1 < len(path_ys) else None)
    if path["stop"] is not None:
        x, y, mechanism = path["stop"]
        ax.plot([xs[-1], x], [ys[-1], y], color=COLOR_KEPT, lw=1.2, zorder=2)
        ax.scatter([x], [y], s=16, color=COLOR_KEPT, edgecolor=COLOR_KEPT, lw=0.9, zorder=3)
        _name(ax, x, y, _display(mechanism), ys[-1], None)
        ax.axhline(path["two_se_mark"], color=COLOR_TWO_SE, lw=0.8, ls=(0, (4, 3)), zorder=0)
    low = min(xs + ([path["stop"][0]] if path["stop"] else []))
    ax.set_xlim(max(xs) + 0.5, low - 0.6)  # fewer mechanisms to the right
    ax.set_xticks(range(low, max(xs) + 1))
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.set_xlabel("mechanisms in the model", fontsize=7, color=INK)
    _style(ax)


def draw_ablation(eliminations: List[Tuple[str, Optional[str], Dict[str, Any]]], out_dir: Path) -> None:
    # Each panel as wide as its steps need.
    widths = [len(e["rounds"]) + 2 for _, _, e in eliminations]
    fig, axes = plt.subplots(1, len(eliminations), figsize=(FULL_WIDTH_IN, 2.6), sharey=True,
                             gridspec_kw=dict(wspace=0.15, width_ratios=widths))
    for ax, letter, (domain, _, elimination) in zip(axes, "ABCD", eliminations):
        draw_panel(ax, elimination, f"{letter}. {PANEL_TITLES[domain]}")
    axes[0].set_ylabel("ELPD-CV vs the full model (nats)", fontsize=7, color=INK)
    handles = [
        Line2D([], [], color=COLOR_KEPT, lw=1.2, marker="o", ms=4, label="best model after each round"),
        Line2D([], [], color=COLOR_TWO_SE, lw=0.8, ls=(0, (4, 3)), label="2 SE below the final model"),
        Line2D([], [], color="none", marker="o", ms=4, mfc=SURFACE, mec=COLOR_KEPT, label="fit not converged"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.5,
               frameon=False, handletextpad=0.4, columnspacing=1.5)
    _save(fig, out_dir, "fig_ablation")


def main(args: Args) -> None:
    sr_final, sr_path = best_sr_elimination(json.loads(args.sr_figure_data.read_text()))
    eliminations = [(domain, final, json.loads(path.read_text()))
                    for domain, final, path in [("subjective_randomness", sr_final, sr_path),
                                                ("rsa", None, RSA_ELIMINATION)]]
    draw_ablation(eliminations, args.out_dir)
    print(f"wrote fig_ablation to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
