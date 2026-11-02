"""The ablation figure (``fig_ablation``): the backward elimination of each
domain's overall winner, one panel each — the subjective-randomness final best
on all runs pooled (Figure 3's; ``docs/sr_ablation.md``), chain 0's
reference-game final (``docs/auto_rsa/ABLATION_C0.md``) and the number-game
dataset run's final (``docs/ng_ablation.md``).

Each panel is the path of the model kept after each round: its grouped-CV
ELPD against the starting model's, by its number of mechanisms, each step
named by the mechanism it dropped, and the removal that stopped the
elimination (the round's cheapest, at ≥ 2 clustered SEs).

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
RSA_ELIMINATION = here() / "data" / "rsa" / "c0_ablation" / "eliminate" / "elimination.json"
NG_ELIMINATION = (here() / "data" / "results" / "number_game_dataset-oct26" / "analysis" / "ablation"
                  / "elimination.json")
PANEL_TITLES = {"subjective_randomness": "Subjective randomness", "rsa": "Reference games",
                "number_game": "Number game"}

COLOR_KEPT = COLOR_DISCOVERED  # Dark2 green
COLOR_STOP = "#9a9892"
SURFACE = "white"


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


def elimination_sources(sr_figure_data: Dict[str, Any]) -> List[Tuple[str, Path]]:
    """Each domain's elimination record, in the paper's domain order."""
    _, sr_path = best_sr_elimination(sr_figure_data)
    return [("subjective_randomness", sr_path), ("rsa", RSA_ELIMINATION), ("number_game", NG_ELIMINATION)]


def elimination_path(elimination: Dict[str, Any]) -> Dict[str, Any]:
    """The model kept after each round, ``kept``: (mechanisms, ELPD-CV minus
    the starting model's, the clustered SE of its ELPD-CV, converged, the
    mechanism it dropped or None); and ``stop``: (mechanisms, ELPD-CV minus
    the starting model's, SE, mechanism) of the removal that stopped the
    elimination (the last round's converged candidate with the highest
    ELPD-CV, as the rule takes it), or None when no candidate converged."""
    rounds = elimination["rounds"]
    start = rounds[0]["elpd_cv"]
    kept = []
    for prev, rd in zip([None] + rounds[:-1], rounds):
        dropped = None if prev is None else (set(prev["mechanisms"]) - set(rd["mechanisms"])).pop()
        kept.append((len(rd["mechanisms"]), rd["elpd_cv"] - start, rd["elpd_se"], rd["converged"], dropped))
    converged = [c for c in rounds[-1]["candidates"] if c["converged"]]
    stop = None
    if converged:
        best = max(converged, key=lambda c: c["elpd_cv"])
        stop = (len(rounds[-1]["mechanisms"]) - 1, best["elpd_cv"] - start, best["elpd_se"], best["mechanism"])
    return {"kept": kept, "stop": stop}


def _display(mechanism: str) -> str:
    return mechanism.replace("_personal", " (personal)").replace("_", " ")


ABOVE, BELOW = ((0, 5), "center", "bottom"), ((0, -6), "center", "top")


def _placements(y: float, y_prev: float, y_next: Optional[float]) -> List[Tuple[Tuple[int, int], str, str]]:
    """Where to name a point, best first: on the side its line leaves free —
    above a peak, below a trough, beside a point the line passes through (to
    the right, above a falling line, below a rising one; else to the left, on
    the other side), to the left of the path's end — then centred above or
    below."""
    if y_next is None:
        return [((-7, 0), "right", "center")]
    if y >= max(y_prev, y_next):
        return [ABOVE, BELOW]
    if y <= min(y_prev, y_next):
        return [BELOW, ABOVE]
    if y_next < y:
        return [((5, 3), "left", "bottom"), ((-5, -3), "right", "top"), ABOVE, BELOW]
    return [((5, -3), "left", "top"), ((-5, 3), "right", "bottom"), BELOW, ABOVE]


def _name(ax, x: float, y: float, text: str, placement: Tuple[Tuple[int, int], str, str]):
    at, ha, va = placement
    return ax.annotate(text, (x, y), xytext=at, textcoords="offset points", ha=ha, va=va, fontsize=6,
                       color=INK_SECONDARY)


def name_points(ax, path: Dict[str, Any], renderer) -> None:
    """Name each step by the mechanism it dropped, at its first placement that
    stays inside the axes and overlaps no name already placed, preferring one
    that does not cross the path either (its first placement when none fits)."""
    xs = [k[0] for k in path["kept"]] + ([path["stop"][0]] if path["stop"] else [])
    ys = [k[1] for k in path["kept"]] + ([path["stop"][1]] if path["stop"] else [])
    line = ax.transData.transform([(x0 + (x1 - x0) * t / 20, y0 + (y1 - y0) * t / 20)
                                   for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:]) for t in range(21)])
    steps = [(x, y, dropped) for x, y, _, _, dropped in path["kept"]]
    if path["stop"] is not None:
        x, y, _, mechanism = path["stop"]
        steps.append((x, y, mechanism))
    frame = ax.get_window_extent(renderer)
    placed = []
    for i, (x, y, dropped) in enumerate(steps):
        if dropped is None:
            continue
        boxes = []
        for placement in _placements(y, ys[i - 1], ys[i + 1] if i + 1 < len(ys) else None):
            name = _name(ax, x, y, _display(dropped), placement)
            box = name.get_window_extent(renderer)
            name.remove()
            fits = frame.x0 <= box.x0 and box.x1 <= frame.x1 and not any(box.overlaps(b) for b in placed)
            boxes.append((placement, box, fits, fits and not any(box.contains(*pt) for pt in line)))
        placement, box, _, _ = next((b for b in boxes if b[3]), next((b for b in boxes if b[2]), boxes[0]))
        _name(ax, x, y, _display(dropped), placement)
        placed.append(box)


def draw_panel(ax, elimination: Dict[str, Any], title: str) -> Dict[str, Any]:
    """Draw the path, unnamed (``name_points`` names it once the axes' limits
    are final), and return it."""
    path = elimination_path(elimination)
    xs = [k[0] for k in path["kept"]]
    ys = [k[1] for k in path["kept"]]
    ax.plot(xs, ys, color=COLOR_KEPT, lw=1.2, zorder=2)
    ax.errorbar(xs, ys, yerr=[k[2] for k in path["kept"]], fmt="none", ecolor=COLOR_KEPT, elinewidth=0.8,
                capsize=0, zorder=2)
    ax.scatter(xs, ys, s=16, color=[COLOR_KEPT if k[3] else SURFACE for k in path["kept"]],
               edgecolor=COLOR_KEPT, lw=0.9, zorder=3)
    if path["stop"] is not None:
        x, y, se, _ = path["stop"]
        ax.plot([xs[-1], x], [ys[-1], y], color=COLOR_STOP, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.errorbar([x], [y], yerr=[se], fmt="none", ecolor=COLOR_STOP, elinewidth=0.8, capsize=0, zorder=2)
        ax.scatter([x], [y], s=16, color=COLOR_STOP, edgecolor=COLOR_STOP, lw=0.9, zorder=3)
    low = min(xs + ([path["stop"][0]] if path["stop"] else []))
    ax.set_xlim(max(xs) + 0.5, low - 0.6)  # fewer mechanisms to the right
    ax.set_xticks(range(low, max(xs) + 1))
    ax.set_title(title, fontsize=8, color=INK, loc="left")
    ax.set_xlabel("mechanisms in the model", fontsize=7, color=INK)
    _style(ax)
    return path


def draw_ablation(eliminations: List[Tuple[str, Dict[str, Any]]], out_dir: Path) -> None:
    # Each panel as wide as its steps need.
    widths = [max(len(e["rounds"]) + 2, 8) for _, e in eliminations]  # room for the title
    fig, axes = plt.subplots(1, len(eliminations), figsize=(FULL_WIDTH_IN, 2.0), sharey=True,
                             gridspec_kw=dict(wspace=0.15, width_ratios=widths))
    paths = [draw_panel(ax, elimination, f"{letter}. {PANEL_TITLES[domain]}")
             for ax, letter, (domain, elimination) in zip(axes, "ABCD", eliminations)]
    axes[0].set_ylabel("ELPD-CV vs the full model (nats)", fontsize=7, color=INK)
    fig.canvas.draw()  # fixes the shared limits the names are placed against
    for ax, path in zip(axes, paths):
        name_points(ax, path, fig.canvas.get_renderer())
    handles = [
        Line2D([], [], color=COLOR_KEPT, lw=1.2, marker="o", ms=4, label="best model after each round (±1 SE)"),
        Line2D([], [], color=COLOR_STOP, lw=1.0, ls=(0, (4, 3)), marker="o", ms=4,
               label="removal that stopped the elimination"),
        Line2D([], [], color="none", marker="o", ms=4, mfc=SURFACE, mec=COLOR_KEPT, label="fit not converged"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=6.5,
               frameon=False, handletextpad=0.4, columnspacing=1.5)
    _save(fig, out_dir, "fig_ablation")


def main(args: Args) -> None:
    sources = elimination_sources(json.loads(args.sr_figure_data.read_text()))
    eliminations = [(domain, json.loads(path.read_text())) for domain, path in sources]
    for (domain, elimination), (_, path) in zip(eliminations, sources):
        if any("elpd_se" not in r for r in elimination["rounds"]):
            raise ValueError(f"{path} records no model's own ELPD SE (elpd_se): rerun its ablation's plan step "
                             "(it replays the saved CVs, no fits) and commit the new elimination.json")
    draw_ablation(eliminations, args.out_dir)
    print(f"wrote fig_ablation to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
