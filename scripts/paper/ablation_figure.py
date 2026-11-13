"""The ablation figure (``fig_ablation``): the backward elimination of each
domain's overall winner, one panel each — the subjective-randomness final best
on all runs pooled (Figure 3's; ``docs/sr_ablation.md``), chain 0's
reference-game final (``docs/auto_rsa/ABLATION_C0.md``) and the number-game
dataset runs' final best on all runs pooled (the November sweep, from the
2016 paper's seeds; ``docs/ng_ablation.md``). A domain whose elimination is
not in yet can be drawn as "results pending" (``--pending-domains``); any
other missing record raises.

Each panel is the path of the model kept after each round: its grouped-CV
ELPD against the starting model's, by its number of mechanisms, each step
named by the mechanism it dropped, and the removal that stopped the
elimination (the round's cheapest, at ≥ 2 clustered SEs).

Usage:
    uv run python -m scripts.paper.ablation_figure
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from pyprojroot import here  # noqa: E402

from scripts.paper.figures import (  # noqa: E402
    COLOR_DISCOVERED,
    FULL_WIDTH_IN,
    INK,
    INK_SECONDARY,
    _pending,
    _save,
    _style,
)

SR_ABLATION = here() / "data" / "results" / "human_experiment_sr-oct26" / "analysis" / "results" / "ablation"
RSA_ELIMINATION = here() / "data" / "rsa" / "c0_ablation" / "eliminate" / "elimination.json"
NG_ELIMINATION = (here() / "data" / "results" / "number_game_dataset-nov26" / "analysis" / "ablation"
                  / "elimination.json")
PANEL_TITLES = {"subjective_randomness": "Subjective randomness", "rsa": "Reference games",
                "number_game": "Number sequence generalization"}

COLOR_KEPT = COLOR_DISCOVERED  # Dark2 green
COLOR_STOP = "#9a9892"
SURFACE = "white"


@dataclass
class Args:
    sr_figure_data: Path = here() / "data" / "paper" / "figure_data" / "subjective_randomness.json"
    """Names the final best on all runs pooled (``pooled.best_model``)."""
    out_dir: Path = here() / "data" / "paper" / "figures"
    pending_domains: List[str] = field(default_factory=list)
    """Domains drawn as a 'results pending' panel (their elimination is not in yet)."""


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
    the starting model's, the clustered SE of that difference, converged, the
    mechanism it dropped or None); and ``stop``: (mechanisms, ELPD-CV minus
    the starting model's, SE, mechanism) of the removal that stopped the
    elimination (the last round's converged candidate with the highest
    ELPD-CV, as the rule takes it), or None when no candidate converged. The
    SE is paired, against the starting model (zero for it): a model's own
    ELPD SE includes the item-to-item variation every model shares, several
    times the whole path."""
    rounds = elimination["rounds"]
    start = rounds[0]["elpd_cv"]
    kept = []
    for prev, rd in zip([None] + rounds[:-1], rounds):
        dropped = None if prev is None else (set(prev["mechanisms"]) - set(rd["mechanisms"])).pop()
        kept.append((len(rd["mechanisms"]), rd["elpd_cv"] - start, rd["se_vs_start"], rd["converged"], dropped))
    converged = [c for c in rounds[-1]["candidates"] if c["converged"]]
    stop = None
    if converged:
        best = max(converged, key=lambda c: c["elpd_cv"])
        stop = (len(rounds[-1]["mechanisms"]) - 1, best["elpd_cv"] - start, best["se_vs_start"], best["mechanism"])
    return {"kept": kept, "stop": stop}


def _display(mechanism: str) -> str:
    return mechanism.replace("_personal", " (personal)").replace("_", " ")


FLAG_ANGLE = 30.0  # degrees: each name rises from its point, parallel to its neighbours
NAME_FONTSIZE = 6


def name_points(ax, path: Dict[str, Any], renderer) -> List[Tuple[Any, int]]:
    """Name each step by the mechanism it dropped, anchored at its own point so
    no name can be read as another's: a flag rising up and to the right from
    the point, just right of its error bar (over the descending path's empty
    side; neighbours' flags run parallel), and the path's last point named
    beside it on the left. Returns
    each name with its point's index (the starting model, index 0, is not
    named). ``fit_names`` then makes room for the flags."""
    kept = path["kept"] + ([(*path["stop"][:3], True, path["stop"][3])] if path["stop"] else [])
    names: List[Tuple[Any, int]] = []
    for i, (x, y, _, _, dropped) in enumerate(kept):
        if dropped is None:
            continue
        style = dict(textcoords="offset points", fontsize=NAME_FONTSIZE, color=INK_SECONDARY)
        if i == len(kept) - 1:
            name = ax.annotate(_display(dropped), (x, y), xytext=(-6, 0), ha="right", va="center", **style)
        else:
            name = ax.annotate(_display(dropped), (x, y), xytext=(4.0, 1.5), ha="left", va="bottom",
                               rotation=FLAG_ANGLE, rotation_mode="anchor", **style)
        names.append((name, i))
    return names


def _text_corners(name, renderer) -> np.ndarray:
    """A name's rotated text rectangle, its four corners in display units."""
    probe = Text(text=name.get_text(), fontsize=name.get_fontsize(), figure=name.figure)
    box = probe.get_window_extent(renderer)
    w, h = box.width, box.height
    x0 = {"left": 0.0, "center": -w / 2, "right": -w}[name.get_horizontalalignment()]
    y0 = {"bottom": 0.0, "baseline": 0.0, "center": -h / 2, "center_baseline": -h / 2,
          "top": -h}[name.get_verticalalignment()]
    corners = np.array([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)])
    angle = np.radians(name.get_rotation())
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    name.update_positions(renderer)
    anchor = np.asarray(name.get_transform().transform(name.get_position()))
    return anchor + corners @ rotation.T


def _polygons_overlap(p: np.ndarray, q: np.ndarray) -> bool:
    """Whether two convex polygons intersect (separating-axis test)."""
    for poly in (p, q):
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            normal = np.array([a[1] - b[1], b[0] - a[0]])
            if (p @ normal).max() < (q @ normal).min() or (q @ normal).max() < (p @ normal).min():
                return False
    return True


def fit_names(axes, names: List[List[Tuple[Any, int]]], renderer, pad_pt: float = 2.0) -> None:
    """Raise the shared y-limit until every name fits inside its panel's top;
    the band added holds names only, so its ticks are hidden."""
    data_top = axes[0].get_ylim()[1]
    for _ in range(4):
        need = 0.0
        for ax, panel in zip(axes, names):
            top = ax.get_window_extent(renderer).y1
            for name, _ in panel:
                need = max(need, _text_corners(name, renderer)[:, 1].max() - top)
        if need <= 0:
            break
        ax = axes[0]
        lo, hi = ax.get_ylim()
        per_px = (hi - lo) / ax.get_window_extent(renderer).height
        ax.set_ylim(lo, hi + (need + pad_pt * renderer.points_to_pixels(1.0)) * per_px)
    lo, hi = axes[0].get_ylim()
    axes[0].set_yticks([t for t in axes[0].get_yticks() if lo <= t <= data_top])
    axes[0].set_ylim(lo, hi)


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


def draw_ablation(eliminations: List[Tuple[str, Optional[Dict[str, Any]]]], out_dir: Path) -> None:
    """One panel per domain; a domain whose elimination is ``None`` is drawn
    as "results pending"."""
    # Each panel as wide as its steps need.
    widths = [max(len(e["rounds"]) + 2, 8) if e else 8 for _, e in eliminations]  # room for the title
    fig, axes = plt.subplots(1, len(eliminations), figsize=(FULL_WIDTH_IN, 2.0), sharey=True,
                             gridspec_kw=dict(wspace=0.15, width_ratios=widths))
    paths: List[Optional[Dict[str, Any]]] = []
    for ax, letter, (domain, elimination) in zip(axes, "ABCD", eliminations):
        if elimination is None:
            _pending(ax, domain)
            paths.append(None)
        else:
            paths.append(draw_panel(ax, elimination, f"{letter}. {PANEL_TITLES[domain]}"))
    axes[0].set_ylabel("ELPD-CV vs the full model (nats)", fontsize=7, color=INK)
    fig.canvas.draw()  # fixes the shared limits the names are placed against
    renderer = fig.canvas.get_renderer()
    names = [name_points(ax, path, renderer) if path else [] for ax, path in zip(axes, paths)]
    fit_names(axes, names, renderer)
    handles = [
        Line2D([], [], color=COLOR_KEPT, lw=1.2, marker="o", ms=4, label="best model after each round (±1 SE of its difference from the full model)"),
        Line2D([], [], color=COLOR_STOP, lw=1.0, ls=(0, (4, 3)), marker="o", ms=4,
               label="removal that stopped the elimination"),
    ]
    if any(not k[3] for path in paths if path for k in path["kept"]):  # only where a hollow point is drawn
        handles.append(Line2D([], [], color="none", marker="o", ms=4, mfc=SURFACE, mec=COLOR_KEPT,
                              label="fit not converged"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=len(handles), fontsize=6.5,
               frameon=False, handletextpad=0.4, columnspacing=1.5)
    _save(fig, out_dir, "fig_ablation")


def main(args: Args) -> None:
    sources = elimination_sources(json.loads(args.sr_figure_data.read_text()))
    eliminations: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    for domain, path in sources:
        if domain in args.pending_domains:
            eliminations.append((domain, None))
        elif not path.exists():
            raise FileNotFoundError(f"no elimination record for {domain} at {path} (pass it in "
                                    "--pending-domains to draw its panel as pending)")
        else:
            eliminations.append((domain, json.loads(path.read_text())))
    for (domain, elimination), (_, path) in zip(eliminations, sources):
        if elimination is not None and any("se_vs_start" not in r for r in elimination["rounds"]):
            raise ValueError(f"{path} records no SE of each model's difference from the full model (se_vs_start): "
                             "rerun its ablation's plan step (it replays the saved CVs, no fits) and commit the "
                             "new elimination.json")
    draw_ablation(eliminations, args.out_dir)
    print(f"wrote fig_ablation to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
