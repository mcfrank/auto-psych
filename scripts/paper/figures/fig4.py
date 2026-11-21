"""Figure 4 (``fig4_ablation``): per domain, the backward elimination of the winning model, on two axes.
Left, the path: ELPD-CV against the full model of the model kept after each round (each step named by the
mechanism it dropped), its bar ±1 SE of the difference from the previous round's model, the SE the stopping
rule judged that drop by. Right, the kept column: every removal from the final model (each kept mechanism),
ELPD-CV against the final model with ±1 SE of that difference; the cheapest costs at least 2 SE, so the
elimination stopped. Each SE is the stimulus-clustered SE the elimination stored (``se`` of each candidate in
``src/ablation/eliminate.py``). Each mechanism's name and origin (new, or the starting model whose code had
it) are in ``scripts/paper/mechanisms.yaml``.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from pyprojroot import here

from scripts.paper.figures.icons import add_title_icons
from scripts.paper.figures.style import DPI, INK, INK_2, TITLE_FONT, TITLES

COLOR_NEW, COLOR_START, PATH = "#08306b", "#6baed6", "#262626"
NAME_PT, SMALL = 7.0, 7.0
NAMES_IN = 0.3  # the kept names' offset from their column (inches)
LEFT, RIGHT = 0.3, 0.04  # shares of the path's axis: room for the first name, past the last step  # PNAS: no text under 6 pt at the printed size
PATH_IN, KEPT_IN, GAP_IN = 3.25, 0.5, 0.82  # widths: the path's axes, the kept column's, between them (inches)
WIDTH_IN, HEIGHT_IN = 7.0, 4.9  # one row per domain; the saved figure: PNAS's full width
# each panel's cues are named by what pushes its response this way (the fitted signs: mechanisms.yaml)
DIRECTION = {"subjective_randomness": "what makes a sequence look less random",
             "rsa": "what makes an object more likely to be chosen",
             "number_game": "patterns the number shares with the examples"}


def sources(figure_data: Path):
    """Each domain's elimination of its winning model (subjective randomness: the pooled data's best)."""
    sr = json.loads((figure_data / "subjective_randomness.json").read_text())["pooled"]
    run = sr["winner_runs"][sr["best_model"]]
    return {"subjective_randomness": here() / f"data/results/human_experiment_sr-oct26/analysis/results/ablation/{run}/elimination.json",
            "rsa": here() / "data/rsa/c0_ablation/eliminate/elimination.json",
            "number_game": here() / "data/results/number_game_dataset-nov26/analysis/ablation_ew/elimination.json"}


def style_of(names, m):
    """A mechanism's name, its point's colour (its origin) and its name's style."""
    name, origin = names[m]
    return name, (COLOR_NEW if origin == "new" else COLOR_START), {"color": INK}


def spread(ys, gap):
    """Label heights for points at ``ys`` (top first): each as near its point as it can be with at least
    ``gap`` between neighbours; a run of crowded labels is centred on its points."""
    groups = []  # runs of labels pushed together: (their points, the top label's height)
    for y in ys:
        groups.append([[y], y])
        while len(groups) > 1 and groups[-2][1] - gap * len(groups[-2][0]) < groups[-1][1]:
            pts, _ = groups.pop()
            groups[-1][0] += pts
        for g in groups[-1:]:  # centre the run on its points
            g[1] = np.mean(g[0]) + gap * (len(g[0]) - 1) / 2
        while len(groups) > 1 and groups[-2][1] - gap * len(groups[-2][0]) < groups[-1][1]:
            pts, _ = groups.pop()
            groups[-1][0] += pts
            groups[-1][1] = np.mean(groups[-1][0]) + gap * (len(groups[-1][0]) - 1) / 2
    return [top - gap * k for pts, top in groups for k in range(len(pts))]


def ablation_points(elimination):
    """What Figure 4 draws of one elimination, each bar the SE the stopping rule used for that comparison.

    The path: each round's model (``n`` mechanisms, ``y`` its ELPD-CV minus the full model's), ``se`` the
    clustered SE of its difference from the previous round's model (the dropped mechanism's ``se`` among
    the previous round's candidates; ``None`` for the full model). The kept column: each removal from the
    final model, cheapest first, ``y`` its ELPD-CV minus the final model's and ``se`` its SE vs the final
    model (the comparisons on which the elimination stopped)."""
    rounds = elimination["rounds"]
    base = rounds[0]["elpd_cv"]
    path = [{"n": len(rounds[0]["mechanisms"]), "y": 0.0, "se": None, "dropped": None}]
    for prev, cur in pairwise(rounds):
        (gone,) = set(prev["mechanisms"]) - set(cur["mechanisms"])
        step = next(c for c in prev["candidates"] if c["mechanism"] == gone)
        path.append({"n": len(cur["mechanisms"]), "y": cur["elpd_cv"] - base, "se": step["se"], "dropped": gone})
    final = rounds[-1]
    kept = [{"mechanism": c["mechanism"], "y": c["elpd_cv"] - final["elpd_cv"], "se": c["se"]}
            for c in sorted(final["candidates"], key=lambda c: -c["elpd_cv"])]
    return path, kept


def per_inch(ax, axis=0):
    """Data units per inch along the axes' x (0) or y (1) axis."""
    pos, fig = ax.get_position(), ax.figure
    span = abs(np.diff(ax.get_xlim() if axis == 0 else ax.get_ylim())[0])
    return span / (pos.width * fig.get_figwidth() if axis == 0 else pos.height * fig.get_figheight())


def draw_path(ax, path, names):
    """The elimination path: ELPD-CV vs. the full model, each step's bar ±1 SE of its difference from the
    previous round's model (the SE its drop was judged by)."""
    n = [p["n"] for p in path]
    y = np.array([p["y"] for p in path])
    se = np.array([p["se"] or 0.0 for p in path])
    ax.plot(n, y, color=PATH, lw=1.0, zorder=3)
    ax.errorbar(n[1:], y[1:], yerr=se[1:], fmt="none", ecolor=PATH, elinewidth=0.6, zorder=3)
    notes = []
    for p in path[1:]:  # each step named by what it dropped
        name, color, text = style_of(names, p["dropped"])
        ax.scatter([p["n"]], [p["y"]], s=12, color=color, linewidths=0, zorder=4)
        notes.append(ax.annotate(name, (p["n"], p["y"]), xytext=(-2, -5), textcoords="offset points", rotation=30,
                                 fontsize=NAME_PT, ha="right", va="center", rotation_mode="anchor", **text))
    ax.scatter([n[0]], [0], s=12, color=PATH, zorder=4)
    ax.axhline(0, color="#b0b0b0", lw=0.6, ls=(0, (2, 2)), zorder=0)  # the full model
    span = (n[0] - n[-1]) / (1 - LEFT - RIGHT)  # the axis's span, in steps
    ax.set_xlim(n[0] + LEFT * span, n[-1] - RIGHT * span)  # room left for the first step's name
    lo, hi = min(0.0, (y - se).min()), max(0.0, (y + se).max())  # a removal can improve the fit: the path can rise above the full model
    lo, hi = lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo)
    # room below for the names hanging under their points: a name reaching a share f of the axes' height
    # below its point at y fits when (y - lo) / (hi - lo) >= f, i.e. lo <= (y - f hi) / (1 - f)
    renderer = ax.figure.canvas.get_renderer()
    height = ax.get_window_extent(renderer).height
    for p, note in zip(path[1:], notes):
        f = (ax.transData.transform((p["n"], p["y"]))[1] - note.get_window_extent(renderer).y0) / height
        lo = min(lo, (p["y"] - f * hi) / (1 - f))
    ax.set_ylim(lo, hi)
    ax.set_xticks(n, [str(v) for v in n])
    ax.tick_params(labelsize=SMALL, length=2, pad=1.5)
    ax.set_xlabel("mechanisms in the model", fontsize=SMALL + 0.5, labelpad=1)
    ax.spines["bottom"].set_bounds(n[0], n[-1])
    ax.set_ylabel("ELPD-CV vs. full model (nats)", fontsize=SMALL)


def draw_kept(ax, kept, names):
    """The final model's mechanisms: each one's removal, ELPD-CV vs. the final model, ±1 SE of that
    difference (the stopping rule's: the cheapest removal costs at least 2)."""
    xo = 0.0
    ax.axhline(0, color="#b0b0b0", lw=0.6, ls=(0, (2, 2)), zorder=0)  # the final model
    ax.set_xlim(-0.5, 0.5)
    ky = np.array([k["y"] for k in kept])
    kse = np.array([k["se"] for k in kept])
    lo, hi = (ky - kse).min(), 0.0
    ax.set_ylim(lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))
    labels = []
    for k in kept:
        name, color, text = style_of(names, k["mechanism"])
        ax.errorbar([xo], [k["y"]], yerr=[k["se"]], fmt="none", ecolor=color, elinewidth=0.6, zorder=2)
        ax.scatter([xo], [k["y"]], s=12, color=color, linewidths=0, zorder=4)
        labels.append((k["y"], name, text))
    # the names in a column right of the axes, spread apart, a leader to each
    per_pt = per_inch(ax, 1) * NAME_PT * 1.35 / 72
    labels.sort(key=lambda t: -t[0])
    placed = spread([yo for yo, *_ in labels], per_pt)
    cap = ax.get_ylim()[1]  # never above the panel: push the top labels down, then their neighbours
    for k, yt in enumerate(placed):
        placed[k] = min(yt, cap if k == 0 else placed[k - 1] - per_pt)
    xt = xo + NAMES_IN * per_inch(ax, 0)
    for (yo, name, text), yt in zip(labels, placed):
        ax.annotate(name, (xo, yo), xytext=(xt, yt), ha="left", va="center", fontsize=NAME_PT, **text,
                    annotation_clip=False, arrowprops={"arrowstyle": "-", "color": "#b0b0b0", "lw": 0.4,
                                                         "shrinkA": 1, "shrinkB": 3, "relpos": (0, 0.5)})
    ax.set_xticks([xo], ["kept"])
    ax.get_xticklabels()[0].set(fontsize=SMALL + 0.5, color=INK)
    ax.tick_params(labelsize=SMALL, length=2, pad=1.5)
    ax.spines["bottom"].set_bounds(-0.25, 0.25)
    ax.set_ylabel("ELPD-CV vs. final model (nats)", fontsize=SMALL)


def draw_ablation(figure_data: Path, mechanisms: Path, out_dir: Path) -> None:
    names = yaml.safe_load(mechanisms.read_text())
    src = sources(figure_data)
    # each domain a row: the path and the kept column side by side, each with its own y axis
    fig, grid = plt.subplots(3, 2, figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI,
                             gridspec_kw={"hspace": 0.62, "width_ratios": [PATH_IN, KEPT_IN],
                                          "wspace": GAP_IN / ((PATH_IN + KEPT_IN) / 2)})
    left = 0.1  # the kept names to the right
    fig.subplots_adjust(left=left, right=left + (PATH_IN + GAP_IN + KEPT_IN) / WIDTH_IN, top=0.95, bottom=0.1)
    axes = grid[:, 0]
    for (ax, kept_ax), d in zip(grid, TITLES):
        path, kept = ablation_points(json.loads(src[d].read_text()))
        draw_path(ax, path, names[d])
        draw_kept(kept_ax, kept, names[d])
    fig.canvas.draw()
    for ax, d in zip(axes, TITLES):
        title = ax.set_title(TITLES[d], fontdict=TITLE_FONT, pad=8, x=0.12, ha="left")  # the icon goes left of it
        ax.annotate(DIRECTION[d], (1, 0.5), xycoords=title, xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=NAME_PT, color=INK_2, style="italic")
    dot = {"lw": 0, "ms": 3.5}
    key = [Line2D([], [], marker="o", color=COLOR_NEW, label="new", **dot),
           Line2D([], [], marker="o", color=COLOR_START, label="from the literature", **dot)]
    fig.legend(handles=key, loc="upper center", ncol=2, frameon=False, fontsize=SMALL, bbox_to_anchor=(0.43, 0.04),
               handletextpad=0.2, columnspacing=1.6)
    add_title_icons(fig, axes, list(TITLES))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):  # uncropped: the canvas is the saved size, PNAS's full width
        fig.savefig(out_dir / f"fig4_ablation.{ext}", dpi=DPI)
    plt.close(fig)
