"""Figure 4 (``fig4_ablation``): per domain, the backward elimination of the winning model, on two axes.
Left, the path: ELPD-CV against the full model of the model kept after each round (each step's x tick named
by the mechanism it dropped), its bar ±1 SE of the difference from the previous round's model, the SE the
stopping rule judged that drop by. Right, the kept column: every removal from the final model (each kept
mechanism), ELPD-CV against the final model with ±1 SE of that difference; the cheapest costs at least 2 SE,
so the elimination stopped. Each SE is the one the elimination stored (``se`` of each candidate in
``src/ablation/eliminate.py``), clustered by the domain's CV unit: an unordered pair of sequences (subjective
randomness), a display or condition (reference games), an example set (number game). Each mechanism's name
and origin (new, or the starting model whose code had it) are in ``scripts/paper/mechanisms.yaml``.
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

from scripts.paper.figures.fig3 import BEST, SEED
from scripts.paper.figures.icons import add_title_icons
from scripts.paper.figures.style import DPI, INK, INK_2, TITLE_FONT, TITLES

# Figures 2 and 3's colours: blue for what auto-psych discovered, seed grey for what came from the literature;
# the path and every error bar in a neutral that is neither
COLOR_NEW, COLOR_START, PATH = BEST, SEED, INK_2
NAME_PT, SMALL = 7.0, 7.0
NAMES_IN, NAME_PITCH = 0.3, 1.15  # the kept names' offset from their column (inches), their spacing (line heights)
LEFT, RIGHT, TICK_DEG = 0.04, 0.04, 35  # shares of the path's axis before the first step and past the last; the step names' angle  # PNAS: no text under 6 pt at the printed size
PATH_IN, KEPT_IN, GAP_IN = 3.25, 0.5, 0.82  # widths: the path's axes, the kept column's, between them (inches)
LEFT_PAD, TOP_PAD, LEGEND_PAD, ROW_GAP = 0.1, 0.01, 0.0, 0.06  # figure shares (left, top, the legend's), inches between rows
ICON_OVER, ONE_LINE, KEPT_DOWN = 0.5, 14, 0.8  # how far each title's icon reaches over it (title heights); the longest one-line step name; the share of the step names' height the kept column extends down
WIDTH_IN, HEIGHT_IN = 7.0, 4.6  # one row per domain; the saved figure: PNAS's full width
# each panel's cues are named by what pushes its response this way (the fitted signs: mechanisms.yaml)
DIRECTION = {"subjective_randomness": "what makes a sequence look less random",
             "rsa": "what makes an object more likely to be chosen",
             "number_game": "what shapes which concepts the learner believes"}


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


def two_lines(name):
    """A long name broken in two at the space nearest its middle."""
    spaces = [i for i, ch in enumerate(name) if ch == " "]
    if len(name) <= ONE_LINE or not spaces:
        return name
    at = min(spaces, key=lambda i: abs(i - len(name) / 2))
    return name[:at] + "\n" + name[at + 1:]


def draw_path(ax, path, names):
    """The elimination path: ELPD-CV vs. the full model, each step's bar ±1 SE of its difference from the
    previous round's model (the SE its drop was judged by)."""
    n = [p["n"] for p in path]
    y = np.array([p["y"] for p in path])
    se = np.array([p["se"] or 0.0 for p in path])
    ax.plot(n, y, color=PATH, lw=1.0, zorder=3)
    ax.errorbar(n[1:], y[1:], yerr=se[1:], fmt="none", ecolor=PATH, elinewidth=0.6, zorder=3)
    for p in path[1:]:
        _, color, _ = style_of(names, p["dropped"])
        ax.scatter([p["n"]], [p["y"]], s=12, color=color, linewidths=0, zorder=4)
    ax.scatter([n[0]], [0], s=12, color=INK, marker="s", linewidths=0, zorder=4)
    ax.axhline(0, color="#b0b0b0", lw=0.6, ls=(0, (2, 2)), zorder=0)  # the full model
    span = (n[0] - n[-1]) / (1 - LEFT - RIGHT)  # the axis's span, in steps
    ax.set_xlim(n[0] + LEFT * span, n[-1] - RIGHT * span)
    lo, hi = min(0.0, (y - se).min()), max(0.0, (y + se).max())  # a removal can improve the fit: the path can rise above the full model
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    # each step's tick named by the mechanism it dropped
    ax.set_xticks(n, ["full model"] + [two_lines(style_of(names, p["dropped"])[0]) for p in path[1:]])
    ax.tick_params(labelsize=NAME_PT, length=2, pad=1.5)
    for tick in ax.get_xticklabels():  # in ink: the points carry the origin's colour
        tick.set(rotation=TICK_DEG, ha="right", va="top", rotation_mode="anchor", multialignment="right", linespacing=1.0,
                 color=INK)
    ax.spines["bottom"].set_bounds(n[0], n[-1])
    ax.set_ylabel("ELPD-CV vs.\nfull model (nats)", fontsize=SMALL)


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
        ax.errorbar([xo], [k["y"]], yerr=[k["se"]], fmt="none", ecolor=PATH, elinewidth=0.6, zorder=2)
        ax.scatter([xo], [k["y"]], s=12, color=color, linewidths=0, zorder=4)
        labels.append((k["y"], name, text))
    # the names in a column right of the axes, spread apart, a leader to each
    per_pt = per_inch(ax, 1) * NAME_PT * NAME_PITCH / 72
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
    # "kept" heads the column: the names never rise above the axes, so no leader crosses it
    ax.set_xticks([])
    ax.annotate("kept", (xo, 1), xycoords=("data", "axes fraction"), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=SMALL + 0.5, color=INK)
    ax.tick_params(labelsize=SMALL, length=2, pad=1.5)
    ax.spines["bottom"].set_bounds(-0.25, 0.25)
    ax.set_ylabel("ELPD-CV vs. model\nafter removals (nats)", fontsize=SMALL)


def stack_rows(fig, grid, n_kept):
    """Place the rows top to bottom, each path axes with room above for its title (and icon) and below for
    its step names and x label, which differ by domain; the axes share the height left over."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    H = fig.get_figheight() * fig.dpi
    above, below = [], []
    for ax in grid[:, 0]:
        box = ax.get_window_extent(renderer)
        title = ax.title.get_window_extent(renderer)
        above.append(title.y1 + ICON_OVER * title.height - box.y1)  # the icon is taller than the title
        below.append(box.y0 - ax.xaxis.get_tightbbox(renderer).y0)
    room = H * (1 - TOP_PAD - LEGEND_PAD) - sum(above) - sum(below) - ROW_GAP * fig.dpi * (len(above) - 1)
    height = room / len(above)
    top = H * (1 - TOP_PAD)
    x0 = LEFT_PAD
    for (ax, kept_ax), a, b, k in zip(grid, above, below, n_kept):
        y0 = (top - a - height) / H
        ax.set_position((x0, y0, PATH_IN / WIDTH_IN, height / H))
        # the kept column also takes the height the path's step names use: its names run longer than the path
        need = k * NAME_PT * NAME_PITCH * fig.dpi / 72 - height  # only as far as its names need
        down = min(KEPT_DOWN * b, max(KEPT_DOWN * b / 2, need))  # at least half the stretch: a short column still spreads
        kept_ax.set_position((x0 + (PATH_IN + GAP_IN) / WIDTH_IN, y0 - down / H, KEPT_IN / WIDTH_IN,
                              (height + down) / H))
        top -= a + height + b + ROW_GAP * fig.dpi


def draw_ablation(figure_data: Path, mechanisms: Path, out_dir: Path) -> None:
    names = yaml.safe_load(mechanisms.read_text())
    src = sources(figure_data)
    # each domain a row: the path and the kept column side by side, each with its own y axis
    fig, grid = plt.subplots(3, 2, figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI)
    axes = grid[:, 0]
    points = {d: ablation_points(json.loads(src[d].read_text())) for d in TITLES}
    for ax, d in zip(axes, TITLES):
        draw_path(ax, points[d][0], names[d])
        title = ax.set_title(TITLES[d], fontdict=TITLE_FONT, pad=8, x=0.12, ha="left")  # the icon goes left of it
        ax.annotate(DIRECTION[d], (1, 0.5), xycoords=title, xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=NAME_PT, color=INK_2, style="italic")
    axes[-1].set_xlabel("mechanism removed, in order", fontsize=SMALL + 0.5, labelpad=1)  # the same in every row
    stack_rows(fig, grid, [len(points[d][1]) for d in TITLES])
    for kept_ax, d in zip(grid[:, 1], TITLES):  # its names are spread in inches: drawn once the rows are placed
        draw_kept(kept_ax, points[d][1], names[d])
    dot = {"lw": 0, "ms": 3.5}
    key = [Line2D([], [], marker="o", color=COLOR_NEW, label="new", **dot),
           Line2D([], [], marker="o", color=COLOR_START, label="from the literature", **dot)]
    fig.canvas.draw()  # the key on the x label's line, under the kept column: no line of its own
    label = axes[-1].xaxis.label.get_window_extent()
    y = fig.transFigure.inverted().transform((0, (label.y0 + label.y1) / 2))[1]
    fig.legend(handles=key, loc="center left", ncol=2, frameon=False, fontsize=SMALL,
               bbox_to_anchor=(grid[-1, 1].get_position().x0, y), handletextpad=0.2, columnspacing=1.6)
    add_title_icons(fig, axes, list(TITLES))
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):  # uncropped: the canvas is the saved size, PNAS's full width
        fig.savefig(out_dir / f"fig4_ablation.{ext}", dpi=DPI)
    plt.close(fig)
