"""Figure 4 (``fig4_ablation``): per domain, the backward elimination of the winning model on one scale,
ELPD-CV against the full model. The line is the model kept after each round (each step named by the
mechanism it dropped); at the end, every remaining
removal (each kept mechanism), all of which cost at least 2 SE, so the elimination stopped. Each mechanism's
name and origin (new, or the starting model whose code had it) are in ``scripts/paper/mechanisms.yaml``.
"""

from __future__ import annotations

import json
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
BREAK_IN, NAMES_IN = 0.06, 0.3  # the kept column's axis half-length, the names' offset (inches)
LEFT, COLUMN, COLUMN_AT = 0.27, 0.1, 0.07  # shares of each axis: room for the first name, past the path, to the column  # PNAS: no text under 6 pt at the printed size
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
            "number_game": here() / "data/results/number_game_dataset-nov26/analysis/ablation/elimination.json"}


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


def draw(ax, elimination, names):
    rounds = elimination["rounds"]
    base = rounds[0]["elpd_cv"]
    n = [len(r["mechanisms"]) for r in rounds]
    y = np.array([r["elpd_cv"] - base for r in rounds])
    se = np.array([r["se_vs_start"] or 0.0 for r in rounds])
    lean_n, lean_y = n[-1], y[-1]
    options = sorted(rounds[-1]["candidates"], key=lambda c: -c["elpd_cv"])
    opt_y = np.array([c["elpd_cv"] - base for c in options])
    lo = min(y.min(), opt_y.min())
    # the path, each step named by what it dropped
    ax.plot(n, y, color=PATH, lw=1.0, zorder=3)
    ax.errorbar(n, y, yerr=se, fmt="none", ecolor=PATH, elinewidth=0.6, zorder=3)
    for k in range(1, len(rounds)):
        gone = (set(rounds[k - 1]["mechanisms"]) - set(rounds[k]["mechanisms"])).pop()
        name, color, text = style_of(names, gone)
        ax.scatter([n[k]], [y[k]], s=12, color=color, linewidths=0, zorder=4)
        ax.annotate(name, (n[k], y[k]), xytext=(-2, -5), textcoords="offset points", rotation=30,
                    fontsize=NAME_PT, ha="right", va="center", rotation_mode="anchor", **text)
    ax.scatter([n[0]], [0], s=12, color=PATH, zorder=4)
    # the options at the end: every kept mechanism's removal, each at least 2 SE
    # the kept column a fixed distance past the path (in inches, the same in every panel): the panels' widths are
    # equal, so a fixed share of each one's span is the same length
    span = (n[0] - lean_n) / (1 - LEFT - COLUMN)  # the axis's span, in steps
    xo = lean_n - COLUMN_AT * span
    ax.axhline(0, color="#b0b0b0", lw=0.6, ls=(0, (2, 2)), zorder=0)  # the full model
    ax.set_xlim(n[0] + LEFT * span, lean_n - COLUMN * span)  # room left for the first step's name
    per_in = abs(np.diff(ax.get_xlim())[0]) / (ax.get_position().width * ax.figure.get_figwidth())  # steps per inch
    labels = []
    for c, yo in zip(options, opt_y):
        name, color, text = style_of(names, c["mechanism"])
        ax.errorbar([xo], [yo], yerr=[c["se_vs_start"]], fmt="none", ecolor=color, elinewidth=0.6, zorder=2)
        ax.scatter([xo], [yo], s=12, color=color, linewidths=0, zorder=4)
        labels.append((yo, name, text, (xo, yo)))
    # the options' names in a column right of them, spread apart, a leader to each
    ax.set_ylim(lo - 0.03 * abs(lo), 0.03 * abs(lo))
    fig = ax.figure
    fig.canvas.draw()
    to_data = ax.transData.inverted()
    per_pt = abs(to_data.transform((0, 0))[1] - to_data.transform((0, NAME_PT * 1.35 * fig.dpi / 72))[1])
    labels.sort(key=lambda t: -t[0])
    placed = spread([yo for yo, *_ in labels], per_pt)
    cap = ax.get_ylim()[1]  # never above the panel: push the top labels down, then their neighbours
    for k, yt in enumerate(placed):
        placed[k] = min(yt, cap if k == 0 else placed[k - 1] - per_pt)
    for (yo, name, text, at), yt in zip(labels, placed):
        ax.annotate(name, at, xytext=(xo - NAMES_IN * per_in, yt), ha="left", va="center", fontsize=NAME_PT, **text,
                    annotation_clip=False, arrowprops={"arrowstyle": "-", "color": "#b0b0b0", "lw": 0.4,
                                                         "shrinkA": 1, "shrinkB": 3, "relpos": (0, 0.5)})
    # the x axis: the path's counts, a gap, then "kept" under the kept column on its own stretch of axis
    ax.set_xticks(n + [xo], [str(v) for v in n] + ["kept"])
    ax.get_xticklabels()[-1].set(fontsize=SMALL + 0.5, color=INK)
    ax.tick_params(labelsize=SMALL, length=2, pad=1.5)
    ax.set_xlabel("mechanisms in the model", fontsize=SMALL + 0.5, labelpad=1)
    ax.spines["bottom"].set_bounds(n[0], lean_n)
    base = ax.get_xaxis_transform()  # x in data, y in axes fractions
    gap = BREAK_IN * per_in
    ax.plot([xo + 2 * gap, xo - 2 * gap], [0, 0], transform=base, color=ax.spines["bottom"].get_edgecolor(),
            lw=ax.spines["bottom"].get_linewidth(), clip_on=False, solid_capstyle="butt")


def draw_ablation(figure_data: Path, mechanisms: Path, out_dir: Path) -> None:
    names = yaml.safe_load(mechanisms.read_text())
    src = sources(figure_data)
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI, gridspec_kw={"hspace": 0.62})
    fig.subplots_adjust(left=0.1, right=0.76, top=0.95, bottom=0.1)  # the options' names to the right
    for ax, d in zip(axes, TITLES):
        draw(ax, json.loads(src[d].read_text()), names[d])
        ax.set_ylabel("ELPD-CV vs. full model (nats)", fontsize=SMALL)
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
