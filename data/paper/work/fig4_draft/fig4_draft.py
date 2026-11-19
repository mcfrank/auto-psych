"""Draft of Figure 4 (throwaway): per domain, the backward elimination of the winning model on one scale,
ELPD-CV against the full model. The line is the model kept after each round (each step named by the
mechanism it dropped); at the end, every remaining
removal (each kept mechanism), all of which cost at least 2 SE, so the elimination stopped.

Run: uv run python -m data.paper.work.fig4_draft.fig4_draft
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from pyprojroot import here  # noqa: E402

from scripts.paper.figures.icons import add_title_icons  # noqa: E402
from scripts.paper.figures.style import INK, INK_2, TITLE_FONT, TITLES, setup_style  # noqa: E402

OUT = here() / "data" / "paper" / "work" / "fig4_draft"
COLOR_NEW, COLOR_START, PATH = "#08306b", "#6baed6", "#262626"
NAME_PT, SMALL = 7.0, 7.0  # PNAS: no text under 6 pt at the printed size
WIDTH_IN, HEIGHT_IN, DPI = 7.0, 5.6, 600  # one row per domain  # the saved figure: PNAS's full width
# each panel's cues are named by what pushes its response this way (the fitted signs: mechanisms.yaml)
DIRECTION = {"subjective_randomness": "what makes a sequence look less random",
             "rsa": "what makes an object more likely to be chosen",
             "number_game": "what makes a target more likely"}


def sources():
    sr = json.loads((here() / "data/paper/figure_data/subjective_randomness.json").read_text())["pooled"]
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
    xo = lean_n - 1
    ax.axhline(0, color="#b0b0b0", lw=0.6, ls=(0, (2, 2)), zorder=0)  # the full model
    labels = []
    for c, yo in zip(options, opt_y):
        name, color, text = style_of(names, c["mechanism"])
        ax.errorbar([xo], [yo], yerr=[c["se_vs_start"]], fmt="none", ecolor=color, elinewidth=0.6, zorder=2)
        ax.scatter([xo], [yo], s=12, color=color, linewidths=0, zorder=4)
        labels.append((yo, name, text, yo))
    # the options' names in a column right of them, spread apart, a leader to each
    ax.set_ylim(lo - 0.03 * abs(lo), 0.03 * abs(lo))
    ax.set_xlim(n[0] + 0.5 + 0.3 * (n[0] - xo), xo - 0.2)  # room left for the first step's name
    fig = ax.figure
    fig.canvas.draw()
    to_data = ax.transData.inverted()
    per_pt = abs(to_data.transform((0, 0))[1] - to_data.transform((0, NAME_PT * 1.35 * fig.dpi / 72))[1])
    # what the path dropped and what the column kept, labelled above each
    note = {"fontsize": NAME_PT, "color": INK_2, "style": "italic"}
    # both on the x tick labels' line: "dropped" ends the path's, "kept" sits under the column
    ax.annotate("kept", (xo, lo), xytext=(0, -9), textcoords="offset points", ha="center", va="top",
                annotation_clip=False, **note)
    labels.sort(key=lambda t: -t[0])
    placed = spread([yo for yo, *_ in labels], per_pt)
    cap = ax.get_ylim()[1]  # never above the panel: push the top labels down, then their neighbours
    for k, yt in enumerate(placed):
        placed[k] = min(yt, cap if k == 0 else placed[k - 1] - per_pt)
    for (yo, name, text, at), yt in zip(labels, placed):
        ax.annotate(name, (xo, at), xytext=(xo - 0.6, yt), ha="left", va="center", fontsize=NAME_PT, **text,
                    annotation_clip=False, arrowprops={"arrowstyle": "-", "color": "#b0b0b0", "lw": 0.4,
                                                         "shrinkA": 1, "shrinkB": 3, "relpos": (0, 0.5)})
    ax.set_xticks(n)
    ax.tick_params(labelsize=SMALL, length=2, pad=1.5)
    ax.set_xlabel("mechanisms in the model (each step drops one)", fontsize=SMALL + 0.5, labelpad=1)
    ax.spines["bottom"].set_bounds(n[0], lean_n)


def figure(canvas_in, names, src):
    fig, axes = plt.subplots(3, 1, figsize=(canvas_in, HEIGHT_IN), dpi=DPI, gridspec_kw={"hspace": 0.42, "height_ratios": [0.8, 0.9, 1.1]})
    fig.subplots_adjust(left=0.1, right=0.76, top=0.95, bottom=0.09)  # the options' names to the right
    for ax, d in zip(axes, TITLES):
        draw(ax, json.loads(src[d].read_text()), names[d])
    for ax in axes:
        ax.set_ylabel("ELPD-CV vs. full model (nats)", fontsize=SMALL)
    # the titles, level, just above the tallest of the steps' names
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
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
    return fig, None


def main():
    setup_style()
    names = yaml.safe_load((OUT / "mechanisms.yaml").read_text())
    fig, _ = figure(WIDTH_IN, names, sources())
    for ext in ("png", "pdf"):  # uncropped: the canvas is the saved size, PNAS's full width
        fig.savefig(OUT / f"fig4_draft.{ext}", dpi=DPI)

if __name__ == "__main__":
    main()
