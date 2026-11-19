"""Draft of Figure 4 (throwaway): per domain, the backward elimination of the winning model on one scale,
ELPD-CV against the full model. The line is the model kept after each round (each step named by the
mechanism it dropped); faint dots are the removals not taken that round; at the end, every remaining
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
NAME_PT, SMALL = 5.0, 5.5
CLIP = 3.0  # the axis reaches CLIP x the deepest of the path and the median option; options past it are arrowed


def sources():
    sr = json.loads((here() / "data/paper/figure_data/subjective_randomness.json").read_text())["pooled"]
    run = sr["winner_runs"][sr["best_model"]]
    return {"subjective_randomness": here() / f"data/results/human_experiment_sr-oct26/analysis/results/ablation/{run}/elimination.json",
            "rsa": here() / "data/rsa/c0_ablation/eliminate/elimination.json",
            "number_game": here() / "data/results/number_game_dataset-nov26/analysis/ablation/elimination.json"}


def style_of(names, m):
    name, origin = names[m]
    return name, (COLOR_NEW if origin == "new" else COLOR_START), ("D" if m.endswith("_personal") else "o")


def draw(ax, elimination, names):
    rounds = elimination["rounds"]
    base = rounds[0]["elpd_cv"]
    n = [len(r["mechanisms"]) for r in rounds]
    y = np.array([r["elpd_cv"] - base for r in rounds])
    se = np.array([r["se_vs_start"] or 0.0 for r in rounds])
    lean_n, lean_y = n[-1], y[-1]
    options = sorted(rounds[-1]["candidates"], key=lambda c: -c["elpd_cv"])
    opt_y = np.array([c["elpd_cv"] - base for c in options])
    lo = CLIP * min(y.min() - 1, np.median(opt_y))
    # the removals not taken, each round
    for k, r in enumerate(rounds[:-1]):
        gone = (set(r["mechanisms"]) - set(rounds[k + 1]["mechanisms"])).pop()
        ys = [c["elpd_cv"] - base for c in r["candidates"] if c["mechanism"] != gone]
        ys = [v for v in ys if v >= lo]  # those past the axis are left out
        ax.scatter([n[k] - 1] * len(ys), ys, s=3, color="#c8c8c8", linewidths=0, zorder=1)
    # the path, each step named by what it dropped
    ax.plot(n, y, color=PATH, lw=1.0, zorder=3)
    ax.errorbar(n, y, yerr=se, fmt="none", ecolor=PATH, elinewidth=0.6, zorder=3)
    for k in range(1, len(rounds)):
        gone = (set(rounds[k - 1]["mechanisms"]) - set(rounds[k]["mechanisms"])).pop()
        name, color, marker = style_of(names, gone)
        ax.scatter([n[k]], [y[k]], s=12 if marker == "o" else 9, marker=marker, color=color, linewidths=0, zorder=4)
        ax.annotate(name, (n[k], y[k]), xytext=(1, 4), textcoords="offset points", rotation=70, fontsize=NAME_PT,
                    color=INK_2, ha="left", va="bottom", rotation_mode="anchor")
    ax.scatter([n[0]], [0], s=12, color=PATH, zorder=4)
    # the options at the end: every kept mechanism's removal, each at least 2 SE
    xo = lean_n - 1
    ax.plot([lean_n, xo], [lean_y, lean_y], color="#c8c8c8", lw=0.6, ls=(0, (2, 2)), zorder=1)
    labels = []
    for c, yo in zip(options, opt_y):
        name, color, marker = style_of(names, c["mechanism"])
        if yo < lo:  # off the axis: an arrow at its foot, with its value
            ax.annotate("", (xo, lo), xytext=(xo, lo * 0.93), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.8})
            labels.append((lo, f"{name} ({yo:,.0f})".replace("-", "−"), color, lo))
            continue
        ax.errorbar([xo], [yo], yerr=[c["se_vs_start"]], fmt="none", ecolor=color, elinewidth=0.6, zorder=2)
        ax.scatter([xo], [yo], s=12 if marker == "o" else 9, marker=marker, color=color, linewidths=0, zorder=4)
        labels.append((yo, name, color, yo))
    # the options' names in a column right of them, spread apart, a leader to each
    ax.set_ylim(lo * 1.05, max(se.max(), 1) * 3)
    ax.set_xlim(n[0] + 0.6, xo - 0.3)
    fig = ax.figure
    fig.canvas.draw()
    to_data = ax.transData.inverted()
    per_pt = abs(to_data.transform((0, 0))[1] - to_data.transform((0, NAME_PT * 1.3 * fig.dpi / 72))[1])
    placed = []
    for yo, name, color, at in sorted(labels, key=lambda t: -t[0]):
        yt = min(yo, placed[-1] - per_pt) if placed else yo
        placed.append(yt)
        ax.annotate(name, (xo, at), xytext=(xo - 0.9, yt), ha="left", va="center", fontsize=NAME_PT, color=INK,
                    annotation_clip=False, arrowprops={"arrowstyle": "-", "color": "#b0b0b0", "lw": 0.4,
                                                         "shrinkA": 1, "shrinkB": 3, "relpos": (0, 0.5)})
    ax.axhline(0, color="#e0e0e0", lw=0.8, zorder=0)
    ax.set_xticks(n + [xo])
    ax.set_xticklabels([str(v) for v in n] + [""])
    ax.tick_params(labelsize=SMALL, length=2, pad=1.5)
    ax.set_xlabel("mechanisms in the model", fontsize=SMALL + 0.5, labelpad=1)
    ax.spines["bottom"].set_bounds(n[0], lean_n)
    depends = [names[m][0] for m in rounds[-1]["mechanisms"] if m not in {c["mechanism"] for c in options}]
    if depends:
        ax.text(0, -0.17, "others depend on:\n" + ",\n".join(depends), transform=ax.transAxes, fontsize=NAME_PT,
                color=INK_2, style="italic", va="top")


def main():
    setup_style()
    names = yaml.safe_load((OUT / "mechanisms.yaml").read_text())
    src = sources()
    # each panel as wide as its rounds, plus room for the options' names
    ratios = [len(json.loads(src[d].read_text())["rounds"]) + 3 for d in TITLES]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), gridspec_kw={"wspace": 1.3, "width_ratios": ratios})
    for ax, d in zip(axes, TITLES):
        draw(ax, json.loads(src[d].read_text()), names[d])
        ax.set_title(TITLES[d], fontdict=TITLE_FONT, pad=62)  # above the steps' names
    axes[0].set_ylabel("ELPD-CV vs. the full model (nats)", fontsize=SMALL + 0.5)
    dot = {"lw": 0, "ms": 3.5}
    key = [Line2D([], [], marker="o", color=INK_2, label="one value for everyone", **dot),
           Line2D([], [], marker="D", color=INK_2, label="fit per person", **{**dot, "ms": 3}),
           Line2D([], [], marker="o", color=COLOR_NEW, label="new", **dot),
           Line2D([], [], marker="o", color=COLOR_START, label="from the literature", **dot),
           Line2D([], [], marker="o", color="#c8c8c8", label="removals not taken", **{**dot, "ms": 2.5})]
    fig.legend(handles=key, loc="upper center", ncol=5, frameon=False, fontsize=SMALL, bbox_to_anchor=(0.5, -0.1),
               handletextpad=0.2, columnspacing=1.6)
    add_title_icons(fig, axes, list(TITLES))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_draft.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
