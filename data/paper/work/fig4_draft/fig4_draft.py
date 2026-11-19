"""Draft of the restyled Figure 4 (throwaway): per domain, a ladder of the winning model's mechanisms,
each at the cost (ELPD-CV, nats) of removing it: the ones the backward elimination dropped, at the cost
when they went, then the ones it kept, at the cost of removing each from the reduced model.

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

from scripts.paper.figures.style import INK, INK_2, TITLE_FONT, setup_style  # noqa: E402
from scripts.paper.figures.icons import STIMULI  # noqa: E402

OUT = here() / "data" / "paper" / "work" / "fig4_draft"
TITLES = {"subjective_randomness": "Subjective randomness", "rsa": "Reference games", "number_game": "Number game"}
# colour: the mechanism's origin (mechanisms.yaml); marker: shared (one value) or person-level (fit per person)
COLOR_NEW, COLOR_START = "#08306b", "#6baed6"
MARKER = {False: "o", True: "D"}
NAME_PT, SMALL = 5.0, 5.5


def sources():
    sr = json.loads((here() / "data/paper/figure_data/subjective_randomness.json").read_text())["pooled"]
    run = sr["winner_runs"][sr["best_model"]]
    return {"subjective_randomness": here() / f"data/results/human_experiment_sr-oct26/analysis/results/ablation/{run}/elimination.json",
            "rsa": here() / "data/rsa/c0_ablation/eliminate/elimination.json",
            "number_game": here() / "data/results/number_game_dataset-nov26/analysis/ablation/elimination.json"}


def ladder(elimination):
    """Rows, top to bottom: each dropped mechanism (its cost and SE in the round it went), then each kept
    one (the cost and SE of removing it from the reduced model, cheapest first); a kept mechanism no
    removal could take (others depend on it) has no cost."""
    rounds = elimination["rounds"]
    rows = []
    for prev, rd in zip(rounds, rounds[1:]):
        gone = (set(prev["mechanisms"]) - set(rd["mechanisms"])).pop()
        c = next(c for c in prev["candidates"] if c["mechanism"] == gone)
        rows.append({"mechanism": gone, "cost": c["cost"], "se": c["se"], "kept": False})
    last = {c["mechanism"]: c for c in rounds[-1]["candidates"]}
    kept = [{"mechanism": m, "cost": last[m]["cost"] if m in last else None,
             "se": last[m]["se"] if m in last else None, "kept": True} for m in rounds[-1]["mechanisms"]]
    kept.sort(key=lambda r: (r["cost"] is None, r["cost"] or 0))
    return rows + kept, len(rounds[0]["mechanisms"]), len(rounds[-1]["mechanisms"])


def draw(ax, domain, elimination, names):
    rows, n_full, n_lean = ladder(elimination)
    y = np.arange(len(rows))
    n_dropped = sum(not r["kept"] for r in rows)
    ax.axvline(0, color="#e0e0e0", lw=0.8, zorder=0)
    ax.axhspan(-0.5, n_dropped - 0.5, color="#f4f4f4", zorder=-1, lw=0)  # under the zero line
    for yi, r in zip(y, rows):
        name, origin = names[r["mechanism"]]
        color = COLOR_NEW if origin == "new" else COLOR_START
        marker = MARKER[r["mechanism"].endswith("_personal")]
        ax.text(-0.02, yi, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=NAME_PT,
                color=INK if r["kept"] else INK_2)
        if r["cost"] is None:
            ax.text(0.02, yi, "others depend on it", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=NAME_PT - 0.5, color=INK_2, style="italic")
            continue
        ax.plot([r["cost"] - 2 * r["se"], r["cost"] + 2 * r["se"]], [yi, yi], color=color, lw=1.0, zorder=2,
                solid_capstyle="butt")
        ax.scatter([r["cost"]], [yi], s=12 if marker == "o" else 9, marker=marker,
                   color=color, linewidths=0, zorder=3)
    ax.text(0.97, 1 - (n_dropped / 2) / len(rows), "dropped", transform=ax.transAxes, ha="right", va="center",
            fontsize=NAME_PT, color=INK_2, style="italic")
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=SMALL, length=2, pad=1.5)
    ax.set_xlabel("cost of removing it (nats)", fontsize=SMALL + 0.5, labelpad=1)
    ax.set_title(TITLES[domain], fontdict=TITLE_FONT, pad=14)
    ax.text(0.5, 1.01, f"{n_full} to {n_lean} mechanisms", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=SMALL, color=INK_2)


def main():
    setup_style()
    names = yaml.safe_load((OUT / "mechanisms.yaml").read_text())
    src = sources()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw={"wspace": 1.15})
    for ax, d in zip(axes, TITLES):
        if d not in src:
            ax.axis("off")
            ax.text(0.5, 0.5, "November ablation pending", transform=ax.transAxes, ha="center", va="center",
                    fontsize=SMALL, color=INK_2, style="italic")
            ax.set_title(TITLES[d], fontdict=TITLE_FONT, pad=14)
            continue
        draw(ax, d, json.loads(src[d].read_text()), names[d])
    dot = {"lw": 0, "ms": 3.5}
    key = [Line2D([], [], marker="o", color=COLOR_NEW, label="new", **dot),
           Line2D([], [], marker="o", color=COLOR_START, label="from a starting model (literature)", **dot),
           Line2D([], [], marker="o", color=INK_2, label="one value for everyone", **dot),
           Line2D([], [], marker="D", color=INK_2, label="fit per person", **{**dot, "ms": 3})]
    fig.legend(handles=key, loc="upper center", ncol=4, frameon=False, fontsize=SMALL, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.2, columnspacing=1.0)
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    for ax, d in zip(axes, TITLES):
        box = ax.title.get_window_extent()
        (x0, y0), (x1, y1) = to_fig.transform((box.x0, box.y0)), to_fig.transform((box.x1, box.y1))
        drawer, aspect = STIMULI[d]
        h = 1.9 * (y1 - y0)
        w = h * aspect * fig.get_figheight() / fig.get_figwidth()
        drawer(fig.add_axes((x0 - w - 0.008, (y0 + y1) / 2 - h / 2, w, h)))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_draft.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
