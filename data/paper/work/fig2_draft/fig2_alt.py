"""Draft of the restyled Figure 2 (throwaway; folded into scripts/paper once the look is agreed).

Run: uv run python data/paper/work/fig2_draft/fig2_draft.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from fontTools.ttLib import TTCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
from pyprojroot import here  # noqa: E402

from scripts.paper.figures import forest_rows, number_game_unseen_view  # noqa: E402

OUT = here() / "data" / "paper" / "work" / "fig2_draft"

# --- house style (rational-reps plot_utils: Avenir, its palette, open axes) ---
AVENIR = Path.home() / ".fonts" / "Avenir.ttc"
FACES = {"Avenir Book": 0, "Avenir Book Oblique": 1, "Avenir Medium": 8}  # TTC face indices


def setup_style():
    """Avenir Book for text, Medium for titles, Book Oblique for italics. matplotlib reads only
    a TTC's first face, so each face is written out once as its own TTF."""
    collection = TTCollection(str(AVENIR))
    for name, index in FACES.items():
        path = OUT / "fonts" / f"{name.replace(' ', '')}.ttf"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            collection.fonts[index].save(str(path))
        fm.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": "Avenir", "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "cm",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.edgecolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    })


INK, INK_2 = "#262626", "#595959"
BLUE, ORANGE, PURPLE, GRAY, GREEN = "#4C72B0", "#DD8452", "#8172B3", "#8C8C8C", "#55A868"  # rational-reps palette
COLOR_DISCOVERED, COLOR_SEED, COLOR_PROMOTED, COLOR_OTHER = BLUE, "#3d3d3d", "#a6a6a6", "#b3b3b3"
TITLE_FONT = {"family": "Avenir", "weight": "medium"}

TITLES = {"subjective_randomness": "Subjective randomness", "rsa": "Reference games",
          "number_game": "Number game"}


def subtitle(domain, data):
    return {"subjective_randomness": "LOO, all runs pooled · 23,104 trials",
            "rsa": "pre-registered held-out experiment · 14,404 trials",
            "number_game": "held-out example sets · 15,090 responses"}[domain]


# --- data --------------------------------------------------------------------

def load():
    fd = here() / "data" / "paper" / "figure_data"
    data = {d: json.loads((fd / f"{d}.json").read_text()) for d in TITLES}
    unseen = json.loads((here() / "data" / "paper" / "number_game_unseen_items.json").read_text())
    data["number_game"] = number_game_unseen_view(data["number_game"], unseen, "unseen_example_sets")
    names = yaml.safe_load((here() / "scripts/paper/model_names.yaml").read_text())
    papers = yaml.safe_load((here() / "scripts/paper/seed_papers.yaml").read_text())
    return data, names, papers


def rsa_vs_textbook(rows, data):
    """Re-zero reference games on the best textbook seed (Frank & Goodman's RSA with a shared
    prior): every value shifted. The winners keep the SE of the pre-registered test (paired with
    the best promoted seed, whose line the figure draws); the seeds' SEs are dropped."""
    shift = -data["best_textbook"]["diff"]
    out = []
    for r in rows:
        r = dict(r)
        if r["kind"] == "winner":
            r["diff"] += shift
        elif r["kind"] == "nonwinning":
            r["values"] = [v + shift for v in r["values"]]
        else:
            r.update(diff=r["diff"] + shift, se=None)
        out.append(r)
    promoted = set(data["promoted_seeds"])
    seeds = [r for r in out if r["kind"] == "seed" and r["model"] not in promoted]
    seeds.append({"kind": "seed", "model": "rsa_l1_shared_prior", "label": "RSA + shared prior",
                  "paper": "Frank & Goodman, 2012", "diff": 0.0, "se": 0.0})
    promo = [r["diff"] for r in out if r["kind"] == "seed" and r["model"] in promoted]
    rest = [r for r in out if r["kind"] != "seed"]
    rest.append({"kind": "promoted", "label": "Promoted seeds", "values": promo})
    return rest + sorted(seeds, key=lambda r: -r["diff"])


# --- drawing -----------------------------------------------------------------

BY_RUN = False
RUN_COLORS = ["#6baed6", "#2171b5", "#08306b"]  # three shades of blue, light to dark
WIDTH = {"winner": 1.0, "seed": 1.0, "nonwinning": 2.6, "promoted": 1.8}
STRIP_LABEL = {"nonwinning": "Other auto-psych models", "promoted": "Promoted seeds"}


def columns(rows):
    """Left to right: literature seeds (worst first), promoted seeds, the other auto-psych
    models, the best discovered models (worst first)."""
    order = {"seed": 0, "promoted": 1, "nonwinning": 2, "winner": 3}
    return sorted(rows, key=lambda r: (order[r["kind"]], r.get("diff", 0)))


def seed_label(r, cols):
    """A seed by its paper ("Hahn & Warren (2009)"), by its own name where the paper is shared
    (Frank & Goodman's three RSA models) or there is none."""
    papers = [c.get("paper") for c in cols if c["kind"] == "seed"]
    if r["paper"] is None or papers.count(r["paper"]) > 1:
        return r["label"]
    return r["paper"].replace(", ", " (") + ")"


def draw_panel(ax, domain, rows):
    rng = np.random.default_rng(0)
    cols = columns(rows)
    widths = np.asarray([WIDTH[c["kind"]] for c in cols])
    xs = np.cumsum(widths) - widths / 2
    ax.axhline(0, color=COLOR_SEED, lw=0.8, ls=(0, (3, 2)), zorder=0)
    promoted = [c for c in cols if c["kind"] == "promoted"]
    if promoted:  # the best promoted seed: the pre-registered bar
        ax.axhline(max(promoted[0]["values"]), color=COLOR_PROMOTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
    labels = []
    for x, w, c in zip(xs, widths, cols):
        if c["kind"] in STRIP_LABEL:
            vals = np.asarray(c["values"], float)
            if c["kind"] == "promoted":
                ax.scatter(x + rng.uniform(-0.35, 0.35, len(vals)) * w, vals, s=4, color=COLOR_PROMOTED,
                           alpha=0.7, linewidths=0, zorder=2)
            else:  # left to right in the order each run admitted them
                at = x + (np.asarray(c["progress"]) - 0.5) * 0.85 * w
                runs = np.asarray(c["runs"])
                if not BY_RUN:
                    ax.scatter(at, vals, s=4, c=COLOR_OTHER, alpha=0.6, linewidths=0, zorder=2)
                for k in sorted(set(runs)) if BY_RUN else []:
                    order = np.argsort(at[runs == k])
                    xk, yk = at[runs == k][order], vals[runs == k][order]
                    ax.scatter(xk, yk, s=3, color=RUN_COLORS[k], alpha=0.3, linewidths=0, zorder=2)
                    ax.step(xk, np.maximum.accumulate(yk), where="post", color=RUN_COLORS[k], lw=1.1,
                            zorder=3)  # the best model so far
            labels.append(STRIP_LABEL[c["kind"]])
            continue
        color = (RUN_COLORS[c["run"]] if BY_RUN else COLOR_DISCOVERED) if c["kind"] == "winner" else COLOR_SEED
        if c["se"]:
            ax.plot([x, x], [c["diff"] - 2 * c["se"], c["diff"] + 2 * c["se"]], color=color, lw=1.3,
                    solid_capstyle="butt", zorder=3)
        ax.scatter([x], [c["diff"]], s=14, color=color, zorder=4, linewidths=0)
        labels.append(c["label"] if c["kind"] == "winner" else seed_label(c, cols))
    ax.set_xticks(xs, labels, rotation=40, ha="right", rotation_mode="anchor", fontsize=6)
    for tick, c in zip(ax.get_xticklabels(), cols):
        win = (RUN_COLORS[c["run"]] if BY_RUN else COLOR_DISCOVERED) if c["kind"] == "winner" else None
        tick.set_color(INK if c["kind"] == "winner" else {"seed": COLOR_SEED}.get(c["kind"], INK_2))
    ax.tick_params(axis="x", length=0, pad=3)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, xs[-1] + widths[-1] / 2)
    ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+,.0f}".replace("-", "−") if v else "0"))
    ax.set_title(TITLES[domain], loc="left", fontdict=TITLE_FONT, pad=6)


def main():
    setup_style()
    data, names, papers = load()
    rows = {d: forest_rows(d, data[d], names[d], papers[d]) for d in TITLES}
    rows["rsa"] = rsa_vs_textbook(rows["rsa"], data["rsa"])
    widths = [sum(WIDTH[r["kind"]] for r in rows[d]) for d in TITLES]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.3, "width_ratios": widths})
    for ax, d in zip(axes, TITLES):
        draw_panel(ax, d, rows[d])
    axes[0].set_ylabel("ΔELPD vs. best literature model (nats)")
    suffix = "_by_run" if BY_RUN else ""
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_draft{suffix}_blacklabels.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    import sys
    BY_RUN = "--by-run" in sys.argv
    main()
