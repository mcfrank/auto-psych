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

from scripts.paper.figures import NG_TEST_SETS, forest_rows, number_game_unseen_view  # noqa: E402

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
COLOR_DISCOVERED, COLOR_SEED, COLOR_PROMOTED, COLOR_OTHER = BLUE, "#b3b3b3", "#3d3d3d", "#b3b3b3"
SEED_LABEL = "#7a7a7a"  # the best seed line's label: darker than the line, to read
TITLE_FONT = {"family": "Avenir", "weight": "medium"}

TITLES = {"subjective_randomness": "Subjective randomness", "rsa": "Reference games",
          "number_game": "Number game"}


def subtitle(domain, data):
    return {"subjective_randomness": "LOO, all runs pooled · 23,104 trials",
            "rsa": "pre-registered held-out experiment · 14,404 trials",
            "number_game": f"{NG_TEST_SETS[NG_TEST_SET]} · {data['n_responses']:,} responses"}[domain]


# --- data --------------------------------------------------------------------

def load():
    fd = here() / "data" / "paper" / "figure_data"
    data = {d: json.loads((fd / f"{d}.json").read_text()) for d in TITLES}
    # The number game is scored on what no run drew (Figure 2's column; scripts/paper/figures.py).
    unseen = json.loads((here() / "data/paper/number_game_unseen_items.json").read_text())
    data["number_game"] = number_game_unseen_view(data["number_game"], unseen, NG_TEST_SET)
    data["number_game"]["n_responses"] = unseen[NG_TEST_SET]["n_responses"]
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



HISTORY = {"subjective_randomness": "data/results/human_experiment_sr-oct26/collected/{run}/subjective_randomness",
           "number_game": "data/results/number_game_dataset-nov26/collected/{run}/number_game",
           "rsa": "data/rsa/live/live_rsa_{run}/outputs"}


def incumbent_paths(domain, data):
    """Per run (in the figure data's run order): the loop's own choice over time, as (progress
    through the run's strip, the chosen model's test value) at every step of every experiment.
    The choice is each step's ``best_model``; progress is the share of the run's strip models
    the loop had admitted by then."""
    if domain == "rsa":
        shift = -data["best_textbook"]["diff"]
        runs = {r["run"]: r for r in data["runs"]}
        value = {run: {m: v + shift for m, v in {**r["pruned_during_run"], **r["others_at_end"],
                                                  **{k: d["diff"] for k, d in r["vs_best_seed"].items()}}.items()}
                 for run, r in runs.items()}
        order = {run: {**r["pruned_order"], **r["others_order"]} for run, r in runs.items()}
        strip = {run: [*r["pruned_during_run"], *r["others_at_end"]] for run, r in runs.items()}
    else:
        pooled = data["pooled"]
        sep = "__"
        every = {**pooled["pruned_during_run"], **pooled["others_at_end"]}
        orders = {**pooled["pruned_order"], **pooled["others_order"]}
        seeds = {m: d["diff"] for m, d in pooled["vs_best_seed"].items()}
        runs = {run: None for run in pooled["runs"]}
        value = {run: {**seeds, **{l.split(sep, 1)[1]: v for l, v in every.items() if l.startswith(run + sep)}}
                 for run in runs}
        order = {run: {l.split(sep, 1)[1]: o for l, o in orders.items() if l.startswith(run + sep)} for run in runs}
        strip = {run: [l.split(sep, 1)[1] for l in every if l.startswith(run + sep)] for run in runs}
    paths = []
    for run in runs:
        ranked = sorted(strip[run], key=order[run].get)
        n = len(ranked)
        seen, path, missing = set(), [], set()
        for exp in (1, 2, 3):
            steps = json.loads((here() / HISTORY[domain].format(run=run) / f"experiment{exp}/model_loop/history.json").read_text())
            for st in steps:
                seen |= set(st.get("standing") or st.get("elpd_loo") or {})
                k = sum(m in seen for m in ranked)
                best = st["best_model"]
                if best not in value[run]:
                    missing.add(best)
                    continue
                path.append(((max(k, 1) - 1) / (n - 1), value[run][best]))
        if missing:
            print(f"[{domain} {run}] incumbents with no test value: {sorted(missing)}")
        paths.append(path)
    return paths


# --- drawing -----------------------------------------------------------------

BY_RUN = False
NG_TEST_SET = "unseen_items"  # the (set, target) items no run drew; or "unseen_example_sets" (--ng-unseen-sets)
RUN_COLORS = ["#08306b", "#2171b5", "#6baed6"]  # a panel's runs by their best model, best (darkest) first
WIDTH = {"winner": 1.0, "seed": 1.0, "nonwinning": 2.6, "promoted": 1.8}
CAPTION_ABOVE = {"number_game"}  # its strip reaches the zero line, its winners cross it: captions moved
STRIP_LABEL = {"nonwinning": "search history", "promoted": "promoted\nseeds"}


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


CAPTIONS = {}  # per panel: the strips' captions, kept clear of the bottom band
BANDS = {}  # per panel: (top names, bottom names), placed once the layout is final
NAME_PT = 4.5  # the models' names
CAPTION_PT = 5.0  # the strips' captions and the lines' labels
LINE_PT = 5.8  # one line of a name band


def place_bands(fig):
    """Name the models in two bands, one line each, a thin leader to each point: the best
    discovered models at the top (the best on the top line, each name ending at its column),
    the seeds and the strips at the bottom (left to right from the bottom line, each name
    starting at its column). In that order no leader crosses a name."""
    fig.canvas.draw()
    data_lo, data_hi = {}, {}
    for ax, (top, bottom) in BANDS.items():
        data_lo[ax], data_hi[ax] = ax.get_ylim()
        ticks = MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]).tick_values(data_lo[ax], data_hi[ax])
        ax.set_yticks([v for v in ticks if data_lo[ax] <= v <= data_hi[ax]])
        height_pt = ax.bbox.height * 72 / fig.dpi
        t_pt, b_pt = len(top) * LINE_PT + 4, len(bottom) * LINE_PT + 4
        mid_pt = height_pt - t_pt - b_pt
        lo, hi = data_lo[ax], data_hi[ax]
        for caption in CAPTIONS.get(ax, []):  # keep each caption (hung below its strip) above the band
            r = (caption.get_window_extent().height * 72 / fig.dpi + 6) / mid_pt
            lo = min(lo, (caption.xy[1] - r * hi) / (1 - r))
        span = (hi - lo) / mid_pt * height_pt
        lo, hi = lo - b_pt / height_pt * span, hi + t_pt / height_pt * span
        ax.set_ylim(lo, hi)
        ax.spines["left"].set_bounds(data_lo[ax], data_hi[ax])  # the axis spans the data, not the bands
        per_pt = (hi - lo) / height_pt
        renderer, inv = fig.canvas.get_renderer(), ax.transData.inverted()
        right = inv.transform((ax.bbox.x1, 0))[0]
        leader = {"arrowstyle": "-", "color": "#9a9a9a", "lw": 0.5, "shrinkA": 1, "shrinkB": 3}
        for k, (text, x, y) in enumerate(sorted(top, key=lambda n: -n[1])):
            yt = hi - (k + 0.5) * LINE_PT * per_pt
            name = ax.annotate(text, (x, y), xytext=(x, yt), ha="right", va="center", fontsize=NAME_PT,
                               color=INK, arrowprops={**leader, "relpos": (1, 0.5)}, annotation_clip=False)

        for k, (text, x, y, style) in enumerate(sorted(bottom, key=lambda n: n[1])):
            yt = lo + (k + 0.5) * LINE_PT * per_pt
            name = ax.annotate(text, (x, y), xytext=(x, yt), ha="left", va="center", fontsize=NAME_PT,
                               color=INK_2, style=style, annotation_clip=False,
                               arrowprops={**leader, "relpos": (0, 0.5)})
            box = name.get_window_extent(renderer)
            if box.x1 > ax.bbox.x1:  # past the panel's right edge: end the name there
                shift = ax.transData.inverted().transform((box.width, 0))[0] - ax.transData.inverted().transform((0, 0))[0]
                name.set_position((right - shift, yt))
        ax.set_ylim(lo, hi)


def draw_panel(ax, domain, rows):
    rng = np.random.default_rng(0)
    # Each run's shade is its rank by its best model: the best run darkest.
    winners = sorted((r for r in rows if r["kind"] == "winner"), key=lambda r: -r["diff"])
    rank = {r["run"]: k for k, r in enumerate(winners)}
    rows = [{**r, "run": rank[r["run"]]} if r["kind"] == "winner" else
            {**r, "runs": [rank[k] for k in r["runs"]],
             "paths": {rank[k]: path for k, path in enumerate(r["paths"])}} if r["kind"] == "nonwinning" else r
            for r in rows]
    cols = columns(rows)
    widths = np.asarray([WIDTH[c["kind"]] for c in cols])
    xs = np.cumsum(widths) - widths / 2
    ax.axhline(0, color=COLOR_SEED, lw=0.8, ls=(0, (3, 2)), zorder=0)
    if domain in CAPTION_ABOVE:  # the winners' bars cross the line at the right edge: end it at the strip's
        strip_end = next(x + 0.425 * w for x, w, c in zip(xs, widths, cols) if c["kind"] == "nonwinning")
        ax.annotate("best seed\nmodel", (strip_end, 0), xytext=(0, -1.5), textcoords="offset points",
                    ha="right", va="top", fontsize=CAPTION_PT, color=SEED_LABEL, linespacing=1.0)
    else:
        ax.annotate("best seed model", (1, 0), xycoords=("axes fraction", "data"), xytext=(0, -1.5),
                    textcoords="offset points", ha="right", va="top", fontsize=CAPTION_PT, color=SEED_LABEL)
    promoted = [c for c in cols if c["kind"] == "promoted"]
    if promoted:  # the best promoted seed: the pre-registered bar
        bar = max(promoted[0]["values"])
        ax.axhline(bar, color=COLOR_PROMOTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.annotate("best promoted seed", (0, bar), xycoords=("axes fraction", "data"), xytext=(2, 1.5),
                    textcoords="offset points", ha="left", va="bottom", fontsize=CAPTION_PT, color=COLOR_PROMOTED)
    top, bottom = [], []
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
                    layer = 3 - k  # the best run on top
                    ax.scatter(xk, yk, s=3, color=RUN_COLORS[k], alpha=0.3, linewidths=0, zorder=2)
                    px, py = zip(*c["paths"][k])  # the loop's own choice at each step, at its test value
                    ax.step(x + (np.asarray(px) - 0.5) * 0.85 * w, py, where="post", color=RUN_COLORS[k],
                            lw=1.1, zorder=2.5 + 0.1 * layer)
            if domain in CAPTION_ABOVE and c["kind"] == "nonwinning":
                ax.annotate(STRIP_LABEL[c["kind"]], (x, vals.max()), xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=CAPTION_PT, color=INK_2, style="italic")
                continue
            CAPTIONS.setdefault(ax, []).append(ax.annotate(STRIP_LABEL[c["kind"]], (x, vals.min()),
                        xytext=(0, -4), textcoords="offset points", ha="center", va="top",
                        fontsize=CAPTION_PT, color=INK_2, style="italic", linespacing=1.0))
            continue
        color = (RUN_COLORS[c["run"]] if BY_RUN else COLOR_DISCOVERED) if c["kind"] == "winner" else COLOR_SEED
        half = 2 * (c["se"] or 0.0)
        if c["se"]:
            ax.plot([x, x], [c["diff"] - half, c["diff"] + half], color=color, lw=1.3,
                    solid_capstyle="butt", zorder=3)
        ax.scatter([x], [c["diff"]], s=14, color=color, zorder=4, linewidths=0)
        if c["kind"] == "winner":
            top.append((c["label"], x, c["diff"] + half))
        else:
            bottom.append((seed_label(c, cols), x, c["diff"] - half, "normal"))
    ax.set_xticks([])
    BANDS[ax] = (top, bottom)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, xs[-1] + widths[-1] / 2)
    ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+,.0f}".replace("-", "−") if v else "0"))
    ax.set_title(TITLES[domain], loc="center", fontdict=TITLE_FONT, pad=6)


def main():
    setup_style()
    data, names, papers = load()
    rows = {d: forest_rows(d, data[d], names[d], papers[d]) for d in data}
    rows["rsa"] = rsa_vs_textbook(rows["rsa"], data["rsa"])
    for d in data:
        paths = incumbent_paths(d, data[d])
        rows[d] = [{**r, "paths": paths} if r["kind"] == "nonwinning" else r for r in rows[d]]
    widths = [sum(WIDTH[r["kind"]] for r in rows[d]) if d in rows else 9.0 for d in TITLES]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.3, "width_ratios": widths})
    for ax, d in zip(axes, TITLES):
        if d not in rows:  # pending: the title (and its icon) over a placeholder
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            ax.text(0.5, 0.5, "November results pending", transform=ax.transAxes, ha="center", va="center",
                    fontsize=CAPTION_PT, color=INK_2, style="italic")
            ax.set_title(TITLES[d], loc="center", fontdict=TITLE_FONT, pad=6)
            continue
        draw_panel(ax, d, rows[d])
    key = [Line2D([], [], color=c, lw=1.1, label=f"{name} run") for name, c in zip(["best", "second", "third"], RUN_COLORS)]

    axes[0].set_ylabel("ΔELPD vs. best seed model (nats)")
    place_bands(fig)
    # The runs' key: the leftmost panel's upper left, just below its band of names.
    axes[0].legend(handles=key, loc="upper left", bbox_to_anchor=(0, axes[0].spines["left"].get_bounds()[1]),
                   bbox_transform=axes[0].get_yaxis_transform(), frameon=False, fontsize=CAPTION_PT,
                   handlelength=1.4, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.4)
    # each panel's stimulus icon, just left of its centred title
    from data.paper.work.icons.icons import STIMULI
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    for ax, d in zip(axes, TITLES):
        box = ax.title.get_window_extent()
        (x0, y0), (x1, y1) = to_fig.transform((box.x0, box.y0)), to_fig.transform((box.x1, box.y1))
        draw, aspect = STIMULI[d]
        h = 1.9 * (y1 - y0)
        w = h * aspect * fig.get_figheight() / fig.get_figwidth()
        draw(fig.add_axes((x0 - w - 0.008, (y0 + y1) / 2 - h / 2, w, h)))
    suffix = "_by_run" if BY_RUN else ""
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_draft{suffix}.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    import sys
    BY_RUN = "--by-run" in sys.argv
    NG_TEST_SET = "unseen_example_sets" if "--ng-unseen-sets" in sys.argv else NG_TEST_SET
    main()
