"""Figure 2 (``fig2_elpd``): per domain, every model's ΔELPD against the best seed model (the runs
collapsed; ``data.forest_rows``), models across, left to right: the literature seeds (worst first), the
promoted seeds (reference games), the other models the loop wrote (one strip, each run's in the order it
admitted them, under a line tracing the loop's own choice, its incumbent, as it went) and the best
discovered models (worst first). The runs are shaded by their best model, the best darkest. Named points
carry ±2 SE of the paired difference. Reference games are re-zeroed on the best textbook model, with the
best promoted seed (the pre-registered bar) as a second line. Names sit in two bands, the winners' above
and the seeds' below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from scripts.paper.figures.data import POOLED_VIEW, forest_rows
from scripts.paper.figures.icons import add_title_icons
from scripts.paper.figures.style import INK, INK_2, TITLE_FONT, panel_title, save, setup_style

COLOR_SEED, COLOR_PROMOTED = "#b3b3b3", "#3d3d3d"
RUN_COLORS = ["#08306b", "#2171b5", "#6baed6"]  # a panel's runs by their best model, best (darkest) first
SEED_LABEL = "#7a7a7a"  # the best seed line's label: darker than the line, to read
WIDTH = {"winner": 1.0, "seed": 1.0, "nonwinning": 2.6, "promoted": 1.8}  # each column's width
PENDING_WIDTH = 9.0  # a pending panel's
STRIP_LABEL = {"nonwinning": "search history", "promoted": "promoted\nseeds"}
NAME_PT = 4.5  # the models' names
CAPTION_PT = 5.0  # the strips' captions and the lines' labels
LINE_PT = 5.8  # one line of a name band


def rsa_vs_textbook(rows: List[Dict[str, Any]], data: Dict[str, Any], names: Dict[str, str],
                    papers: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
    """Re-zero reference games on the best textbook seed (``best_textbook``): every value shifted. The
    winners keep the SE of the pre-registered test (paired with the best promoted seed, whose line the
    figure draws); the seeds' SEs are dropped, the textbook model's own row added and the promoted seeds
    collected into one strip."""
    textbook = data["best_textbook"]
    shift = -textbook["diff"]
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
    seeds.append({"kind": "seed", "model": textbook["model"], "label": names[textbook["model"]],
                  "paper": papers[textbook["model"]], "diff": 0.0, "se": 0.0})
    rest = [r for r in out if r["kind"] != "seed"]
    promo = [r for r in out if r["kind"] == "seed" and r["model"] in promoted]
    rest.append({"kind": "promoted", "label": "Promoted seeds", "values": [r["diff"] for r in promo],
                 "models": [r["model"] for r in promo]})
    return rest + sorted(seeds, key=lambda r: -r["diff"])


def load_history(template: str, run: str) -> List[Dict[str, Any]]:
    """A run's steps, experiment by experiment (each experiment's ``model_loop/history.json``)."""
    root = Path(template.format(run=run))
    return [st for exp in (1, 2, 3)
            for st in json.loads((root / f"experiment{exp}/model_loop/history.json").read_text())]


def incumbent_paths(domain: str, rows: List[Dict[str, Any]], histories: List[List[Dict[str, Any]]]) -> List[list]:
    """Per run (``histories`` in the domain's run order): the loop's own choice over time, as (progress
    through the run's strip, the chosen model's value) at every step. The choice is each step's
    ``best_model``; progress is the share of the run's strip models the loop had admitted by then (those
    standing at the step or any before). A choice with no value in the figure is said and skipped."""
    shared = {r["model"]: r["diff"] for r in rows if r["kind"] == "seed"}
    for r in rows:
        if r["kind"] == "promoted":
            shared.update(zip(r["models"], r["values"]))
    (strip,) = [r for r in rows if r["kind"] == "nonwinning"]
    paths = []
    for k, steps in enumerate(histories):
        mine = [(m, v) for m, v, run in zip(strip["models"], strip["values"], strip["runs"]) if run == k]
        ranked = [m for m, _ in mine]
        value = {**shared, **dict(mine), **{r["model"]: r["diff"] for r in rows if r["kind"] == "winner" and r["run"] == k}}
        seen, path, missing = set(), [], set()
        for st in steps:
            seen |= set(st.get("standing") or st.get("elpd_loo") or {})
            if st["best_model"] not in value:
                missing.add(st["best_model"])
                continue
            n_in = sum(m in seen for m in ranked)
            path.append(((max(n_in, 1) - 1) / (len(ranked) - 1) if len(ranked) > 1 else 0.5, value[st["best_model"]]))
        if missing:
            print(f"[{domain}] run {k + 1}'s incumbents with no value in the figure, skipped: {sorted(missing)}",
                  flush=True)
        paths.append(path)
    return paths


def columns(rows):
    """Left to right: literature seeds (worst first), promoted seeds, the other auto-psych models, the
    best discovered models (worst first)."""
    order = {"seed": 0, "promoted": 1, "nonwinning": 2, "winner": 3}
    return sorted(rows, key=lambda r: (order[r["kind"]], r.get("diff", 0)))


def seed_names(cols):
    """The seeds' names, each with its models: a paper's models under one name ("Hahn & Warren (2009)"),
    a seed no paper proposed by its own."""
    names: Dict[str, List[str]] = {}
    for c in cols:
        if c["kind"] == "seed":
            names.setdefault(c["label"] if c["paper"] is None else c["paper"].replace(", ", " (") + ")",
                             []).append(c["model"])
    return list(names.items())


def place_bands(fig, bands, captions):
    """Name the models in two bands, one line each, a thin leader to each point: the best discovered
    models at the top (the best on the top line, each name ending at its column), the seeds at the bottom
    (left to right from the bottom line, each name starting at its leftmost model's column, a leader to
    each of its models). In that order no leader crosses a name. Each panel's y range grows to fit its
    bands, and its strips' captions (``captions``: those hung below, those set above) between them."""
    fig.canvas.draw()
    for ax, (top, bottom) in bands.items():
        data_lo, data_hi = ax.get_ylim()
        ticks = MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]).tick_values(data_lo, data_hi)
        ax.set_yticks([v for v in ticks if data_lo <= v <= data_hi])
        height_pt = ax.bbox.height * 72 / fig.dpi
        t_pt, b_pt = len(top) * LINE_PT + 4, len(bottom) * LINE_PT + 4
        mid_pt = height_pt - t_pt - b_pt
        lo, hi = data_lo, data_hi
        below, above = captions[ax]
        for caption in below:  # keep each caption hung below its strip above the bottom band
            r = (caption.get_window_extent().height * 72 / fig.dpi + 6) / mid_pt
            lo = min(lo, (caption.xy[1] - r * hi) / (1 - r))
        for caption in above:  # and each set above its strip below the top one
            r = (caption.get_window_extent().height * 72 / fig.dpi + 6) / mid_pt
            hi = max(hi, (caption.xy[1] - r * lo) / (1 - r))
        span = (hi - lo) / mid_pt * height_pt
        lo, hi = lo - b_pt / height_pt * span, hi + t_pt / height_pt * span
        ax.set_ylim(lo, hi)
        ax.spines["left"].set_bounds(data_lo, data_hi)  # the axis spans the data, not the bands
        per_pt = (hi - lo) / height_pt
        renderer, inv = fig.canvas.get_renderer(), ax.transData.inverted()
        right = inv.transform((ax.bbox.x1, 0))[0]
        leader = {"arrowstyle": "-", "color": "#9a9a9a", "lw": 0.5, "shrinkA": 1, "shrinkB": 3}
        for k, (text, x, y) in enumerate(sorted(top, key=lambda n: -n[1])):
            ax.annotate(text, (x, y), xytext=(x, hi - (k + 0.5) * LINE_PT * per_pt), ha="right", va="center",
                        fontsize=NAME_PT, color=INK, arrowprops={**leader, "relpos": (1, 0.5)},
                        annotation_clip=False)
        for k, (text, points) in enumerate(sorted(bottom, key=lambda n: min(n[1]))):
            yt = lo + (k + 0.5) * LINE_PT * per_pt
            (x, y), *rest = sorted(points)
            name = ax.annotate(text, (x, y), xytext=(x, yt), ha="left", va="center", fontsize=NAME_PT,
                               color=INK_2, annotation_clip=False, arrowprops={**leader, "relpos": (0, 0.5)})
            for xr, yr in rest:  # the name's other models: a leader from its top
                ax.annotate("", (xr, yr), xytext=(xr, yt + 0.5 * LINE_PT * per_pt), annotation_clip=False,
                            arrowprops={**leader, "shrinkA": 0})
            box = name.get_window_extent(renderer)
            if box.x1 > ax.bbox.x1:  # past the panel's right edge: end the name there
                shift = inv.transform((box.width, 0))[0] - inv.transform((0, 0))[0]
                name.set_position((right - shift, yt))
        ax.set_ylim(lo, hi)


def draw_panel(ax, domain, rows, paths):
    """One domain's columns, each run's incumbent ``paths`` over its strip; returns its name bands (top,
    bottom) and its strips' captions."""
    rng = np.random.default_rng(0)
    # Each run's shade is its rank by its best model: the best run darkest.
    rank = {r["run"]: k for k, r in enumerate(sorted((r for r in rows if r["kind"] == "winner"),
                                                     key=lambda r: -r["diff"]))}
    cols = columns(rows)
    widths = np.asarray([WIDTH[c["kind"]] for c in cols])
    xs = np.cumsum(widths) - widths / 2
    ax.axhline(0, color=COLOR_SEED, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.annotate("best seed model", (1, 0), xycoords=("axes fraction", "data"), xytext=(0, -1.5),
                textcoords="offset points", ha="right", va="top", fontsize=CAPTION_PT, color=SEED_LABEL)
    promoted = [c for c in cols if c["kind"] == "promoted"]
    if promoted:  # the best promoted seed: the pre-registered bar
        bar = max(promoted[0]["values"])
        ax.axhline(bar, color=COLOR_PROMOTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
        ax.annotate("best promoted seed", (0, bar), xycoords=("axes fraction", "data"), xytext=(2, 1.5),
                    textcoords="offset points", ha="left", va="bottom", fontsize=CAPTION_PT, color=COLOR_PROMOTED)
    top, low, below, above = [], {}, [], []
    for x, w, c in zip(xs, widths, cols):
        if c["kind"] in STRIP_LABEL:
            vals = np.asarray(c["values"], float)
            if c["kind"] == "promoted":
                ax.scatter(x + rng.uniform(-0.35, 0.35, len(vals)) * w, vals, s=4, color=COLOR_PROMOTED,
                           alpha=0.7, linewidths=0, zorder=2)
            else:  # left to right in the order each run admitted them, under its incumbent's path
                at, runs = x + (np.asarray(c["progress"]) - 0.5) * 0.85 * w, np.asarray(c["runs"])
                for run in sorted(range(len(paths)), key=rank.get):  # the best run first
                    path = paths[run]
                    color, layer = RUN_COLORS[rank[run]], 3 - rank[run]  # the best run on top
                    ax.scatter(at[runs == run], vals[runs == run], s=3, color=color, alpha=0.3, linewidths=0,
                               zorder=2)
                    px, py = zip(*path)
                    ax.step(x + (np.asarray(px) - 0.5) * 0.85 * w, py, where="post", color=color, lw=1.1,
                            zorder=2.5 + 0.1 * layer)
            if c["kind"] == "nonwinning":  # above the search history and its paths
                above.append(ax.annotate(STRIP_LABEL[c["kind"]], (x, vals.max()), xytext=(0, 4),
                                         textcoords="offset points", ha="center", va="bottom", fontsize=CAPTION_PT,
                                         color=INK_2, style="italic"))
            else:
                below.append(ax.annotate(STRIP_LABEL[c["kind"]], (x, vals.min()), xytext=(0, -4),
                                         textcoords="offset points", ha="center", va="top", fontsize=CAPTION_PT,
                                         color=INK_2, style="italic", linespacing=1.0))
            continue
        color = RUN_COLORS[rank[c["run"]]] if c["kind"] == "winner" else COLOR_SEED
        half = 2 * (c["se"] or 0.0)
        if c["se"]:
            ax.plot([x, x], [c["diff"] - half, c["diff"] + half], color=color, lw=1.3,
                    solid_capstyle="butt", zorder=3)
        ax.scatter([x], [c["diff"]], s=14, color=color, zorder=4, linewidths=0)
        if c["kind"] == "winner":
            top.append((c["label"], x, c["diff"] + half))
        else:
            low[c["model"]] = (x, c["diff"] - half)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, xs[-1] + widths[-1] / 2)
    ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+,.0f}".replace("-", "−") if v else "0"))
    bottom = [(text, [low[m] for m in models]) for text, models in seed_names(cols)]
    return (top, bottom), (below, above)


def draw(domain_data: Dict[str, Optional[Dict[str, Any]]], out_dir: Path, mock_domains: List[str],
         names: Dict[str, Dict[str, str]], papers: Dict[str, Dict[str, Optional[str]]],
         histories: Dict[str, str]) -> None:
    """``domain_data``: each domain's figure data (the number game's through
    ``data.number_game_unseen_view``), ``None`` for a pending domain; ``histories``: each domain's run
    directory, ``{run}`` for the run."""
    rows = {d: forest_rows(d, data, names[d], papers[d]) for d, data in domain_data.items() if data is not None}
    if "rsa" in rows:
        rows["rsa"] = rsa_vs_textbook(rows["rsa"], domain_data["rsa"], names["rsa"], papers["rsa"])
    paths = {d: incumbent_paths(d, rows[d], [load_history(histories[d], run) for run in (
        domain_data[d]["pooled"]["runs"] if d in POOLED_VIEW else [r["run"] for r in domain_data[d]["runs"]])])
        for d in rows}
    with plt.rc_context():
        setup_style()
        widths = [sum(WIDTH[r["kind"]] for r in rows[d]) if d in rows else PENDING_WIDTH for d in domain_data]
        fig, axes = plt.subplots(1, len(domain_data), figsize=(7.2, 3.0),
                                 gridspec_kw={"wspace": 0.3, "width_ratios": widths})
        axes = np.atleast_1d(axes)
        bands, captions = {}, {}
        for ax, d in zip(axes, domain_data):
            ax.set_title(panel_title(d, mock_domains=mock_domains), loc="center", fontdict=TITLE_FONT, pad=6)
            if d not in rows:
                ax.set_xticks([]); ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                ax.text(0.5, 0.5, "results pending", transform=ax.transAxes, ha="center", va="center",
                        fontsize=CAPTION_PT, color=INK_2, style="italic")
                continue
            bands[ax], captions[ax] = draw_panel(ax, d, rows[d], paths[d])
        axes[0].set_ylabel("ΔELPD vs. best seed model (nats)")
        place_bands(fig, bands, captions)
        # the runs' key: the leftmost panel's upper left, just below its band of names
        key = [Line2D([], [], color=c, lw=1.1, label=f"{name} run")
               for name, c in zip(["best", "second", "third"], RUN_COLORS)]
        axes[0].legend(handles=key, loc="upper left", bbox_to_anchor=(0, axes[0].spines["left"].get_bounds()[1]),
                       bbox_transform=axes[0].get_yaxis_transform(), frameon=False, fontsize=CAPTION_PT,
                       handlelength=1.4, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.4)
        add_title_icons(fig, axes, list(domain_data))
        save(fig, out_dir, "fig2_elpd")
