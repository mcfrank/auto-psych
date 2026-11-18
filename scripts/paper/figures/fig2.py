"""Figure 2 (``fig2_elpd``): per domain, every model's ΔELPD against the best seed model (the runs
collapsed; ``data.forest_rows``), models across, left to right: the literature seeds (worst first), the
promoted seeds (reference games), the other models the loop wrote (one strip, each run's in the order it
admitted them) and the best discovered models (worst first), each named point with ±2 SE of the paired
difference. Reference games are re-zeroed on the best textbook model, with the best promoted seed (the
pre-registered bar) as a second line. Names sit in two bands, the winners' above and the seeds' below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from scripts.paper.figures.data import forest_rows
from scripts.paper.figures.icons import add_title_icons
from scripts.paper.figures.style import INK, INK_2, TITLE_FONT, panel_title, save, setup_style

COLOR_DISCOVERED, COLOR_SEED, COLOR_PROMOTED, COLOR_OTHER = "#4C72B0", "#b3b3b3", "#3d3d3d", "#b3b3b3"
SEED_LABEL = "#7a7a7a"  # the best seed line's label: darker than the line, to read
WIDTH = {"winner": 1.0, "seed": 1.0, "nonwinning": 2.6, "promoted": 1.8}  # each column's width
PENDING_WIDTH = 9.0  # a pending panel's
STRIP_LABEL = {"nonwinning": "search history", "promoted": "promoted\nseeds"}
CAPTION_ABOVE = {"number_game"}  # its strip reaches the zero line, its winners cross it: captions moved
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
    rest.append({"kind": "promoted", "label": "Promoted seeds",
                 "values": [r["diff"] for r in out if r["kind"] == "seed" and r["model"] in promoted]})
    return rest + sorted(seeds, key=lambda r: -r["diff"])


def columns(rows):
    """Left to right: literature seeds (worst first), promoted seeds, the other auto-psych models, the
    best discovered models (worst first)."""
    order = {"seed": 0, "promoted": 1, "nonwinning": 2, "winner": 3}
    return sorted(rows, key=lambda r: (order[r["kind"]], r.get("diff", 0)))


def seed_label(r, cols):
    """A seed by its paper ("Hahn & Warren (2009)"), by its own name where another seed shares the paper
    (Frank & Goodman's RSA models) or there is none."""
    papers = [c.get("paper") for c in cols if c["kind"] == "seed"]
    if r["paper"] is None or papers.count(r["paper"]) > 1:
        return r["label"]
    return r["paper"].replace(", ", " (") + ")"


def place_bands(fig, bands, captions):
    """Name the models in two bands, one line each, a thin leader to each point: the best discovered
    models at the top (the best on the top line, each name ending at its column), the seeds and the strips
    at the bottom (left to right from the bottom line, each name starting at its column). In that order no
    leader crosses a name. Each panel's y range grows to fit its bands, and its captions above the bottom
    one."""
    fig.canvas.draw()
    for ax, (top, bottom) in bands.items():
        data_lo, data_hi = ax.get_ylim()
        ticks = MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]).tick_values(data_lo, data_hi)
        ax.set_yticks([v for v in ticks if data_lo <= v <= data_hi])
        height_pt = ax.bbox.height * 72 / fig.dpi
        t_pt, b_pt = len(top) * LINE_PT + 4, len(bottom) * LINE_PT + 4
        mid_pt = height_pt - t_pt - b_pt
        lo, hi = data_lo, data_hi
        for caption in captions.get(ax, []):  # keep each caption (hung below its strip) above the band
            r = (caption.get_window_extent().height * 72 / fig.dpi + 6) / mid_pt
            lo = min(lo, (caption.xy[1] - r * hi) / (1 - r))
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
        for k, (text, x, y) in enumerate(sorted(bottom, key=lambda n: n[1])):
            yt = lo + (k + 0.5) * LINE_PT * per_pt
            name = ax.annotate(text, (x, y), xytext=(x, yt), ha="left", va="center", fontsize=NAME_PT,
                               color=INK_2, annotation_clip=False, arrowprops={**leader, "relpos": (0, 0.5)})
            box = name.get_window_extent(renderer)
            if box.x1 > ax.bbox.x1:  # past the panel's right edge: end the name there
                shift = inv.transform((box.width, 0))[0] - inv.transform((0, 0))[0]
                name.set_position((right - shift, yt))
        ax.set_ylim(lo, hi)


def draw_panel(ax, domain, rows):
    """One domain's columns; returns its name bands (top, bottom) and its strips' captions."""
    rng = np.random.default_rng(0)
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
    top, bottom, captions = [], [], []
    for x, w, c in zip(xs, widths, cols):
        if c["kind"] in STRIP_LABEL:
            vals = np.asarray(c["values"], float)
            if c["kind"] == "promoted":
                ax.scatter(x + rng.uniform(-0.35, 0.35, len(vals)) * w, vals, s=4, color=COLOR_PROMOTED,
                           alpha=0.7, linewidths=0, zorder=2)
            else:  # left to right in the order each run admitted them
                ax.scatter(x + (np.asarray(c["progress"]) - 0.5) * 0.85 * w, vals, s=4, c=COLOR_OTHER,
                           alpha=0.6, linewidths=0, zorder=2)
            if domain in CAPTION_ABOVE and c["kind"] == "nonwinning":
                ax.annotate(STRIP_LABEL[c["kind"]], (x, vals.max()), xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=CAPTION_PT, color=INK_2, style="italic")
                continue
            captions.append(ax.annotate(STRIP_LABEL[c["kind"]], (x, vals.min()), xytext=(0, -4),
                                        textcoords="offset points", ha="center", va="top", fontsize=CAPTION_PT,
                                        color=INK_2, style="italic", linespacing=1.0))
            continue
        color = COLOR_DISCOVERED if c["kind"] == "winner" else COLOR_SEED
        half = 2 * (c["se"] or 0.0)
        if c["se"]:
            ax.plot([x, x], [c["diff"] - half, c["diff"] + half], color=color, lw=1.3,
                    solid_capstyle="butt", zorder=3)
        ax.scatter([x], [c["diff"]], s=14, color=color, zorder=4, linewidths=0)
        if c["kind"] == "winner":
            top.append((c["label"], x, c["diff"] + half))
        else:
            bottom.append((seed_label(c, cols), x, c["diff"] - half))
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, xs[-1] + widths[-1] / 2)
    ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+,.0f}".replace("-", "−") if v else "0"))
    return (top, bottom), captions


def draw(domain_data: Dict[str, Optional[Dict[str, Any]]], out_dir: Path, mock_domains: List[str],
         names: Dict[str, Dict[str, str]], papers: Dict[str, Dict[str, Optional[str]]]) -> None:
    """``domain_data``: each domain's figure data (the number game's through
    ``data.number_game_unseen_view``), ``None`` for a pending domain."""
    rows = {d: forest_rows(d, data, names[d], papers[d]) for d, data in domain_data.items() if data is not None}
    if "rsa" in rows:
        rows["rsa"] = rsa_vs_textbook(rows["rsa"], domain_data["rsa"], names["rsa"], papers["rsa"])
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
            bands[ax], captions[ax] = draw_panel(ax, d, rows[d])
        axes[0].set_ylabel("ΔELPD vs. best seed model (nats)")
        place_bands(fig, bands, captions)
        add_title_icons(fig, axes, list(domain_data))
        save(fig, out_dir, "fig2_elpd")
