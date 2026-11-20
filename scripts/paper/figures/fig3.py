"""Figure 3 (``fig3_effects``), one column per domain, from the figure data (``<domain>.json``'s ``conditions``:
each condition's observed proportion and the pooled best model's and best seed's predictions):
(a) observed against each model's prediction, with trial-weighted R² and RMSE; (b) the domain's classic effect,
drawn as its original paper drew it, which both models capture; (c) a new effect only the winner captures.
Reference games use the pre-registered experiment's displays (``data/rsa/viz/panel_displays.json`` says what
each showed). Panels whose data are not in yet are drawn as pending."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.paper.figures.icons import STIMULI
from scripts.paper.figures.style import INK, INK_2, TITLE_FONT, TITLES, save

SEED, BEST, HUMAN = "#b3b3b3", "#2171b5", INK  # best seed model, best winner, people
WIDTH_IN, HEIGHT_IN = 7.0, 5.4
ORDER = ["subjective_randomness", "rsa", "number_game"]
ROWS = ("(a)", "(b)", "(c)")
ROW_TITLES = (None, "Classic effect captured by both models", "New effect captured by our model")
MODELS = ("best_seed", "best_model")
SR_LENGTH = 8  # the commonest sequence length: alternation counts compare only within a length
PCT = plt.FuncFormatter(lambda v, _: f"{v:.0%}")


def load(figure_data: Path, displays: Path) -> Dict[str, pd.DataFrame]:
    """Each domain's conditions; reference games' only from the pre-registered experiment, with what each
    display showed."""
    out = {d: pd.DataFrame(json.loads((figure_data / f"{d}.json").read_text())["conditions"]) for d in ORDER}
    shown = pd.DataFrame(json.loads(displays.read_text()))
    rsa = out["rsa"]
    if len(shown) != rsa.display.max() + 1 or (shown.n.to_numpy()[rsa.display] != rsa.n).any():
        raise RuntimeError(f"{displays} does not describe the figure data's reference-game displays")
    out["rsa"] = rsa[rsa.dataset == "prereg"].join(shown[["objects", "utterance", "choice_classes"]], on="display")
    return out


def r2(o, p, n) -> float:
    """Trial-weighted R² of the predictions: 1 - residual / total variance of the observed proportions."""
    w = n / n.sum()
    return float(1 - (w * (o - p) ** 2).sum() / (w * (o - (w * o).sum()) ** 2).sum())


def _pending(ax, what):
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"{what}: pending", transform=ax.transAxes, ha="center", va="center", color=INK_2,
            style="italic")


def _scatter(ax, cond, col, color, label):
    ax.plot([0, 1], [0, 1], color="#e0e0e0", lw=0.8, zorder=0)
    ax.scatter(cond[col], cond.observed, s=4 + 12 * np.sqrt(cond.n / cond.n.max()), color=color, alpha=0.25,
               linewidths=0, clip_on=False, zorder=1)
    rmse = np.sqrt(np.average((cond.observed - cond[col]) ** 2, weights=cond.n))
    ax.text(0.04, 0.97, f"$R^2$ = {r2(cond.observed, cond[col], cond.n):.2f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6, zorder=5,
            bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1})
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlabel(label, labelpad=1)


def alternation_curve(sr: pd.DataFrame) -> pd.DataFrame:
    """Each length-8 sequence's chance of being chosen as more random (observed, each model's), pooled over
    every pair it was in, by its number of alternations; people's binomial 95% CI."""
    a = pd.DataFrame({"seq": sr.pair.str[0], "n": sr.n, **{c: sr[c] for c in ("observed", *MODELS)}})
    b = pd.DataFrame({"seq": sr.pair.str[1], "n": sr.n, **{c: 1 - sr[c] for c in ("observed", *MODELS)}})
    both = pd.concat([a, b])
    both = both[both.seq.str.len() == SR_LENGTH]
    both["alt"] = both.seq.map(lambda q: sum(x != y for x, y in zip(q, q[1:])))
    g = both.groupby("alt")
    t = pd.DataFrame({c: g.apply(lambda d: np.average(d[c], weights=d.n), include_groups=False)
                      for c in ("observed", *MODELS)})
    t["ci"] = 1.96 * np.sqrt(t.observed * (1 - t.observed) / g.n.sum())
    return t


def classic_sr(ax, sr):
    """Falk & Konold (1997): how random a sequence looks against its number of alternations, peaking above a
    fair coin's average."""
    t = alternation_curve(sr)
    fair = (SR_LENGTH - 1) / 2
    ax.axvline(fair, color="#b0b0b0", lw=0.6, ls=(0, (2, 2)), zorder=0)
    for c, color in zip(MODELS, (SEED, BEST)):
        ax.plot(t.index, t[c], color=color, lw=1.3, zorder=2)
    ax.errorbar(t.index, t.observed, yerr=t.ci, fmt="o", color=HUMAN, ms=3, elinewidth=0.8, zorder=3)
    ax.text(fair + 0.1, 0.03, "fair coin", transform=ax.get_xaxis_transform(), fontsize=6, color=INK_2)
    ax.set(xlim=(-0.3, SR_LENGTH - 0.7), ylim=(0, 0.8), xticks=range(SR_LENGTH))
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel(f"alternations in {SR_LENGTH} flips", labelpad=1)
    ax.set_ylabel("chosen as more random", labelpad=1)


GAME = [[0, 1], [1, 0], [1, 1]]  # Frank & Goodman (2012)'s game: the word (feature 0) fits two of three objects


def reference_game(rsa: pd.DataFrame) -> pd.DataFrame:
    """The pre-registered display with Frank & Goodman's structure, one row per object (its class's share)."""
    d = rsa[rsa.objects.map(lambda o: o == GAME) & (rsa.utterance == 0)]
    if d.display.nunique() != 1:
        raise RuntimeError(f"{d.display.nunique()} pre-registered displays have Frank & Goodman's structure, not 1")
    return d.sort_values("cls").set_index("cls")


def classic_rsa(ax, rsa):
    """Frank & Goodman (2012): one display, the share choosing each object. People pick the object with only the
    word's feature: a speaker would have named the other object's second feature. The pictures and words varied
    across participants; the display's structure did not."""
    d = reference_game(rsa)
    x = np.arange(len(d))
    w = 0.26
    o = d.observed.to_numpy()
    ax.bar(x - w, o, w, color=HUMAN, zorder=2)
    ax.errorbar(x - w, o, yerr=1.96 * np.sqrt(o * (1 - o) / d.n), fmt="none", ecolor=INK_2, elinewidth=0.7, zorder=3)
    for dx, c, color in ((0, MODELS[0], SEED), (w, MODELS[1], BEST)):
        ax.bar(x + dx, d[c], w, color=color, zorder=2)
    labels = []
    for cls, classes in zip(d.index, d.choice_classes):  # each class by its object's features
        has = GAME[classes.index(cls)]
        labels.append("both" if all(has) else "only the\nword's feature" if has[0] else "only the\nother feature")
    ax.set_xticks(x, labels, fontsize=6, va="top")
    ax.tick_params(axis="x", length=0)
    ax.set(xlim=(-0.6, len(x) - 0.4), ylim=(0, 1))
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("chosen", labelpad=1)


def draw(figure_data: Path, displays: Path, out_dir: Path) -> None:
    tables = load(figure_data, displays)
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    # one grid for every row, so a domain's panels share a column: row (a)'s two scatterplots side by side,
    # rows (b) and (c) spanning both; a spacer column between domains
    gs = fig.add_gridspec(3, 8, width_ratios=[1, 1, 0.32, 1, 1, 0.32, 1, 1], height_ratios=[1.2, 1, 1],
                          wspace=0.18, hspace=0.68, left=0.075, right=0.99, top=0.9, bottom=0.11)
    tops = []
    for k, d in enumerate(ORDER):
        a = [fig.add_subplot(gs[0, 3 * k + j]) for j in (0, 1)]
        _scatter(a[0], tables[d], MODELS[0], SEED, "best seed")
        _scatter(a[1], tables[d], MODELS[1], BEST, "best winner")
        for ax in a if k else a[1:]:
            ax.tick_params(labelleft=False)
        tops.append(a[0])
        b = fig.add_subplot(gs[1, 3 * k:3 * k + 2])
        {"subjective_randomness": classic_sr, "rsa": classic_rsa,
         "number_game": lambda ax, _: _pending(ax, "generalization profiles")}[d](b, tables[d])
        _pending(fig.add_subplot(gs[2, 3 * k:3 * k + 2]), "new effect")
    tops[0].set_ylabel("observed", labelpad=1)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    title_y = tops[0].get_position().y1 + 0.035
    for k, d in enumerate(ORDER):  # each domain's icon and name, together centred over its pair of scatterplots
        centre = (gs[0, 3 * k].get_position(fig).x0 + gs[0, 3 * k + 1].get_position(fig).x1) / 2
        t = fig.text(0, title_y, TITLES[d], fontdict=TITLE_FONT, va="center", ha="left")
        box = t.get_window_extent(r)
        (x0, y0), (x1, y1) = to_fig.transform((box.x0, box.y0)), to_fig.transform((box.x1, box.y1))
        draw_icon, aspect = STIMULI[d]
        h = 1.9 * (y1 - y0)
        w = h * aspect * fig.get_figheight() / fig.get_figwidth()
        gap = 0.008
        left = centre - (w + gap + x1 - x0) / 2
        draw_icon(fig.add_axes((left, title_y - h / 2, w, h)))
        t.set_x(left + w + gap)
    for row, label in enumerate(ROWS):
        y = title_y if row == 0 else gs[row, 0].get_position(fig).y1 + 0.035  # clear of the y axis titles
        t = fig.text(0.005, y, label, fontdict=TITLE_FONT, fontsize=10, va="center")
        if ROW_TITLES[row]:  # just right of the letter
            fig.canvas.draw()
            x1 = to_fig.transform((t.get_window_extent().x1, 0))[0]
            fig.text(x1 + 0.008, y, ROW_TITLES[row], fontdict=TITLE_FONT, fontsize=8.5, va="center")
    key = [Line2D([], [], color=HUMAN, marker="o", ms=3, lw=0, label="people (95% CI)"),
           Line2D([], [], color=SEED, lw=1.5, label="best seed model"),
           Line2D([], [], color=BEST, lw=1.5, label="best winning model")]
    fig.legend(handles=key, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, out_dir, "fig3_effects")
