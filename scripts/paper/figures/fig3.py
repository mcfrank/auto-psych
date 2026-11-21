"""Figure 3 (``fig3_effects``), one column per domain, from the figure data (``<domain>.json``'s ``conditions``:
each condition's observed proportion and the pooled best model's and best seed's predictions; a column whose
model is not the pooled best, ``COLUMN_MODEL``, takes that model's predictions from ``<domain>_model_conditions.json``):
(a) observed against each model's prediction, with trial-weighted R² and RMSE; (b) the domain's classic effect,
drawn as its original paper drew it, which both models capture; (c) a new effect only the winner captures.
Row (c) is exploratory: its patterns were chosen after the fact (``streak_choices``, ``match_roles``,
``digit_confidence``). Reference games use the pre-registered experiment's displays
(``data/rsa/viz/panel_displays.json`` says what each showed). Panels whose data are not in yet are drawn as
pending."""

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
# A column's model when it is not the pooled best (the figure data's ``best_model``): its name and axis label.
# Its predictions, of the same kind as the figure data's, are in ``<domain>_model_conditions.json``
# (``scripts/paper/number_game_model_conditions.py``). The number game shows run 3's winner, the best on
# unseen items (Fig. 2) and the model Fig. 4 ablates; the pooled CV best is run 2's.
COLUMN_MODEL = {"number_game": ("exception_weighted_rule_rejection", "run 3 winner")}
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
    for d, (model, _) in COLUMN_MODEL.items():
        out[d] = with_model(out[d], json.loads((figure_data / f"{d}_model_conditions.json").read_text()), model)
    return out


def with_model(cond: pd.DataFrame, preds: dict, model: str) -> pd.DataFrame:
    """The conditions with ``best_model`` replaced by ``model``'s predictions (``preds``: its per-condition
    ``p_yes``), which must cover exactly the same conditions."""
    if preds["model"] != model:
        raise RuntimeError(f"the predictions are {preds['model']!r}'s, not {model!r}'s")
    key = ["number_set", "target", "n"]
    p = pd.DataFrame(preds["conditions"])
    both = cond.reset_index().merge(p, on=key, how="outer", suffixes=("", "_p"), indicator=True)
    if (both._merge != "both").any() or not np.allclose(both.observed, both.observed_p):
        raise RuntimeError(f"{model!r}'s predictions are for other conditions than the figure data's")
    both = both.set_index("index").loc[cond.index]
    return cond.assign(best_model=both.p_yes.to_numpy())


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
    """Falk & Konold (1997, Figure 1): how random a sequence looks against its probability of alternation,
    P(A) = alternations / (length - 1), peaking above a fair coin's 0.5. Our measure: each length-8 sequence's
    share of choices as the more random of a pair (79% of trials are length 8; P(A) compares within a length)."""
    t = alternation_curve(sr)
    pa = t.index / (SR_LENGTH - 1)
    ax.axvline(0.5, color="#b0b0b0", lw=0.6, ls=(0, (2, 2)), zorder=0)
    for c, color in zip(MODELS, (SEED, BEST)):
        ax.plot(pa, t[c], color=color, lw=1.3, zorder=2)
    ax.errorbar(pa, t.observed, yerr=t.ci, fmt="o", color=HUMAN, ms=3, elinewidth=0.8, zorder=3)
    ax.text(0.515, 0.03, "fair coin", transform=ax.get_xaxis_transform(), fontsize=6, color=INK_2)
    ax.set(xlim=(-0.04, 1.04), ylim=(0, 0.8), xticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel(f"probability of alternation ({SR_LENGTH} flips)", labelpad=1)
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


# --- row (c): exploratory, patterns chosen after the fact ---------------------------------------------------

def _weighted(g, extra=()) -> pd.DataFrame:
    """Each group's trial-weighted mean of the observed and predicted proportions, its trials and people's
    binomial 95% CI."""
    t = pd.DataFrame({c: g.apply(lambda d: np.average(d[c], weights=d.n), include_groups=False)
                      for c in ("observed", *MODELS, *extra)})
    t["n"] = g.n.sum()
    t["ci"] = 1.96 * np.sqrt(t.observed * (1 - t.observed) / t.n)
    return t


SWITCH_BINS = [0, 0.2, 0.4, 0.6, 0.8, 1]


def streak_choices(sr: pd.DataFrame) -> pd.DataFrame:
    """Pairs in which exactly one sequence (of 5 or more flips) is all heads or all tails: the chance of choosing
    it as more random, by the other sequence's switch rate (switches / (length - 1)), binned; each bin's
    trial-weighted mean switch rate."""
    first, second = sr.pair.str[0], sr.pair.str[1]
    const = [s.map(lambda q: len(set(q)) == 1) for s in (first, second)]
    keep = (const[0] ^ const[1]) & (first.str.len() >= 5)
    d, a = sr[keep], const[0][keep]
    rival = np.where(a, second[keep], first[keep])
    t = pd.DataFrame({c: np.where(a, d[c], 1 - d[c]) for c in ("observed", *MODELS)}, index=d.index)
    t["n"] = d.n
    t["switch_rate"] = [sum(x != y for x, y in zip(q, q[1:])) / (len(q) - 1) for q in rival]
    return _weighted(t.groupby(pd.cut(t.switch_rate, SWITCH_BINS, include_lowest=True), observed=True),
                     ["switch_rate"])


ROLES = ("sparsest", "middle", "richest")


def _role(objects, utterance, classes, cls) -> str:
    """The class's object among those the word fits, by number of features, when they have three or more
    different numbers of features ('other' otherwise)."""
    if pd.isna(utterance):
        return "other"
    fits = [sum(o) for o in objects if o[int(utterance)]]
    obj = objects[classes.index(cls)]
    if len(set(fits)) < 3 or not obj[int(utterance)]:
        return "other"
    return ROLES[0] if sum(obj) == min(fits) else ROLES[2] if sum(obj) == max(fits) else ROLES[1]


def match_roles(rsa: pd.DataFrame) -> pd.DataFrame:
    """Displays whose word fits objects with three or more different numbers of features: the share choosing
    the sparsest, a middle and the richest object it fits, averaged over displays weighting each by its trials."""
    d = rsa.assign(role=[_role(*r) for r in zip(rsa.objects, rsa.utterance, rsa.choice_classes, rsa.cls)])
    d = d[d.role != "other"]
    per = d.groupby(["display", "role"]).agg({"n": "first", **{c: "sum" for c in ("observed", *MODELS)}})
    return _weighted(per.reset_index().groupby("role")).loc[list(ROLES)]


def digit_confidence(ng: pd.DataFrame) -> pd.DataFrame:
    """Example sets of two or more numbers that all end in the same digit: the chance of 'yes' to a target that
    is not an example, by whether it keeps that final digit and by the number of examples."""
    ex = ng.number_set.map(lambda s: [int(v) for v in s.split(",")])
    shared = ex.map(lambda e: len(e) >= 2 and len({v % 10 for v in e}) == 1)
    new = pd.Series([t not in e for t, e in zip(ng.target, ex)], index=ng.index)
    d = ng[shared & new].copy()
    d["keeps"] = [t % 10 == e[0] % 10 for t, e in zip(d.target, ex[d.index])]
    d["examples"] = ex[d.index].map(len)
    return _weighted(d.groupby(["keeps", "examples"]))


def _panel_title(ax, text):
    ax.set_title(text, fontsize=7, color=INK, pad=3)


def _models_and_people(ax, x, t, **people):
    for c, color in zip(MODELS, (SEED, BEST)):
        ax.plot(x, t[c], color=color, lw=1.3, marker="o", ms=1.8, zorder=2)
    ax.errorbar(x, t.observed, yerr=t.ci, fmt="o", color=HUMAN, ms=3, elinewidth=0.8, zorder=3, **people)


def new_sr(ax, sr):
    """Pairs with one all-heads or all-tails sequence: people choose it at about the same rate whatever its
    rival; the best seed chooses it less as the rival alternates more."""
    t = streak_choices(sr)
    _models_and_people(ax, t.switch_rate, t)
    ax.set(xlim=(0, 1), ylim=(0, 0.4), xticks=[0, 0.25, 0.5, 0.75, 1])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("other sequence's switch rate", labelpad=1)
    ax.set_ylabel("all heads/tails chosen", labelpad=1)
    _panel_title(ax, "All heads or all tails: a flat rate")


def new_rsa(ax, rsa):
    """Displays whose word fits objects with three or more numbers of features: people rarely choose one in
    the middle."""
    t = match_roles(rsa)
    x = np.arange(len(t))
    w = 0.26
    ax.bar(x - w, t.observed, w, color=HUMAN, zorder=2)
    ax.errorbar(x - w, t.observed, yerr=t.ci, fmt="none", ecolor=INK_2, elinewidth=0.7, zorder=3)
    for dx, c, color in ((0, MODELS[0], SEED), (w, MODELS[1], BEST)):
        ax.bar(x + dx, t[c], w, color=color, zorder=2)
    ax.set_xticks(x, ["fewest\nfeatures", "in\nbetween", "most\nfeatures"], fontsize=6, va="top")
    ax.tick_params(axis="x", length=0)
    ax.set(xlim=(-0.6, len(x) - 0.4), ylim=(0, 0.8))
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("chosen", labelpad=1)
    ax.set_xlabel("object the word fits", labelpad=1)
    _panel_title(ax, "The middle candidate is passed over")


def new_ng(ax, ng):
    """Example sets sharing a final digit: people's yes to targets that keep it grows slowly with more examples;
    the best seed becomes nearly certain."""
    t = digit_confidence(ng)
    for keeps, label in ((True, "keeps the\nfinal digit"), (False, "breaks it")):
        d = t.loc[keeps]
        _models_and_people(ax, d.index, d)
        ax.text(d.index[-1] + 0.15, d.observed.iloc[-1], label, fontsize=6, color=INK_2, va="center")
    ax.set(xlim=(1.8, 4.9), ylim=(0, 1.02), xticks=[2, 3, 4])
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("examples sharing a final digit", labelpad=1)
    ax.set_ylabel("yes to a new number", labelpad=1)
    _panel_title(ax, "Graded confidence in a shared digit")


def draw(figure_data: Path, displays: Path, out_dir: Path) -> None:
    tables = load(figure_data, displays)
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    # one grid for every row, so a domain's panels share a column: row (a)'s two scatterplots side by side,
    # rows (b) and (c) spanning both; a spacer column between domains
    gs = fig.add_gridspec(3, 8, width_ratios=[1, 1, 0.32, 1, 1, 0.32, 1, 1], height_ratios=[1.2, 1, 1],
                          wspace=0.18, hspace=0.78, left=0.075, right=0.99, top=0.9, bottom=0.11)
    tops = []
    for k, d in enumerate(ORDER):
        a = [fig.add_subplot(gs[0, 3 * k + j]) for j in (0, 1)]
        _scatter(a[0], tables[d], MODELS[0], SEED, "best seed")
        _scatter(a[1], tables[d], MODELS[1], BEST, COLUMN_MODEL.get(d, (None, "best winner"))[1])
        for ax in a if k else a[1:]:
            ax.tick_params(labelleft=False)
        tops.append(a[0])
        b = fig.add_subplot(gs[1, 3 * k:3 * k + 2])
        {"subjective_randomness": classic_sr, "rsa": classic_rsa,
         "number_game": lambda ax, _: _pending(ax, "generalization profiles")}[d](b, tables[d])
        {"subjective_randomness": new_sr, "rsa": new_rsa, "number_game": new_ng}[d](
            fig.add_subplot(gs[2, 3 * k:3 * k + 2]), tables[d])
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
        # clear of the y axis titles, and in (c) of the panel titles
        y = title_y if row == 0 else gs[row, 0].get_position(fig).y1 + (0.035, 0.035, 0.052)[row]
        t = fig.text(0.005, y, label, fontdict=TITLE_FONT, fontsize=10, va="center")
        if ROW_TITLES[row]:  # just right of the letter
            fig.canvas.draw()
            x1 = to_fig.transform((t.get_window_extent().x1, 0))[0]
            fig.text(x1 + 0.008, y, ROW_TITLES[row], fontdict=TITLE_FONT, fontsize=8.5, va="center")
    key = [Line2D([], [], color=HUMAN, marker="o", ms=3, lw=0, label="people (95% CI)"),
           Line2D([], [], color=SEED, lw=1.5, label="best seed model"),
           Line2D([], [], color=BEST, lw=1.5, label="winning model")]
    fig.legend(handles=key, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, out_dir, "fig3_effects")
