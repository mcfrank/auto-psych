"""Figure 3 (``fig3_effects``), one column per domain: (a) observed against each model's prediction, with
trial-weighted R² and RMSE; (b) the domain's classic effect, drawn as its original paper drew it, which both
models capture; (c) a new effect only the winner captures.

Row (a) uses Figure 2's split (``load_heldout``: ``<domain>_heldout_conditions.json``, written by
``scripts/paper/fig3_heldout_conditions.py``): subjective randomness' leave-one-out (PSIS-LOO) predictions on all
runs pooled, the number game's predictions for the items no run drew (people's data restricted to those items),
and reference games' pre-registered test (``data/rsa/viz/panel_displays.json`` says what each display showed).
Row (b) shows the full-data fit on all the data: subjective randomness' in-sample predictions, the number game's
on every item, seen and unseen (``all_items``), and reference games' pre-registered test as in row (a).
Row (c) shows where row (a)'s held-out advantage concentrates, on row (a)'s predictions: each pattern
(``streak_choices``, ``fits_three_roles``, ``one_example_roles``) is a candidate of the search over a fixed list of
stimulus features (``scripts/paper/fig3_patterns.py``), chosen by hand for its reading; the number game's panel draws
only two of ``one_example_roles``' four roles, the factors and "other" (``NG_SHOWN_ROLES``). ``load`` gives the figure
data's in-sample ``conditions``, each condition's observed proportion and the pooled best model's and best seed's
posterior-mean predictions; a column whose model is not the pooled best, ``COLUMN_MODEL``, takes that model's from
``<domain>_model_conditions.json``."""

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
from src.models.clustered_se import cluster_dse

SEED, BEST, HUMAN = "#b3b3b3", "#2171b5", INK  # best seed model, best winner, people
WIDTH_IN, HEIGHT_IN = 7.0, 5.4
ORDER = ["subjective_randomness", "rsa", "number_game"]
ROWS = ("(a)", "(b)", "(c)")
ROW_TITLES = (None, "Classic effect captured by both models", "Where our model beats the seed")
MODELS = ("best_seed", "best_model")
# A column's model when it is not the pooled best (the figure data's ``best_model``): its name and axis label.
# Its predictions, of the same kind as the figure data's, are in ``<domain>_model_conditions.json``
# (``scripts/paper/number_game_model_conditions.py``). The number game shows run 3's winner, the best on
# unseen items (Fig. 2) and the model Fig. 4 ablates; the pooled CV best is run 2's.
COLUMN_MODEL = {"number_game": ("exception_weighted_rule_rejection", "best discovered")}
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


HELDOUT = "{}_heldout_conditions.json"  # rows (a) and (c): Figure 2's split, per domain


def load_heldout(figure_data: Path) -> Dict[str, pd.DataFrame]:
    """Rows (a) and (c)'s conditions on Figure 2's split for subjective randomness and the number game, of each
    column's models: the figure data's best seed and its best model, or the column's ``COLUMN_MODEL``. (Reference
    games' conditions are the pre-registered test already.)"""
    out = {}
    for d in ("subjective_randomness", "number_game"):
        pooled = json.loads((figure_data / f"{d}.json").read_text())["pooled"]
        held = json.loads((figure_data / HELDOUT.format(d)).read_text())
        want = {"best_model": COLUMN_MODEL.get(d, (pooled["best_model"],))[0], "best_seed": pooled["best_seed"]}
        if held["models"] != want:
            raise RuntimeError(f"{d}'s held-out predictions are {held['models']}'s, not the column's {want}")
        out[d] = pd.DataFrame(held["conditions"])
    return out


def all_items(seen: pd.DataFrame, unseen: pd.DataFrame) -> pd.DataFrame:
    """The number game's items the runs drew (``load``) and those none drew (``load_heldout``), one table of the
    full-data fit's predictions on every item."""
    key = ["number_set", "target"]
    if len(seen.merge(unseen[key], on=key)):
        raise RuntimeError("an item is both among the seen items and the unseen ones")
    return pd.concat([seen, unseen[seen.columns]], ignore_index=True)


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


SCATTER_INK = 250  # a scatterplot's total opacity budget, alpha times relative point area summed over the points
MIN_ALPHA = 0.1  # fainter points stack into a colour shift (8-bit alpha), so many points shrink instead


def _scatter(ax, cond, col, color, label):
    ax.plot([0, 1], [0, 1], color="#e0e0e0", lw=0.8, zorder=0)
    # many conditions (the number game's 23,360 held-out items) get fainter, then smaller points, so density shows
    alpha = max(MIN_ALPHA, min(0.25, SCATTER_INK / len(cond)))
    shrink = min(1.0, SCATTER_INK / (len(cond) * alpha))
    ax.scatter(cond[col], cond.observed, s=shrink * (4 + 12 * np.sqrt(cond.n / cond.n.max())), color=color,
               alpha=alpha, linewidths=0, clip_on=False, zorder=1, rasterized=True)
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


GAME = [[0, 1], [1, 0], [1, 1]]  # the ad-hoc implicature game: the word (feature 0) fits two of three objects


def reference_game(rsa: pd.DataFrame) -> pd.DataFrame:
    """The pre-registered display with the ad-hoc implicature game's structure, one row per object (its class's share)."""
    d = rsa[rsa.objects.map(lambda o: o == GAME) & (rsa.utterance == 0)]
    if d.display.nunique() != 1:
        raise RuntimeError(f"{d.display.nunique()} pre-registered displays have the ad-hoc implicature game's structure, not 1")
    return d.sort_values("cls").set_index("cls")


# Stiller, Goodman & Frank (2015)'s names for the three objects, and what each is in our display
GAME_ROLES = {(0, 1): ("distractor", "doesn't fit\nthe word"),
              (1, 0): ("one-feature", "contains only\nthe word's\nfeature"),
              (1, 1): ("two-feature", "contains the\nword's feature\n+ another")}


def classic_rsa(ax, rsa):
    """Ad-hoc implicature (Stiller, Goodman & Frank, 2015; Frank & Goodman, 2012): of the two objects the word
    fits, people pick the one-feature object, since a speaker would have named the two-feature one's other
    feature. One pre-registered display; its pictures and words varied across participants."""
    d = reference_game(rsa)
    x = np.arange(len(d))
    roles = [GAME_ROLES[tuple(GAME[classes.index(cls)])] for cls, classes in zip(d.index, d.choice_classes)]
    w = 0.26
    o = d.observed.to_numpy()
    ax.bar(x - w, o, w, color=HUMAN, zorder=2)
    ax.errorbar(x - w, o, yerr=1.96 * np.sqrt(o * (1 - o) / d.n), fmt="none", ecolor=INK_2, elinewidth=0.7, zorder=3)
    for dx, c, color in ((0, MODELS[1], BEST), (w, MODELS[0], SEED)):  # people, the discovered model, the seed
        ax.bar(x + dx, d[c], w, color=color, zorder=2)
    ax.set_xticks(x, [name for name, _ in roles])
    ax.tick_params(axis="x", length=0)
    for xi, (_, what) in zip(x, roles):  # each role's description, smaller, under its name
        ax.annotate(what, (xi, 0), xycoords=("data", "axes fraction"), xytext=(0, -17), textcoords="offset points",
                    ha="center", va="top", fontsize=5.5, color=INK_2, annotation_clip=False)
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


ONE_EXAMPLE_ROLES = ("the example", "a multiple", "a factor", "other")


def one_example_roles(ng: pd.DataFrame) -> pd.DataFrame:
    """Single-example sets: the chance of 'yes' by what the number is to the example (the search's terms,
    ``fig3_patterns._ng``), trial-weighted; each role's trials and people's 95% CI clustered by example set."""
    from scripts.paper.fig3_patterns import FEATURES  # it imports this module

    role = pd.Series([FEATURES["number_game"](c)["after one example, the number is"] for c in ng.itertuples()],
                     index=ng.index)
    d = ng[role.isin(ONE_EXAMPLE_ROLES)].assign(role=role)
    rows = {}
    for r, g in d.groupby("role"):
        t = {k: float(np.average(g[k], weights=g.n)) for k in ("observed", *MODELS)}
        resid = g.n * (g.observed - t["observed"])
        se = cluster_dse(resid, np.zeros(len(g)), pd.factorize(g.number_set)[0]) / g.n.sum()
        rows[r] = {**t, "trials": int(g.n.sum()), "ci": 1.96 * se}
    return pd.DataFrame(rows).T.loc[list(ONE_EXAMPLE_ROLES)].astype({"trials": int})


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
    ax.set_xlabel("switch rate of the other sequence", labelpad=1)
    ax.set_ylabel("chosen as more random", labelpad=1)
    _panel_title(ax, "An all-heads or all-tails sequence")


FITS_THREE_ROLES = ("doesn't fit the word", "fewest features", "in between", "most features")


def fits_three_roles(rsa: pd.DataFrame) -> pd.DataFrame:
    """Displays where the word fits exactly three objects (``ROW_C``): each object's chance of being chosen (its
    class's share over its copies), by its role in the search's terms (``fig3_patterns._rsa``), averaged over
    objects weighted by their display's trials; each role's displays' trials and people's 95% CI clustered by
    display. Displays whose fitting objects all have as many features ("all fitting alike") are left out: 7
    displays, each such object at about a third for people and both models."""
    from scripts.paper.fig3_patterns import FEATURES  # it imports this module

    feature, value = ROW_C["rsa"]
    feats = pd.DataFrame([FEATURES["rsa"](c) for c in rsa.itertuples()], index=rsa.index)
    d = rsa[feats[feature] == value].assign(role=feats["the object"])
    d = d[d.role.isin(FITS_THREE_ROLES)]
    copies = np.array([list(c.choice_classes).count(c.cls) for c in d.itertuples()])
    d = d.assign(w=d.n * copies, **{k: d[k] / copies for k in ("observed", *MODELS)})  # each copy one object
    rows = {}
    for role, g in d.groupby("role"):
        t = {k: float(np.average(g[k], weights=g.w)) for k in ("observed", *MODELS)}
        resid = g.w * (g.observed - t["observed"])
        se = cluster_dse(resid, np.zeros(len(g)), pd.factorize(g.display)[0]) / g.w.sum()
        rows[role] = {**t, "trials": int(g.drop_duplicates("display").n.sum()), "ci": 1.96 * se}
    return pd.DataFrame(rows).T.loc[list(FITS_THREE_ROLES)].astype({"trials": int})


def _role_bars(ax, t, labels):
    """Per role, side-by-side bars for people (with their 95% CI), the discovered model and the seed."""
    x = np.arange(len(t))
    w = 0.26
    ax.bar(x - w, t.observed, w, color=HUMAN, zorder=2)
    ax.errorbar(x - w, t.observed, yerr=t.ci, fmt="none", ecolor=INK_2, elinewidth=0.7, zorder=3)
    for dx, c, color in ((0, MODELS[1], BEST), (w, MODELS[0], SEED)):
        ax.bar(x + dx, t[c], w, color=color, zorder=2)
    ax.set_xticks(x, labels, fontsize=6, va="top")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(x) - 0.4)


def new_rsa(ax, rsa):
    """When the word fits three objects, people pick the one with the fewest features; the seed spreads its choices
    to the one in between."""
    _role_bars(ax, fits_three_roles(rsa), ["doesn't\nfit", "fewest\nfeatures", "middle", "most\nfeatures"])
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("chosen", labelpad=1)
    ax.set_xlabel("object", labelpad=1)
    _panel_title(ax, "The word fits three objects")


NG_SHOWN_ROLES = ("a factor", "other")  # of ``ONE_EXAMPLE_ROLES``: the factor effect against the search's baseline


def new_ng(ax, ng):
    """After one example, people often say yes to its factors; the best seed rarely does. Only the factors and
    "other" (the search's numbers that are neither the example, a multiple nor a factor) are drawn,
    ``NG_SHOWN_ROLES``, their rows of ``one_example_roles`` unchanged; y from 0 to 60%, as reference games'."""
    _role_bars(ax, one_example_roles(ng).loc[list(NG_SHOWN_ROLES)], ["factor", "other"])
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("said yes", labelpad=1)
    ax.set_xlabel("queried number, relative to the example", labelpad=1)
    _panel_title(ax, "Factors of a single example")


NG_SETS = ("100", "80,70,90,100")  # one example, then four that fit a rule (as the figure data write them)


def ng_profiles(ng: pd.DataFrame, sets=NG_SETS) -> Dict[str, pd.DataFrame]:
    """Tenenbaum (2000)'s generalization profiles: per example set of ``sets``, people's share of yes and each
    model's P(yes) for every target 1-100 that ``ng`` has, NaN for the others."""
    out = {}
    for ex in sets:
        d = ng[ng.number_set == ex]
        if d.empty:
            raise RuntimeError(f"no conditions for example set {ex!r}")
        out[ex] = d.set_index("target")[["observed", *MODELS]].reindex(range(1, 101))
    return out


def classic_ng(cell, fig, profiles):
    """Each set's profile: per number, side-by-side bars for people, the discovered
    model and the seed (as in the reference-games panel)."""
    axes = cell.subgridspec(len(profiles), 1, hspace=0.5).subplots(sharex=True)
    w = 0.3
    for ax, (ex, p) in zip(axes, profiles.items()):
        for dx, c, color in ((-w, "observed", HUMAN), (0, MODELS[1], BEST), (w, MODELS[0], SEED)):
            ax.bar(p.index + dx, p[c], width=w, color=color, linewidth=0)
        ax.set(xlim=(0, 101), ylim=(0, 1), yticks=[0, 1])
        ax.yaxis.set_major_formatter(PCT)
        ax.set_title("examples: " + ex.replace(",", ", "), fontsize=6, loc="left", pad=2)
    axes[0].tick_params(axis="x", length=0)
    axes[-1].set(xticks=[1, 25, 50, 75, 100])
    axes[-1].set_xlabel("number queried", labelpad=1)
    axes[0].set_ylabel("said yes", labelpad=1, y=-0.25)


# What is not yet as it should be, drawn on the panel in red until it is fixed: (row, domain) -> what is missing.
TO_FIX: Dict[tuple, str] = {}
# Row (c): per domain, the search's candidate on the held-out predictions (scripts/paper/fig3_patterns.py) with the
# clearest behavioural reading
ROW_C = {"subjective_randomness": ("a pure streak in the pair", "one"), "rsa": ("objects the word fits", "3"),
         "number_game": ("after one example, the number is", "a factor")}


def check_row_c(patterns: Path) -> None:
    """Each of ``ROW_C`` must be one of the search's candidates."""
    found = json.loads(patterns.read_text())
    for d, pick in ROW_C.items():
        candidates = [(r["feature"], r["value"]) for r in found[d] if r["candidate"]]
        if pick not in candidates:
            raise RuntimeError(f"{d}: row (c)'s pattern {pick} is not one of the search's candidates {candidates}")


FIX_RED = "#c62828"


def _flag(fig, cell, text):
    """A red note over a panel that is not yet as it should be."""
    box = cell.get_position(fig)
    fig.text((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, text, ha="center", va="center", fontsize=6.5,
             color=FIX_RED, weight="bold", zorder=10,
             bbox={"fc": "white", "ec": FIX_RED, "lw": 0.8, "alpha": 0.9, "boxstyle": "round,pad=0.3"})


def draw(figure_data: Path, displays: Path, out_dir: Path) -> None:
    check_row_c(figure_data / "fig3_patterns.json")
    tables = load(figure_data, displays)  # in sample
    held = {**tables, **load_heldout(figure_data)}  # Figure 2's split: rows (a) and (c)
    full = {**tables, "number_game": all_items(tables["number_game"], held["number_game"])}  # row (b)
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    # one grid for every row, so a domain's panels share a column: row (a)'s two scatterplots side by side,
    # rows (b) and (c) spanning both; a spacer column between domains
    gs = fig.add_gridspec(3, 8, width_ratios=[1, 1, 0.32, 1, 1, 0.32, 1, 1], height_ratios=[1.2, 1, 1],
                          wspace=0.18, hspace=0.78, left=0.075, right=0.99, top=0.9, bottom=0.11)
    tops = []
    for k, d in enumerate(ORDER):
        a = [fig.add_subplot(gs[0, 3 * k + j]) for j in (0, 1)]
        _scatter(a[0], held[d], MODELS[0], SEED, "best seed")
        _scatter(a[1], held[d], MODELS[1], BEST, COLUMN_MODEL.get(d, (None, "best discovered"))[1])
        for ax in a if k else a[1:]:
            ax.tick_params(labelleft=False)
        tops.append(a[0])
        if d == "number_game":
            classic_ng(gs[1, 3 * k:3 * k + 2], fig, ng_profiles(full[d]))
        else:
            {"subjective_randomness": classic_sr, "rsa": classic_rsa}[d](fig.add_subplot(gs[1, 3 * k:3 * k + 2]),
                                                                       full[d])
        {"subjective_randomness": new_sr, "rsa": new_rsa, "number_game": new_ng}[d](
            fig.add_subplot(gs[2, 3 * k:3 * k + 2]), held[d])
    tops[0].set_ylabel("observed", labelpad=1)
    for (row, d), text in TO_FIX.items():
        k = ORDER.index(d)
        _flag(fig, gs[row, 3 * k:3 * k + 2], text)
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
           Line2D([], [], color=BEST, lw=1.5, label="best discovered model")]
    fig.legend(handles=key, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, out_dir, "fig3_effects")
