"""Figure 3 (``fig3_phenomena``): six patterns in people's responses that the winning models predict and every
starting model misses, from ``data/paper/figure_data/phenomena/`` (its README defines each pattern). Each
panel's bars are people; on each, a dot per winning model and per starting model, and the starting models'
cross-validated blend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scripts.paper.figures.style import INK, INK_2, TITLE_FONT, TITLES, save

COLOR_WIN, COLOR_SEED, BAR = "#08306b", "#6baed6", "#e0e0e0"  # Figure 4's new / from-the-literature blues
WIDTH_IN, HEIGHT_IN = 7.0, 4.2
STREAK, VARIETY, REVERSERS = ("A pure streak gets ~10%, whatever it faces", "Varied runs look random",
                              "Some people reverse")
MULTIPLES, SOFT_YES, MIDDLE = ("One example extends to its multiples", "Soft yes to members, firm no to breakers",
                               "The middle candidate is passed over")
REVERSED = 0.55  # agreeing with the majority on fewer trials than this
SIMULATIONS = 200  # simulated choice sets per model for the reversers' share


@dataclass
class Pattern:
    title: str
    domain: str
    ylabel: str
    rows: Dict[str, Dict]  # category -> people, n, winners, seeds, blend
    xlabel: str = ""


def load(phenomena: Path) -> Dict[str, pd.DataFrame]:
    return {k: pd.read_csv(phenomena / f"{k}.csv") for k in ("sr_trials", "ng_trials", "rsa_display_classes")}


def _columns(df, prefix):
    cols = [c for c in df if c.startswith(prefix)]
    return ([c for c in cols if "_win_" in c or "_final_" in c],
            [c for c in cols if ("_seed_" in c or "_promoted_" in c) and not c.endswith("seed_blend_cv")],
            prefix + "seed_blend_cv")


def _summary(people, n, value, wins, seeds, blend):
    """One category: people's share and each model's (``value``: column -> its share)."""
    return {"people": float(people), "n": int(n), "winners": [value(c) for c in wins],
            "seeds": [value(c) for c in seeds], "blend": value(blend)}


def _row(observed, predicted, wins, seeds, blend):
    return _summary(np.mean(observed), len(observed), lambda c: predicted[c].mean(), wins, seeds, blend)


def _sr(sr: pd.DataFrame) -> List[Pattern]:
    wins, seeds, blend = _columns(sr, "p_a_")
    models = wins + seeds + [blend]

    def toward(df, a_is):  # choosing the sequence ``a_is`` marks: observed and predicted
        flip = lambda v: np.where(a_is, v, 1 - v)  # noqa: E731
        return flip(df.chose_a.to_numpy()), pd.DataFrame({c: flip(df[c].to_numpy()) for c in models})

    s = sr[sr.streak_floor_pair == 1]
    other = np.where(s.constant_a == 1, s.switches_b, s.switches_a)
    chose, pred = toward(s, (s.constant_a == 1).to_numpy())
    streak = {str(k): _row(chose[other == k], pred[other == k], wins, seeds, blend) for k in np.unique(other)}
    v = sr[sr.variety_pair == 1]
    chose, pred = toward(v, (v.run_variety_a > v.run_variety_b).to_numpy())
    # reversers: people agreeing with each pair's majority on under 55% of their trials (as the phenomena's
    # provenance/sr_person2.py: a tie goes to the later sequence); each model's share from choices simulated
    # from its predictions (people's choices are noisy too)
    first = np.minimum(sr.sequence_a, sr.sequence_b)
    chose_first = pd.Series(np.where(sr.sequence_a == first, sr.chose_a, 1 - sr.chose_a))
    first_wins = chose_first.groupby([first, np.maximum(sr.sequence_a, sr.sequence_b)]).transform("mean") > 0.5
    maj_is_a = (sr.sequence_a == first).to_numpy() == first_wins.to_numpy()
    person = sr.person.to_numpy()

    def share(chose_a):  # (simulations x trials) -> each simulation's share of reversers
        agree = pd.DataFrame((chose_a == maj_is_a).T).groupby(person).mean()
        return (agree < REVERSED).mean(axis=0).to_numpy()

    rng = np.random.default_rng(0)
    sim = {c: share(rng.random((SIMULATIONS, len(sr))) < sr[c].to_numpy()).mean() for c in models}
    reversers = _summary(share(sr.chose_a.to_numpy()[None] == 1)[0], sr.person.nunique(), sim.get, wins, seeds, blend)
    return [Pattern(STREAK, "subjective_randomness", "chose the streak", streak, "switches in the other sequence"),
            Pattern(VARIETY, "subjective_randomness", "chose it", {"more varied": _row(chose, pred, wins, seeds, blend)}),
            Pattern(REVERSERS, "subjective_randomness", "people", {"agree with the\nmajority < 55%": reversers})]


def _ng(ng: pd.DataFrame) -> List[Pattern]:
    wins, seeds, blend = _columns(ng, "p_yes_")
    row = lambda d: _row(d.said_yes, d, wins, seeds, blend)  # noqa: E731
    one = ng[ng.n_examples == 1]
    digit, parity = ng[ng.shared_digit_set == 1], ng[ng.shared_parity_set == 1]
    return [Pattern(MULTIPLES, "number_game", "said yes",
                    {"multiples": row(one[one.target_multiple_of_single_example == 1]),
                     "other numbers": row(one[one.target_multiple_of_single_example == 0])}),
            Pattern(SOFT_YES, "number_game", "said yes",
                    {"keeps\nlast digit": row(digit[digit.target_keeps_digit == 1]),
                     "breaks\nlast digit": row(digit[digit.target_keeps_digit == 0]),
                     "breaks\nparity": row(parity[parity.target_keeps_parity == 0])})]


def _rsa(rsa: pd.DataFrame) -> List[Pattern]:
    wins, seeds, blend = _columns(rsa, "p_")
    models = wins + seeds + [blend]
    per = rsa[rsa.interior_display == 1].groupby(["display", "role"]).agg(
        n=("n_trials", "first"), observed=("observed_count", "sum"), **{c: (c, "sum") for c in models}).reset_index()
    per["observed"] /= per.n
    rows = {}
    for role in ("simplest", "middle", "richest"):
        r = per[per.role == f"{role}_match"]
        mean = lambda c: float(np.average(r[c], weights=r.n))  # noqa: E731
        rows[role] = _summary(mean("observed"), r.n.sum(), mean, wins, seeds, blend)
    return [Pattern(MIDDLE, "rsa", "chosen", rows, "objects the word fits, by features")]


def patterns(tables: Dict[str, pd.DataFrame]) -> List[Pattern]:
    return _sr(tables["sr_trials"]) + _ng(tables["ng_trials"]) + _rsa(tables["rsa_display_classes"])


def _panel(ax, p: Pattern) -> None:
    cats = list(p.rows)
    x = np.arange(len(cats))
    people = np.array([p.rows[c]["people"] for c in cats])
    se = np.sqrt(people * (1 - people) / np.array([p.rows[c]["n"] for c in cats]))
    ax.bar(x, people, width=0.7, color=BAR, zorder=1)
    ax.errorbar(x, people, yerr=1.96 * se, fmt="none", ecolor=INK_2, elinewidth=0.6, zorder=2)
    for xi, c in zip(x, cats):
        r = p.rows[c]
        ax.scatter(np.full(len(r["winners"]), xi - 0.15), r["winners"], s=10, color=COLOR_WIN, linewidths=0, zorder=4)
        ax.scatter(np.full(len(r["seeds"]), xi + 0.15), r["seeds"], s=10, color=COLOR_SEED, linewidths=0, zorder=3)
        ax.scatter([xi + 0.15], [r["blend"]], s=22, facecolors="white", edgecolors=COLOR_SEED, linewidths=1.0, zorder=5)
    ax.set_xticks(x, cats)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    top = max(people.max(), *(max(v["winners"] + v["seeds"]) for v in p.rows.values()))
    ax.set_ylim(0, max(0.2, 1.15 * top))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set(ylabel=p.ylabel, xlabel=p.xlabel)
    ax.set_title(p.title, fontsize=7, loc="left", color=INK)


def draw(phenomena: Path, out_dir: Path) -> None:
    ps = patterns(load(phenomena))
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    rows = fig.subfigures(2, 1)
    axes = list(rows[0].subplots(1, 3, gridspec_kw={"width_ratios": [5, 1.4, 1.4], "wspace": 0.6}))
    axes += list(rows[1].subplots(1, 3, gridspec_kw={"width_ratios": [2, 3, 3], "wspace": 0.45}))
    for r in rows:
        r.subplots_adjust(left=0.07, right=0.99, top=0.74, bottom=0.2)
    for ax, p in zip(axes, ps):
        _panel(ax, p)
    fig.canvas.draw()
    for ax, p in zip(axes, ps):  # each domain's name over its first panel
        if p is next(q for q in ps if q.domain == p.domain):
            ax.annotate(TITLES[p.domain], (0, 1), xycoords="axes fraction", xytext=(0, 16), textcoords="offset points",
                        **TITLE_FONT, fontsize=8.5, ha="left", va="bottom")
    key = [Patch(color=BAR, label="people (95% CI)"),
           Line2D([], [], marker="o", lw=0, color=COLOR_WIN, ms=3.5, label="winning models"),
           Line2D([], [], marker="o", lw=0, color=COLOR_SEED, ms=3.5, label="starting models"),
           Line2D([], [], marker="o", lw=0, mfc="white", mec=COLOR_SEED, ms=4.5, label="blend of starting models")]
    fig.legend(handles=key, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    save(fig, out_dir, "fig3_phenomena")
