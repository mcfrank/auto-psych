"""Figure 3 (``fig3_effects``), one column per domain, from ``data/paper/figure_data/phenomena/`` (its README
defines the patterns): (A) each condition's observed proportion against the best starting model's prediction
and against the best winner's, with R² and RMSE (trial-weighted); (B) a classic effect that both capture; (C) a pattern only the winner
captures. "Best" is the highest log-likelihood on these trials (the starting models' blend excluded)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from scripts.paper.figures.icons import STIMULI
from scripts.paper.figures.style import INK, INK_2, TITLE_FONT, TITLES, save

SEED, BEST, HUMAN = "#b3b3b3", "#2171b5", INK  # best starting model, best winner, people
WIDTH_IN, HEIGHT_IN = 7.0, 5.0
ORDER = ["subjective_randomness", "rsa", "number_game"]
ROWS = ("(a)", "(b)", "(c)")
DISTANCE_BANDS = ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 99))


def load(phenomena: Path) -> Dict[str, pd.DataFrame]:
    """Each domain's table: one row per binary trial (``y``) or, for RSA, per display x choice option."""
    sr = pd.read_csv(phenomena / "sr_trials.csv").rename(columns={"chose_a": "y"})
    ng = pd.read_csv(phenomena / "ng_trials.csv").rename(columns={"said_yes": "y"})
    rsa = pd.read_csv(phenomena / "rsa_display_classes.csv")
    for df, prefix in ((sr, "p_a_"), (ng, "p_yes_"), (rsa, "p_")):
        df.columns = [c.replace(prefix, "p_", 1) if c.startswith(prefix) else c for c in df.columns]
    return {"subjective_randomness": sr, "number_game": ng, "rsa": rsa}


def best(df: pd.DataFrame, domain: str) -> Dict[str, str]:
    """The best starting model's and the best winner's columns, by log-likelihood."""
    def ll(c):
        p = df[c].clip(1e-9, 1 - 1e-9)
        if domain == "rsa":
            return (df.observed_count * np.log(p)).sum()
        return (df.y * np.log(p) + (1 - df.y) * np.log(1 - p)).sum()
    pick = lambda cs: max(cs, key=ll)  # noqa: E731
    cols = [c for c in df if c.startswith("p_") and c != "p_seed_blend_cv"]
    return {"seed": pick([c for c in cols if "seed_" in c or "promoted_" in c]),
            "winner": pick([c for c in cols if "win_" in c or "final_" in c])}


def r2(o, p, n) -> float:
    """Trial-weighted R² of the predictions: 1 - residual / total variance of the observed proportions."""
    w = n / n.sum()
    return float(1 - (w * (o - p) ** 2).sum() / (w * (o - (w * o).sum()) ** 2).sum())


def conditions(df: pd.DataFrame, domain: str, cols) -> pd.DataFrame:
    """One row per condition: n, observed and each model's predicted proportion."""
    if domain == "rsa":
        out = pd.DataFrame({"n": df.n_trials, "observed": df.observed_count / df.n_trials})
        return out.assign(**{c: df[c] for c in cols})
    if domain == "subjective_randomness":  # a pair either way round, from its alphabetically first sequence
        first = np.minimum(df.sequence_a, df.sequence_b)
        flip = (df.sequence_a != first).to_numpy()
        df = df.assign(key=first + "|" + np.maximum(df.sequence_a, df.sequence_b), y=np.where(flip, 1 - df.y, df.y),
                       **{c: np.where(flip, 1 - df[c], df[c]) for c in cols})
    else:
        df = df.assign(key=df.number_set + "|" + df.target.astype(str))
    return df.groupby("key").agg(n=("y", "size"), observed=("y", "mean"), **{c: (c, "mean") for c in cols})


def groups(df, by, cols, labels=None):
    """People's share (and binomial 95% CI) and each model's mean prediction in each group of trials."""
    g = df.groupby(by)
    out = pd.DataFrame({"observed": g.y.mean(), "ci": 1.96 * np.sqrt(g.y.mean() * (1 - g.y.mean()) / g.y.size()),
                        **{c: g[c].mean() for c in cols}})
    return out if labels is None else out.loc[list(labels)].set_axis(list(labels.values()))


def rsa_roles(df, cols, displays, roles):
    """Per display, each role's share (observed and predicted), then the displays' trial-weighted mean."""
    d = df[df.display.isin(displays) & df.role.isin(roles)]
    per = d.groupby(["display", "role"]).agg(n=("n_trials", "first"), observed=("observed_count", "sum"),
                                             **{c: (c, "sum") for c in cols}).reset_index()
    per["observed"] /= per.n
    rows = {}
    for role in roles:
        r = per[per.role == role]
        o = np.average(r.observed, weights=r.n)
        rows[role.replace("_match", "")] = {"observed": o, "ci": 1.96 * np.sqrt(o * (1 - o) / r.n.sum()),
                                            **{c: np.average(r[c], weights=r.n) for c in cols}}
    return pd.DataFrame(rows).T


def effects(tables: Dict[str, pd.DataFrame]):
    """Each domain's (B) classic and (C) new effect: (table, x label, y label, drawn as a line)."""
    out = {}
    sr = tables["subjective_randomness"]
    cols = list(best(sr, "subjective_randomness").values())
    # B: over-alternation (Falk & Konold, 1997): each sequence's chance of being chosen by its alternation rate
    both = pd.concat([pd.DataFrame({"seq": sr.sequence_a, "y": sr.y, **{c: sr[c] for c in cols}}),
                      pd.DataFrame({"seq": sr.sequence_b, "y": 1 - sr.y, **{c: 1 - sr[c] for c in cols}})])
    alt = both.seq.map(lambda s: sum(a != b for a, b in zip(s, s[1:])) / (len(s) - 1))
    both["band"] = np.clip((alt * 5).round() / 5, 0, 1)
    classic = groups(both, "band", cols)
    classic.index = [f"{v:.1f}" for v in classic.index]
    # C: a pure streak gets ~10% of choices whatever it faces
    s = sr[sr.streak_floor_pair == 1]
    a_is = (s.constant_a == 1).to_numpy()
    s = s.assign(y=np.where(a_is, s.y, 1 - s.y), other=np.where(a_is, s.switches_b, s.switches_a),
                 **{c: np.where(a_is, s[c], 1 - s[c]) for c in cols})
    new = groups(s, "other", cols)
    new.index = new.index.astype(str)
    out["subjective_randomness"] = ((classic, "alternation rate", "chosen as more random", True),
                                    (new, "switches in the sequence a streak faces", "chose the streak", True))
    ng = tables["number_game"]
    cols = list(best(ng, "number_game").values())
    # B: the similarity gradient (Shepard, 1987; Tenenbaum, 2000): yes by distance to the nearest example
    dist = [min(abs(t - int(e)) for e in s.split(",")) for s, t in zip(ng.number_set, ng.target)]
    labels = {i: (str(lo) if lo == hi else f"{lo}+" if hi == 99 else f"{lo}–{hi}")
              for i, (lo, hi) in enumerate(DISTANCE_BANDS)}
    band = np.searchsorted([hi for _, hi in DISTANCE_BANDS], dist)
    classic = groups(ng.assign(band=band), "band", cols, labels)
    # C: one example extends to its multiples
    one = ng[ng.n_examples == 1]
    new = groups(one, "target_multiple_of_single_example", cols, {1: "multiples", 0: "other numbers"})
    out["number_game"] = ((classic, "distance to the nearest example", "said yes", True),
                          (new, "after one example", "said yes", False))
    rsa = tables["rsa"]
    cols = list(best(rsa, "rsa").values())
    roles = rsa.groupby("display").role.agg(set)
    # B: pragmatic inference (Frank & Goodman, 2012): of two objects a word fits, the one with fewer features
    two = roles[roles.map(lambda r: {"simplest_match", "richest_match"} <= r and "middle_match" not in r)].index
    classic = rsa_roles(rsa, cols, two, ["simplest_match", "richest_match"])
    # C: the candidate in the middle is passed over
    new = rsa_roles(rsa, cols, rsa.display[rsa.interior_display == 1].unique(),
                    ["simplest_match", "middle_match", "richest_match"])
    out["rsa"] = ((classic, "of two objects the word fits", "chosen", False),
                  (new, "of three objects the word fits", "chosen", False))
    return out


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


def _effect(ax, table, cols, xlabel, ylabel, line):
    x = np.arange(len(table))
    for c, color in zip(cols, (SEED, BEST)):
        if line:
            ax.plot(x, table[c], color=color, lw=1.3, zorder=2)
        else:
            for xi, v in zip(x, table[c]):
                ax.plot([xi - 0.25, xi + 0.25], [v, v], color=color, lw=1.8, solid_capstyle="butt", zorder=2)
    ax.errorbar(x, table.observed, yerr=table.ci, fmt="o", color=HUMAN, ms=3, elinewidth=0.8, zorder=3)
    ax.set_xticks(x, table.index)
    ax.set_xlim(-0.5, len(table) - 0.5)
    ax.set_ylim(0, max(0.2, 1.15 * table[["observed", *cols]].to_numpy().max()))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)


def draw(phenomena: Path, out_dir: Path) -> None:
    tables = load(phenomena)
    eff = effects(tables)
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    # one grid for every row, so a domain's panels share a column: row (a)'s two scatterplots side by side,
    # rows (b) and (c) spanning both; a spacer column between domains
    gs = fig.add_gridspec(3, 8, width_ratios=[1, 1, 0.32, 1, 1, 0.32, 1, 1], height_ratios=[1.2, 1, 1],
                          wspace=0.18, hspace=0.5, left=0.075, right=0.99, top=0.9, bottom=0.11)
    tops = []
    for k, d in enumerate(ORDER):
        cols = list(best(tables[d], d).values())
        cond = conditions(tables[d], d, cols)
        a = [fig.add_subplot(gs[0, 3 * k + j]) for j in (0, 1)]
        _scatter(a[0], cond, cols[0], SEED, "best seed")
        _scatter(a[1], cond, cols[1], BEST, "best winner")
        for ax in a if k else a[1:]:
            ax.tick_params(labelleft=False)
        tops.append(a[0])
        for row, (table, xlabel, ylabel, line) in zip((1, 2), eff[d]):
            _effect(fig.add_subplot(gs[row, 3 * k:3 * k + 2]), table, cols, xlabel, ylabel, line)
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
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    for row, label in enumerate(ROWS):  # over the row's highest text (row A: the domains' titles)
        y = title_y if row == 0 else gs[row, 0].get_position(fig).y1 + (0.035 if row == 1 else 0.025)  # (b) clear of the y axis title
        fig.text(0.005, y, label, fontdict=TITLE_FONT, fontsize=10, va="center")
    key = [Line2D([], [], color=HUMAN, marker="o", ms=3, lw=0, label="people (95% CI)"),
           Line2D([], [], color=SEED, lw=1.5, label="best starting model"),
           Line2D([], [], color=BEST, lw=1.5, label="best winning model")]
    fig.legend(handles=key, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, out_dir, "fig3_effects")
