"""Figure 3 (``fig3_effects``), one column per domain, from ``data/paper/figure_data/phenomena/`` (its README
defines the patterns): (A) each condition's observed proportion against the best seed model's prediction
and against the best winner's, with R² and RMSE (trial-weighted); (B) a classic effect that both capture; (C) a pattern only the winner
captures. "Best" is the highest log-likelihood on these trials (among the original seed models: not RSA's promoted seeds, nor the seeds' blend)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import json

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
DISTANCE_BANDS = ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 99))


def load(phenomena: Path) -> Dict[str, pd.DataFrame]:
    """Each domain's table: one row per binary trial (``y``) or, for RSA, per display x choice option."""
    sr = pd.read_csv(phenomena / "sr_trials.csv").rename(columns={"chose_a": "y"})
    ng = pd.read_csv(phenomena / "ng_trials.csv").rename(columns={"said_yes": "y"})
    rsa = pd.read_csv(phenomena / "rsa_display_classes.csv")
    rsa_trials = pd.read_csv(phenomena / "rsa_trials.csv", usecols=["display", "feature_names"])
    for df, prefix in ((sr, "p_a_"), (ng, "p_yes_"), (rsa, "p_")):
        df.columns = [c.replace(prefix, "p_", 1) if c.startswith(prefix) else c for c in df.columns]
    return {"subjective_randomness": sr, "number_game": ng, "rsa": rsa, "rsa_trials": rsa_trials}


def best(df: pd.DataFrame, domain: str) -> Dict[str, str]:
    """The best seed model's and the best winner's columns, by log-likelihood."""
    def ll(c):
        p = df[c].clip(1e-9, 1 - 1e-9)
        if domain == "rsa":
            return (df.observed_count * np.log(p)).sum()
        return (df.y * np.log(p) + (1 - df.y) * np.log(1 - p)).sum()
    pick = lambda cs: max(cs, key=ll)  # noqa: E731
    cols = [c for c in df if c.startswith("p_") and c != "p_seed_blend_cv"]
    return {"seed": pick([c for c in cols if "seed_" in c]),  # the original seeds, not RSA's promoted ones
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
    """Each domain's (c) new effect: (table, x label, y label, drawn as a line)."""
    out = {}
    sr = tables["subjective_randomness"]
    cols = list(best(sr, "subjective_randomness").values())
    s = sr[sr.streak_floor_pair == 1]  # a pure streak gets ~10% of choices whatever it faces
    a_is = (s.constant_a == 1).to_numpy()
    s = s.assign(y=np.where(a_is, s.y, 1 - s.y), other=np.where(a_is, s.switches_b, s.switches_a),
                 **{c: np.where(a_is, s[c], 1 - s[c]) for c in cols})
    new = groups(s, "other", cols)
    new.index = new.index.astype(str)
    out["subjective_randomness"] = (new, "switches in the sequence a streak faces", "chose the streak", True)
    ng = tables["number_game"]
    cols = list(best(ng, "number_game").values())
    one = ng[ng.n_examples == 1]  # one example extends to its multiples
    out["number_game"] = (groups(one, "target_multiple_of_single_example", cols, {1: "multiples", 0: "other numbers"}),
                          "after one example", "said yes", False)
    rsa = tables["rsa"]
    cols = list(best(rsa, "rsa").values())
    out["rsa"] = (rsa_roles(rsa, cols, rsa.display[rsa.interior_display == 1].unique(),  # the middle candidate
                            ["simplest_match", "middle_match", "richest_match"]),
                  "of three objects the word fits", "chosen", False)
    return out


# --- row (b): each classic effect as its original paper drew it ----------------------------------------------

SR_LENGTH = 8  # the commonest sequence length (79% of trials): alternation counts compare only within a length


def classic_sr(ax, sr, cols):
    """Falk & Konold (1997): how random a sequence looks against its number of alternations, peaking above a
    fair coin's average. Each length-8 sequence's chance of being chosen, over every pair it was in."""
    both = pd.concat([pd.DataFrame({"seq": sr.sequence_a, "y": sr.y, **{c: sr[c] for c in cols}}),
                      pd.DataFrame({"seq": sr.sequence_b, "y": 1 - sr.y, **{c: 1 - sr[c] for c in cols}})])
    both = both[both.seq.str.len() == SR_LENGTH]
    t = groups(both.assign(alt=both.seq.map(lambda q: sum(a != b for a, b in zip(q, q[1:])))), "alt", cols)
    fair = (SR_LENGTH - 1) / 2
    ax.axvline(fair, color="#b0b0b0", lw=0.6, ls=(0, (2, 2)), zorder=0)
    for c, color in zip(cols, (SEED, BEST)):
        ax.plot(t.index, t[c], color=color, lw=1.3, zorder=2)
    ax.errorbar(t.index, t.observed, yerr=t.ci, fmt="o", color=HUMAN, ms=3, elinewidth=0.8, zorder=3)
    ax.text(fair + 0.1, 0.03, "fair coin", transform=ax.get_xaxis_transform(), fontsize=6, color=INK_2)
    ax.set(xlim=(-0.3, SR_LENGTH - 0.7), ylim=(0, 0.8), xticks=range(SR_LENGTH))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel(f"alternations in {SR_LENGTH} flips", labelpad=1)
    ax.set_ylabel("chosen as more random", labelpad=1)


RSA_DISPLAY = 70  # the Frank & Goodman (2012) game: a word ("ornaments") fits two of three objects
def _tree(ax, ornaments, lights):
    """A Christmas tree in the house icons' style (icons.py): a grey tree, ornaments as open rings, lights as
    small dark dots."""
    from matplotlib.patches import Circle, Polygon, Rectangle
    from scripts.paper.figures.icons import GREY
    ax.add_patch(Rectangle((-0.12, -1.0), 0.24, 0.3, facecolor=INK_2, edgecolor="none"))
    ax.add_patch(Polygon([(-0.8, -0.7), (0.8, -0.7), (0, 1.0)], facecolor=GREY, edgecolor="none"))
    if ornaments:
        for x, y in ((-0.35, -0.45), (0.3, -0.35), (0.0, 0.05), (-0.12, 0.5)):
            ax.add_patch(Circle((x, y), 0.11, facecolor="white", edgecolor=INK, lw=0.6))
    if lights:
        for x, y in ((-0.55, -0.6), (-0.2, -0.15), (0.25, 0.1), (0.5, -0.6), (0.12, 0.55), (-0.3, 0.2)):
            ax.add_patch(Circle((x, y), 0.06, facecolor=INK, edgecolor="none"))


def classic_rsa(ax, rsa, cols, trials):
    """Frank & Goodman (2012): one display, the share choosing each object (people, the best seed, the best
    winner). People pick the object whose only feature is the word's: a speaker would have named the other
    object's second feature."""
    d = rsa[rsa.display == RSA_DISPLAY].sort_values("choice_class")
    names = json.loads(trials.loc[trials.display == RSA_DISPLAY, "feature_names"].iloc[0])
    objects = json.loads(d.objects.iloc[0])
    o = (d.observed_count / d.n_trials).to_numpy()
    x = np.arange(len(d))
    w = 0.26
    ax.bar(x - w, o, w, color=HUMAN, zorder=2)
    ax.errorbar(x - w, o, yerr=1.96 * np.sqrt(o * (1 - o) / d.n_trials), fmt="none", ecolor=INK_2, elinewidth=0.7,
                zorder=3)
    for dx, c, color in ((0, cols[0], SEED), (w, cols[1], BEST)):
        ax.bar(x + dx, d[c], w, color=color, zorder=2)
    feats = [[f for f, on in zip(names, objects[json.loads(m)[0]]) if on] for m in d.member_objects]
    ax.set_xticks(x, ["\n+ ".join(f) for f in feats], fontsize=6, va="top")
    ax.tick_params(axis="x", length=0, pad=22)
    ax.set(xlim=(-0.6, len(x) - 0.4), ylim=(0, 1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("chosen", labelpad=1)
    fig = ax.figure
    for xi, f in zip(x, feats):  # each object, drawn under its bars
        x0 = fig.transFigure.inverted().transform(ax.transData.transform((xi, 0)))[0]
        icon = fig.add_axes((x0 - 0.02, ax.get_position().y0 - 0.05, 0.04, 0.045))
        icon.set(xlim=(-1, 1), ylim=(-1.05, 1.05), aspect="equal")
        icon.axis("off")
        _tree(icon, "ornaments" in f, "lights" in f)
    ax.text(0.98, 0.96, f"the word: \u201c{names[0]}\u201d", transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5)


NG_SETS = ("100", "80,70,90,100")  # one example, then more: the data's richest such chain (10, 45 targets)


def classic_ng(cell, fig, ng, cols):
    """Tenenbaum (2000): the generalization profile, P(yes) for each target from 1 to 100, sharpening as the
    examples grow from one (broad) to several fitting a rule (a spike at each number it fits). People, the best
    seed and the best winner side by side at each target."""
    axes = cell.subgridspec(len(NG_SETS), 1, hspace=0.45).subplots(sharex=True)
    for ax, ex in zip(axes, NG_SETS):
        t = groups(ng[ng.number_set == ex], "target", cols)
        for dx, c, color in ((-0.9, "observed", HUMAN), (0, cols[0], SEED), (0.9, cols[1], BEST)):
            ax.bar(t.index + dx, t[c], width=0.9, color=color)
        ax.set(xlim=(0, 102), ylim=(0, 1), yticks=[0, 0.5, 1])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title("examples: " + ", ".join(sorted(ex.split(","), key=int)), fontsize=6, loc="left", pad=2)
    axes[0].tick_params(axis="x", length=0)
    axes[-1].set(xticks=[1, 25, 50, 75, 100])
    axes[-1].set_xlabel("target", labelpad=1)
    axes[0].set_ylabel("said yes", labelpad=1, y=-0.2)
    return axes[0]


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
                          wspace=0.18, hspace=0.68, left=0.075, right=0.99, top=0.9, bottom=0.11)
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
        if d == "number_game":
            classic_ng(gs[1, 3 * k:3 * k + 2], fig, tables[d], cols)
        else:
            ax = fig.add_subplot(gs[1, 3 * k:3 * k + 2])
            if d == "rsa":
                rsa_ax = ax  # its icons are placed once the layout is final
            else:
                classic_sr(ax, tables[d], cols)
        _effect(fig.add_subplot(gs[2, 3 * k:3 * k + 2]), *eff[d][:1], cols, *eff[d][1:])
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
    classic_rsa(rsa_ax, tables["rsa"], list(best(tables["rsa"], "rsa").values()), tables["rsa_trials"])
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    for row, label in enumerate(ROWS):  # over the row's highest text (row A: the domains' titles)
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
