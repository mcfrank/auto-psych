"""Draft of the restyled Figure 3 (throwaway): per domain, (A) calibration, (B) a behavioural
signature, (C) the conditions where the discovered model gains most over the best seed.

Run: uv run python -m data.paper.work.fig3_draft.fig3_draft
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from pyprojroot import here  # noqa: E402

from data.paper.work.fig2_draft.fig2_draft import INK, INK_2, TITLE_FONT, setup_style  # noqa: E402
from data.paper.work.icons.icons import STIMULI  # noqa: E402

OUT = here() / "data" / "paper" / "work" / "fig3_draft"
SEED, BEST, HUMAN = "#b3b3b3", "#2171b5", INK  # best seed, best discovered model, human data
TITLES = {"subjective_randomness": "Subjective randomness", "rsa": "Reference games", "number_game": "Number game"}
SMALL = 5.5


def binned(x, y, w, edges):
    """Weighted mean of y (and its binomial 95% CI over the bin's trials) in bins of x."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi) if hi < edges[-1] else (x >= lo) & (x <= hi)
        if w[m].sum() == 0:
            continue
        mean = np.average(y[m], weights=w[m])
        out.append((np.average(x[m], weights=w[m]), mean, 1.96 * np.sqrt(mean * (1 - mean) / w[m].sum())))
    return np.array(out)


def weighted_r_rmse(o, p, n):
    w = n / n.sum()
    mo, mp = (w * o).sum(), (w * p).sum()
    r = (w * (o - mo) * (p - mp)).sum() / np.sqrt((w * (o - mo) ** 2).sum() * (w * (p - mp) ** 2).sum())
    return r, np.sqrt((w * (o - p) ** 2).sum())


def _axes_style(ax):
    ax.tick_params(labelsize=SMALL, length=2, pad=1.5)


# --- row A: calibration --------------------------------------------------------

def calibration(ax, o, n, preds):
    """Observed against predicted, conditions binned by each model's predicted P (deciles)."""
    ax.plot([0, 1], [0, 1], color="#e0e0e0", lw=0.8, zorder=0)
    for k, (name, p, color) in enumerate(preds):
        edges = np.quantile(p, np.linspace(0, 1, 11))
        b = binned(p, o, n, edges)
        ax.errorbar(b[:, 0], b[:, 1], yerr=b[:, 2], color=color, lw=1.1, marker="o", ms=2.5, elinewidth=0.8,
                    capsize=0, zorder=2 + k)
        r, rmse = weighted_r_rmse(o, p, n)
        ax.text(0.03, 0.95 - 0.09 * k, f"r = {r:.2f}, RMSE = {rmse:.3f}", transform=ax.transAxes, fontsize=SMALL,
                color=color if color != SEED else "#8c8c8c", va="top")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("predicted", fontsize=SMALL + 0.5, labelpad=1)
    _axes_style(ax)


# --- subjective randomness -------------------------------------------------------

def alternation(seq):
    return sum(a != b for a, b in zip(seq, seq[1:])) / (len(seq) - 1)


def _per_sequence(rows):
    """Each sequence's trials and its chance of being chosen as more random, observed and
    under each model, over every pair it appeared in."""
    per = {}
    for r in rows:
        for seq, first in ((r["pair"][0], True), (r["pair"][1], False)):
            d = per.setdefault(seq, {"n": 0, "observed": 0, "best_seed": 0, "best_model": 0})
            d["n"] += r["n"]
            for k in ("observed", "best_seed", "best_model"):
                d[k] += r["n"] * (r[k] if first else 1 - r[k])
    return {seq: {k: (v / d["n"] if k != "n" else v) for k, v in d.items()} for seq, d in per.items()}


def symmetry(seq):
    if seq == seq[::-1].translate(str.maketrans("HT", "TH")):
        return "mirror-complement"
    if seq == seq[::-1]:
        return "palindrome"
    return "other"


def sr_classic(ax, rows):
    """The domain's classic effect, fixed in advance: over-alternation (Falk & Konold, 1997), sequences
    that switch a little more often than a fair coin look most random. Each sequence's chance of being
    chosen, binned by its alternation rate."""
    per = _per_sequence(rows)
    seqs = list(per)
    alt = np.array([alternation(q) for q in seqs])
    n = np.array([per[q]["n"] for q in seqs], float)
    edges = np.linspace(0, 1, 8)
    for key, color in (("best_seed", SEED), ("best_model", BEST)):
        b = binned(alt, np.array([per[q][key] for q in seqs]), n, edges)
        ax.plot(b[:, 0], b[:, 1], color=color, lw=1.3, zorder=2)
    b = binned(alt, np.array([per[q]["observed"] for q in seqs]), n, edges)
    ax.errorbar(b[:, 0], b[:, 1], yerr=b[:, 2], fmt="o", color=HUMAN, ms=2.5, elinewidth=0.8, capsize=0, zorder=3)
    ax.axhline(0.5, color="#e0e0e0", lw=0.8, zorder=0)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("alternation rate of the sequence", fontsize=SMALL + 0.5, labelpad=1)
    ax.set_ylabel("P(chosen as more random)", fontsize=SMALL + 0.5, labelpad=1)
    _axes_style(ax)


def sr_new(ax, rows):
    """The effect only the discovered model captures: the stimulus feature that best carries its
    improvement over the seed (`regret.rank_features` over `regret.SR_FEATURES`), sequences grouped by
    it. People: points with 95% CIs; the models: ticks. Under each group, the sequence where the
    discovered model gains most."""
    from data.paper.work.fig3_draft.regret import SR_FEATURES, rank_features
    per = _per_sequence(rows)
    items = [{"seq": q, **d} for q, d in per.items()]
    top = rank_features(items, lambda i: i["seq"], SR_FEATURES)[0]
    if not top["binary"]:
        raise NotImplementedError(f"the top feature {top['feature']!r} is not yes/no: bin it")
    f = SR_FEATURES[top["feature"]]
    groups = (("other", 0.0), (top["feature"], 1.0))
    lo, hi = 0.38, 0.6
    for i, (label, value) in enumerate(groups):
        members = [it for it in items if f(it["seq"]) == value]
        n = np.array([it["n"] for it in members], float)
        mean = {k: np.average([it[k] for it in members], weights=n) for k in ("observed", "best_seed", "best_model")}
        ci = 1.96 * np.sqrt(mean["observed"] * (1 - mean["observed"]) / n.sum())
        ax.plot([i - 0.22, i + 0.22], [mean["best_seed"]] * 2, color=SEED, lw=1.8, zorder=2, solid_capstyle="butt")
        ax.plot([i - 0.22, i + 0.22], [mean["best_model"]] * 2, color=BEST, lw=1.8, zorder=3, solid_capstyle="butt")
        ax.errorbar(i, mean["observed"], yerr=ci, fmt="o", color=HUMAN, ms=3, elinewidth=0.8, capsize=0, zorder=4)
        ax.text(i, lo - 0.012, f"{label} ({len(members)})", ha="center", va="top", fontsize=SMALL - 0.5, color=INK_2)
        # the group's example: the sequence where the discovered model gains most over the seed
        example = max(members, key=lambda it: _loglik(it["observed"], it["best_model"], it["n"])
                      - _loglik(it["observed"], it["best_seed"], it["n"]))["seq"]
        step = 0.075
        _coins(ax, i - step * (len(example) - 1) / 2, lo - 0.045, example, step, size=6)
    ax.axhline(0.5, color="#e0e0e0", lw=0.8, zorder=0)
    ax.set_xlim(-0.6, len(groups) - 0.4); ax.set_ylim(lo, hi)
    ax.set_xticks([]); ax.set_yticks([0.4, 0.5, 0.6])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("P(chosen as more random)", fontsize=SMALL + 0.5, labelpad=1)
    _axes_style(ax)


def _loglik(o, p, n):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return n * (o * np.log(p) + (1 - o) * np.log(1 - p))


def _coins(ax, x0, y, seq, step, size=9):
    """A coin sequence as round markers (heads filled), so it stays round in any axes."""
    xs = x0 + step * np.arange(len(seq))
    ax.scatter(xs, [y] * len(seq), s=size, marker="o", linewidths=0.5, edgecolors=INK_2, clip_on=False,
               facecolors=[INK_2 if c == "H" else "white" for c in seq], zorder=3)


def sr_examples(ax, rows, k=3):
    """The k pairs where the discovered model's log-likelihood gains most over the seed's:
    each drawn, with P(choose the top sequence) observed and under each model."""
    gain = [_loglik(r["observed"], r["best_model"], r["n"]) - _loglik(r["observed"], r["best_seed"], r["n"])
            for r in rows]
    top = [rows[i] for i in np.argsort(gain)[::-1][:k]]
    ax.set_xlim(0, k); ax.set_ylim(0, 1)
    ax.axis("off")
    for j, r in enumerate(top):
        x = j + 0.5
        a, b = r["pair"]
        step = 0.075
        width = (max(len(a), len(b)) - 1) * step
        _coins(ax, x - width / 2, 0.95, a, step)
        _coins(ax, x - width / 2, 0.86, b, step)
        for i, (key, color) in enumerate((("observed", HUMAN), ("best_seed", SEED), ("best_model", BEST))):
            bx = x - 0.22 + i * 0.22
            ax.bar(bx, r[key] * 0.62, width=0.16, bottom=0.08, color=color)
        ax.plot([x - 0.36, x + 0.36], [0.08, 0.08], color=INK_2, lw=0.6)
        ax.plot([x - 0.36, x + 0.36], [0.08 + 0.31, 0.08 + 0.31], color="#e0e0e0", lw=0.6, zorder=0)
    ax.text(0, 0.0, "P(top sequence chosen)", fontsize=SMALL, color=INK_2, va="top")


# --- figure ----------------------------------------------------------------------

def pending(ax):
    ax.axis("off")
    ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center", fontsize=SMALL,
            color=INK_2, style="italic")


def main():
    setup_style()
    sr = json.loads((here() / "data/paper/figure_data/subjective_randomness.json").read_text())["conditions"]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.0), gridspec_kw={"hspace": 0.55, "wspace": 0.35,
                                                                     "height_ratios": [1, 1, 1]})
    o, n = np.array([r["observed"] for r in sr]), np.array([r["n"] for r in sr], float)
    calibration(axes[0, 0], o, n, (("best seed", np.array([r["best_seed"] for r in sr]), SEED),
                                   ("discovered", np.array([r["best_model"] for r in sr]), BEST)))
    axes[0, 0].set_ylabel("observed", fontsize=SMALL + 0.5, labelpad=1)
    sr_classic(axes[1, 0], sr)
    sr_new(axes[2, 0], sr)
    for col in (1, 2):
        for row in range(3):
            pending(axes[row, col])
    for col, d in enumerate(TITLES):
        axes[0, col].set_title(TITLES[d], fontdict=TITLE_FONT, pad=14)
    for row, label in enumerate(("A  overall fit", "B  the classic effect", "C  a new effect")):
        axes[row, 0].text(-0.32, 1.12 if row else 1.03, label, transform=axes[row, 0].transAxes, fontdict=TITLE_FONT, va="bottom")
    key = [Line2D([], [], color=HUMAN, marker="o", ms=2.5, lw=0, label="people"),
           Line2D([], [], color=SEED, lw=1.3, label="best seed model"),
           Line2D([], [], color=BEST, lw=1.3, label="best discovered model")]
    fig.legend(handles=key, loc="upper center", ncol=3, frameon=False, fontsize=SMALL + 0.5,
               bbox_to_anchor=(0.5, 0.06))
    # the stimulus icons, left of the titles
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    for col, d in enumerate(TITLES):
        box = axes[0, col].title.get_window_extent()
        (x0, y0), (x1, y1) = to_fig.transform((box.x0, box.y0)), to_fig.transform((box.x1, box.y1))
        draw, aspect = STIMULI[d]
        h = 1.9 * (y1 - y0)
        w = h * aspect * fig.get_figheight() / fig.get_figwidth()
        draw(fig.add_axes((x0 - w - 0.008, (y0 + y1) / 2 - h / 2, w, h)))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_draft.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
