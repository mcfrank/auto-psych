#!/usr/bin/env python3
"""Analyze how well a full outer-loop run did, and plot it.

Reads the artifacts each experiment's inner model loop wrote
(`model_loop/model_posterior.json`, `model_loop/history.json`) plus the
collected behavioral data (`data/responses.csv`) and produces four figures:

1. model_posterior.png        — final posterior model probability per model
                                 (inner-loop candidates vs. starting models).
2. elpd_distinguishability.png — PSIS-LOO elpd relative to the best model, with
                                 ±dse error bars and the "within 2·dse" rule;
                                 shows whether anything beats the incumbent.
3. loop_trajectory.png         — best-overall vs. best inner-loop-candidate
                                 posterior across baseline → iter0 → iter1.
4. response_bias.png           — per-participant choice rate; a data-quality
                                 check that flags degenerate, stimulus-ignoring
                                 responses (all choices on one side).

Everything is read straight from disk; nothing re-runs MCMC. The script fails
loudly if an expected artifact is missing.

Usage:
    uv run python scripts/subjective_randomness/analyze_loop_results.py
    uv run python scripts/subjective_randomness/analyze_loop_results.py --experiments 1 2
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from pyprojroot import here

REPO_ROOT = here()

# Colors: starting models (project seeds / theory-proposed) vs. models the inner
# loop generated this run (named iterN_candidateM).
COLOR_START = "#4a90d9"
COLOR_CANDIDATE = "#e07b39"
COLOR_WINNER = "#2e8b57"


@dataclass
class Args:
    """Analyze and plot outer-loop results for a project."""

    project: str = "subjective_randomness"
    """Project id under data/outer_loop/."""
    experiments: List[int] = field(default_factory=lambda: [1, 2])
    """Experiment numbers to analyze."""
    out_subdir: str = "analysis"
    """Output dir (under the project's outer-loop data dir) for the figures."""


def _exp_dir(project: str, exp_num: int) -> Path:
    return REPO_ROOT / "data" / "outer_loop" / project / f"experiment{exp_num}"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _is_candidate(model_name: str) -> bool:
    """True for inner-loop-generated candidates (iterN_candidateM)."""
    return model_name.startswith("iter")


def _alternation_proportion(seq: str) -> float:
    """Fraction of adjacent pairs that switch H<->T (the featurizer's p_alts)."""
    s = seq.strip().upper()
    if len(s) < 2:
        return 0.0
    switches = sum(1 for i in range(1, len(s)) if s[i] != s[i - 1])
    return switches / (len(s) - 1)


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion (sane at small n / extremes)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - half, center + half)


# ─────────────────────────────────────────────
# Figure 1: final posterior model probability
# ─────────────────────────────────────────────


def plot_model_posterior(args: Args, posteriors_by_exp: dict, out_path: Path) -> None:
    n = len(posteriors_by_exp)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 5.2), squeeze=False)
    for ax, (exp_num, mp) in zip(axes[0], posteriors_by_exp.items()):
        posteriors = mp["posteriors"]
        items = sorted(posteriors.items(), key=lambda kv: kv[1])  # asc for barh
        names = [k for k, _ in items]
        values = [v for _, v in items]
        best_name = max(posteriors, key=posteriors.get)
        colors = []
        for name in names:
            if name == best_name:
                colors.append(COLOR_WINNER)
            elif _is_candidate(name):
                colors.append(COLOR_CANDIDATE)
            else:
                colors.append(COLOR_START)
        ax.barh(names, values, color=colors)
        for y, v in enumerate(values):
            ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_xlabel("posterior model probability")
        ax.set_title(
            f"Experiment {exp_num}  (n={mp['n_trials']} trials)\n"
            f"winner: {best_name}  (p={posteriors[best_name]:.3f})",
            fontsize=11,
        )
        ax.tick_params(axis="y", labelsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_WINNER),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_START),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_CANDIDATE),
    ]
    fig.legend(
        handles,
        ["winner", "starting model (seed / theory)", "inner-loop candidate"],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Final posterior over cognitive models", fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# Figure 2: ELPD-LOO distinguishability forest
# ─────────────────────────────────────────────


def plot_elpd_distinguishability(args: Args, posteriors_by_exp: dict, out_path: Path) -> None:
    n = len(posteriors_by_exp)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 5.2), squeeze=False)
    for ax, (exp_num, mp) in zip(axes[0], posteriors_by_exp.items()):
        comparison = mp["comparison"]
        # Sort by rank (best first); plot best at top.
        ranked = sorted(comparison.items(), key=lambda kv: kv[1]["rank"])
        names = [k for k, _ in ranked][::-1]
        diffs = [-comparison[k]["elpd_diff"] for k in names]  # 0 = best, worse = left
        dse = [comparison[k]["dse"] for k in names]
        best_name = names[-1]
        for y, name in enumerate(names):
            # Distinguishable from best when elpd_diff > 2*dse.
            elpd_diff = comparison[name]["elpd_diff"]
            distinguishable = elpd_diff > 2 * comparison[name]["dse"]
            color = COLOR_WINNER if name == best_name else (
                COLOR_CANDIDATE if _is_candidate(name) else COLOR_START
            )
            ax.errorbar(
                diffs[y], y, xerr=dse[y], fmt="o",
                color=color, ecolor=color, capsize=3,
                markerfacecolor=(color if distinguishable or name == best_name else "white"),
                markeredgecolor=color, markersize=8, elinewidth=1.5,
            )
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0, color="#2e8b57", ls="--", lw=1, label="best model")
        n_distinguishable = sum(
            1 for k in comparison
            if comparison[k]["elpd_diff"] > 2 * comparison[k]["dse"]
        )
        ax.set_xlabel("ELPD-LOO − best  (0 = best; ← worse)")
        ax.set_title(
            f"Experiment {exp_num}: model distinguishability\n"
            f"{n_distinguishable}/{len(comparison)-1} models clearly worse than best "
            f"(elpd_diff > 2·dse)",
            fontsize=11,
        )
        ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle(
        "PSIS-LOO predictive accuracy (open marker = indistinguishable from best)",
        fontsize=13, y=1.0,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# Figure 3: loop trajectory across iterations
# ─────────────────────────────────────────────


def plot_loop_trajectory(args: Args, history_by_exp: dict, out_path: Path) -> None:
    n = len(history_by_exp)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.0), squeeze=False)
    for ax, (exp_num, history) in zip(axes[0], history_by_exp.items()):
        steps = [h["step"] for h in history]
        best_overall = []
        best_candidate = []
        winner_names = []
        for h in history:
            post = h["posteriors"]
            winner = max(post, key=post.get)
            winner_names.append(winner)
            best_overall.append(post[winner])
            cand_posts = [v for k, v in post.items() if _is_candidate(k)]
            best_candidate.append(max(cand_posts) if cand_posts else np.nan)
        ax.plot(steps, best_overall, "-o", color=COLOR_WINNER, lw=2,
                label="best model overall")
        ax.plot(steps, best_candidate, "-s", color=COLOR_CANDIDATE, lw=2,
                label="best inner-loop candidate")
        for x, y, name in zip(steps, best_overall, winner_names):
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7.5, color=COLOR_WINNER)
        ax.set_xticks(steps)
        ax.set_xticklabels(["baseline\n(start models)", "after\niter 0", "after\niter 1"][: len(steps)])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("posterior model probability")
        ax.set_title(f"Experiment {exp_num}", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Did the inner loop's generated models overtake the starting models?",
        fontsize=13, y=1.0,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# Figure 4: response-bias diagnostic (data quality)
# ─────────────────────────────────────────────


def plot_response_bias(args: Args, responses_by_exp: dict, out_path: Path) -> None:
    """Per-participant choice rate. A spread around 0.5 is healthy; everyone
    pinned at 0 or 1 means the simulated participants ignored the stimulus."""
    n = len(responses_by_exp)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.0), squeeze=False)
    for ax, (exp_num, df) in zip(axes[0], responses_by_exp.items()):
        per_p = df.groupby("participant_id")["chose_left"].mean()
        overall = df["chose_left"].mean()
        n_right = int(df["chose_right"].sum())
        degenerate = df["chose_left"].nunique() <= 1
        bar_color = "#c0392b" if degenerate else COLOR_START
        ax.bar([str(p) for p in per_p.index], per_p.values, color=bar_color, alpha=0.85)
        ax.axhline(0.5, color="gray", ls="--", lw=1.2, label="unbiased (0.5)")
        ax.axhline(overall, color="black", ls=":", lw=1.2,
                   label=f"overall = {overall:.2f}")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("participant")
        ax.set_ylabel("P(chose left / sequence A)")
        flag = "  ⚠ DEGENERATE" if degenerate else ""
        ax.set_title(
            f"Experiment {exp_num}  (n={len(df)} trials){flag}\n"
            f"chose 'right' on {n_right}/{len(df)} trials",
            fontsize=11,
            color="#c0392b" if degenerate else "black",
        )
        ax.legend(loc="lower center", fontsize=8, frameon=False)
    fig.suptitle(
        "Data-quality check: per-participant choice rate "
        "(all pinned at 1.0 ⇒ participants always chose A, ignoring the stimulus)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(args: Args) -> None:
    posteriors_by_exp: dict[int, dict] = {}
    history_by_exp: dict[int, list] = {}
    responses_by_exp: dict[int, pd.DataFrame] = {}

    for exp_num in args.experiments:
        exp_dir = _exp_dir(args.project, exp_num)
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")
        posteriors_by_exp[exp_num] = _load_json(exp_dir / "model_loop" / "model_posterior.json")
        history_by_exp[exp_num] = _load_json(exp_dir / "model_loop" / "history.json")
        responses_path = exp_dir / "data" / "responses.csv"
        if not responses_path.exists():
            raise FileNotFoundError(f"Required artifact missing: {responses_path}")
        responses_by_exp[exp_num] = pd.read_csv(responses_path)

    out_dir = REPO_ROOT / "data" / "outer_loop" / args.project / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_model_posterior(args, posteriors_by_exp, out_dir / "model_posterior.png")
    plot_elpd_distinguishability(args, posteriors_by_exp, out_dir / "elpd_distinguishability.png")
    plot_loop_trajectory(args, history_by_exp, out_dir / "loop_trajectory.png")
    plot_response_bias(args, responses_by_exp, out_dir / "response_bias.png")

    # Console summary.
    print(f"\nWrote 4 figures to {out_dir}\n")
    for exp_num, mp in posteriors_by_exp.items():
        best = max(mp["posteriors"], key=mp["posteriors"].get)
        n_dist = sum(
            1 for k in mp["comparison"]
            if mp["comparison"][k]["elpd_diff"] > 2 * mp["comparison"][k]["dse"]
        )
        n_cand = sum(1 for k in mp["posteriors"] if _is_candidate(k))
        best_is_candidate = _is_candidate(best)
        df = responses_by_exp[exp_num]
        p_left = df["chose_left"].mean()
        degenerate = df["chose_left"].nunique() <= 1
        print(
            f"Experiment {exp_num}: best={best} (p={mp['posteriors'][best]:.3f}, "
            f"elpd={mp['elpd_loo'][best]:.2f}); "
            f"{n_cand} inner-loop candidates; "
            f"best is {'an inner-loop candidate' if best_is_candidate else 'a starting model'}; "
            f"{n_dist}/{len(mp['comparison'])-1} models clearly worse than best; "
            f"P(chose_left)={p_left:.3f}"
            f"{'  ⚠ DEGENERATE DATA (no response variation)' if degenerate else ''}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
