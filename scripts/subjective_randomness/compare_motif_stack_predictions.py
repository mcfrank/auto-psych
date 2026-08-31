"""How different are motif_stack_softmax's predictions from the Viterbi motif_stack?

The softmax rewrite changes what the model *means* (the exact marginal
likelihood instead of the single best path/method), so its predictions differ.
This quantifies by how much, with no MCMC — it evaluates both pure-Python twins
directly.

Two views:

* per-sequence regularity: the randomness score of every H/T sequence over
  lengths 4-8, comparing ``score_max`` vs ``score_softmax`` (softmax <= max by
  construction), and how well the two rank sequences the same way;
* per-choice prediction: ``p_left`` for every same-length pair (the actual
  observable the likelihood is built on), reporting |Δp_left| and how often the
  two models pick a different "more random" option (a decision flip).

Both are reported at the default parameters and, for robustness, aggregated over
random parameter draws.  A per-sequence breakdown by the number of active
production methods (memory flags) shows *where* the difference concentrates.

Usage (Sherlock dev node, NOT login):
    eltest_venv3/bin/python \\
        scripts/subjective_randomness/compare_motif_stack_predictions.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import tyro

from src.subjective_randomness.model_families import motif_stack as viterbi
from src.subjective_randomness.model_families import motif_stack_softmax as softmax


@dataclass
class Args:
    """Quantify the prediction gap between the Viterbi and softmax motif_stack."""

    lengths: List[int] = field(default_factory=lambda: [4, 5, 6, 7, 8])
    """Sequence lengths to enumerate exhaustively."""
    n_param_draws: int = 100
    """Random parameter vectors for the robustness aggregate (0 = defaults only)."""
    betas: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    """beta values at which to report the default-params p_left gap (beta scales
    the score gap into the choice, so the p_left difference depends on it)."""
    seed: int = 0
    """Seed for the random parameter draws."""


def _sequences(length: int) -> List[str]:
    return ["".join("HT"[(i >> bit) & 1] for bit in range(length)) for i in range(2**length)]


def _pair_p_left(
    scores: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray, beta: float, side_bias: float
) -> np.ndarray:
    """p_left for every pair, from precomputed per-sequence scores (beta=slope)."""
    return 1.0 / (1.0 + np.exp(-(beta * (scores[idx_a] - scores[idx_b]) + side_bias)))


def _pair_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    pairs = np.array(list(combinations(range(n), 2)), dtype=np.int64)
    return pairs[:, 0], pairs[:, 1]


def _scores(model, sequences: List[str], params: Dict[str, float]) -> np.ndarray:
    return np.array([model.score_sequence(s, params) for s in sequences], dtype="float64")


def _describe(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "rms": float(np.sqrt(np.mean(x**2))),
    }


def _n_active_methods(seq: str) -> int:
    """Number of memory production methods that fire for ``seq`` (0-3).

    Repetition always applies, so the *total* number of production methods is
    this + 1; this counts only the memory methods, which is what distinguishes a
    plain sequence (max == a single method) from one where the mixture-vs-max
    difference can bite.
    """
    return sum(viterbi._memory_patterns(seq).values())


def analyse_defaults(args: Args) -> None:
    params = dict(viterbi.DEFAULT_PARAMS)  # same names/values for both models
    print("=== Per-sequence regularity score (default params) ===")
    print(f"{'len':>4}{'#seq':>7}{'mean|Δscore|':>14}{'max|Δscore|':>13}{'Spearman ρ':>13}{'Pearson r':>12}")
    all_dscore, all_active = [], []
    for length in args.lengths:
        seqs = _sequences(length)
        sm = _scores(viterbi, seqs, params)
        ss = _scores(softmax, seqs, params)
        d = sm - ss  # >= 0 by construction (softmax score <= max score)
        # Spearman via rank correlation.
        rho = np.corrcoef(_rank(sm), _rank(ss))[0, 1]
        r = np.corrcoef(sm, ss)[0, 1]
        print(f"{length:>4}{len(seqs):>7}{np.mean(np.abs(d)):>14.4f}{np.max(np.abs(d)):>13.4f}{rho:>13.5f}{r:>12.5f}")
        all_dscore.append(d)
        all_active.append(np.array([_n_active_methods(s) for s in seqs]))

    dscore = np.concatenate(all_dscore)
    active = np.concatenate(all_active)
    print("\nΔscore (max − softmax) by number of active memory methods:")
    print(f"{'#active':>8}{'#seq':>8}{'mean Δscore':>14}{'max Δscore':>13}")
    for k in sorted(set(active.tolist())):
        m = active == k
        print(f"{k:>8}{int(m.sum()):>8}{float(np.mean(dscore[m])):>14.4f}{float(np.max(dscore[m])):>13.4f}")

    print("\n=== Per-choice p_left gap over ALL same-length pairs (default params) ===")
    print(f"{'beta':>6}{'mean|Δp|':>11}{'median':>9}{'p95':>9}{'max':>9}{'rms':>9}{'Pearson r':>12}{'flip %':>9}")
    for beta in args.betas:
        dabs, r, flip, npairs = _pair_gap_over_pool(args.lengths, params, beta, params["side_bias"])
        s = _describe(dabs)
        print(
            f"{beta:>6.1f}{s['mean']:>11.4f}{s['median']:>9.4f}{s['p95']:>9.4f}"
            f"{s['max']:>9.4f}{s['rms']:>9.4f}{r:>12.5f}{100 * flip:>8.2f}%"
        )
    print(f"(pooled over {npairs} same-length pairs across lengths {args.lengths})")


def _pair_gap_over_pool(
    lengths: List[int], params: Dict[str, float], beta: float, side_bias: float
) -> Tuple[np.ndarray, float, float, int]:
    """|Δp_left|, Pearson r, and decision-flip rate pooled over all same-length pairs."""
    dabs_parts, pmax_parts, psm_parts = [], [], []
    flips = 0
    npairs = 0
    for length in lengths:
        seqs = _sequences(length)
        idx_a, idx_b = _pair_indices(len(seqs))
        sm = _scores(viterbi, seqs, params)
        ss = _scores(softmax, seqs, params)
        p_max = _pair_p_left(sm, idx_a, idx_b, beta, side_bias)
        p_sm = _pair_p_left(ss, idx_a, idx_b, beta, side_bias)
        dabs_parts.append(np.abs(p_max - p_sm))
        pmax_parts.append(p_max)
        psm_parts.append(p_sm)
        # A decision flip = the two models put p_left on opposite sides of 0.5.
        flips += int(np.sum((p_max - 0.5) * (p_sm - 0.5) < 0))
        npairs += len(idx_a)
    dabs = np.concatenate(dabs_parts)
    r = float(np.corrcoef(np.concatenate(pmax_parts), np.concatenate(psm_parts))[0, 1])
    return dabs, r, flips / npairs, npairs


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype="float64")
    ranks[order] = np.arange(len(x))
    return ranks


def analyse_random_params(args: Args) -> None:
    if args.n_param_draws <= 0:
        return
    rng = np.random.default_rng(args.seed)
    print(f"\n=== Robustness: |Δp_left| aggregated over {args.n_param_draws} random param draws ===")
    means, p95s, maxes = [], [], []
    for _ in range(args.n_param_draws):
        params = {
            name: float(rng.uniform(low, high))
            for name, (low, high) in viterbi.PARAM_BOUNDS.items()
        }
        dabs, _, _, _ = _pair_gap_over_pool(
            args.lengths, params, params["beta"], params["side_bias"]
        )
        means.append(float(np.mean(dabs)))
        p95s.append(float(np.percentile(dabs, 95)))
        maxes.append(float(np.max(dabs)))
    print(f"  mean  |Δp_left| across draws: mean {np.mean(means):.4f}, range [{np.min(means):.4f}, {np.max(means):.4f}]")
    print(f"  p95   |Δp_left| across draws: mean {np.mean(p95s):.4f}, range [{np.min(p95s):.4f}, {np.max(p95s):.4f}]")
    print(f"  max   |Δp_left| across draws: mean {np.mean(maxes):.4f}, worst {np.max(maxes):.4f}")


def main(args: Args) -> None:
    analyse_defaults(args)
    analyse_random_params(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
