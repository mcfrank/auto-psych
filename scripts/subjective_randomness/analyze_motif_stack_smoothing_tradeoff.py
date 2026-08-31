"""How little can motif_stack's predictions move while still being smoothed?

Sweeps the smoothing temperature ``tau`` of ``motif_stack_tempered`` and reports
how far its choice predictions move away from the hard-max ``motif_stack``, over
the exhaustive same-length pair pool.  ``tau -> 0`` is the Viterbi model itself
(zero change); ``tau = 1`` is the full marginal ``motif_stack_softmax``.  The
question this answers is how small a ``tau`` still removes the kinks while
keeping predictions close.

Three softening modes are compared, because the two maxes need not be softened
equally:

* ``both``        — path and method temperatures both = tau;
* ``path-only``   — soften the hidden-path max only (method stays hard max);
* ``method-only`` — soften the production-method max only (paths stay Viterbi).

An earlier decomposition found the hidden-path max drives most of the prediction
change, so ``method-only`` is expected to move predictions least for a given
tau.  This is a prediction-fidelity sweep only (pure-Python twins, no MCMC); the
fitting-difficulty payoff at a chosen tau is measured separately by
``compare_motif_stack_fit_difficulty.py``.

Usage (Sherlock dev node, NOT login):
    eltest_venv3/bin/python \\
        scripts/subjective_randomness/analyze_motif_stack_smoothing_tradeoff.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tyro
from pyprojroot import here

from src.subjective_randomness.model_families import motif_stack as viterbi
from src.subjective_randomness.model_families import motif_stack_tempered as tempered

MODES = {
    "both": lambda tau: (tau, tau),
    "path-only": lambda tau: (tau, 0.0),
    "method-only": lambda tau: (0.0, tau),
}


@dataclass
class Args:
    """Sweep tau and report prediction drift from the hard-max motif_stack."""

    lengths: List[int] = field(default_factory=lambda: [4, 5, 6, 7, 8])
    taus: List[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5, 0.7, 1.0]
    )
    out_dir: Path = Path("data/analysis/motif_stack_smoothing_tradeoff")


def _sequences(length: int) -> List[str]:
    return [
        "".join("HT"[(i >> bit) & 1] for bit in range(length)) for i in range(2**length)
    ]


def _pair_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    pairs = np.array(list(combinations(range(n), 2)), dtype=np.int64)
    return pairs[:, 0], pairs[:, 1]


def _p_left(scores: np.ndarray, ia, ib, beta: float, side_bias: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(beta * (scores[ia] - scores[ib]) + side_bias)))


def _tempered_scores(seqs, params, path_t, method_t) -> np.ndarray:
    return np.array(
        [
            tempered.score_sequence(
                s, params, path_temperature=path_t, method_temperature=method_t
            )
            for s in seqs
        ],
        dtype="float64",
    )


def main(args: Args) -> None:
    params = dict(viterbi.DEFAULT_PARAMS)
    beta, side_bias = params["beta"], params["side_bias"]

    # Per-length pool + the Viterbi baseline p_left, computed once.
    pool = {}
    for length in args.lengths:
        seqs = _sequences(length)
        ia, ib = _pair_indices(len(seqs))
        viterbi_scores = np.array(
            [viterbi.score_sequence(s, params) for s in seqs], dtype="float64"
        )
        pool[length] = {
            "seqs": seqs,
            "ia": ia,
            "ib": ib,
            "viterbi_p": _p_left(viterbi_scores, ia, ib, beta, side_bias),
        }
    n_pairs = sum(len(pool[L]["ia"]) for L in args.lengths)

    rows: List[Dict[str, float]] = []
    print(
        f"Prediction drift from hard-max motif_stack over {n_pairs} same-length "
        f"pairs (default params, beta={beta}):\n"
    )
    for mode, temps in MODES.items():
        print(f"--- mode: {mode} ---")
        print(f"{'tau':>6}{'mean|Δp|':>11}{'p95|Δp|':>10}{'max|Δp|':>10}{'flip %':>9}")
        for tau in args.taus:
            path_t, method_t = temps(tau)
            dabs_parts, flips = [], 0
            for length in args.lengths:
                entry = pool[length]
                scores = _tempered_scores(entry["seqs"], params, path_t, method_t)
                p = _p_left(scores, entry["ia"], entry["ib"], beta, side_bias)
                dabs_parts.append(np.abs(p - entry["viterbi_p"]))
                flips += int(np.sum((p - 0.5) * (entry["viterbi_p"] - 0.5) < 0))
            dabs = np.concatenate(dabs_parts)
            row = {
                "mode": mode,
                "tau": tau,
                "mean_abs_dp": float(np.mean(dabs)),
                "p95_abs_dp": float(np.percentile(dabs, 95)),
                "max_abs_dp": float(np.max(dabs)),
                "flip_pct": 100.0 * flips / n_pairs,
            }
            rows.append(row)
            print(
                f"{tau:>6.2f}{row['mean_abs_dp']:>11.4f}{row['p95_abs_dp']:>10.4f}"
                f"{row['max_abs_dp']:>10.4f}{row['flip_pct']:>8.2f}%"
            )
        print()

    out_dir = here() / args.out_dir if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "smoothing_tradeoff.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["mode", "tau", "mean_abs_dp", "p95_abs_dp", "max_abs_dp", "flip_pct"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")
    print(
        "\nReference: the full marginal (motif_stack_softmax, = tau 1 'both') moves "
        "predictions by mean|Δp|~0.11 and flips ~9.8% of pairs at default params."
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
