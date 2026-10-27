"""Backward elimination over one sr-oct26 final's mechanisms, by CV grouped by stimulus pair.

    python -m src.subjective_randomness.ablation.eliminate --final run1 --data <pooled.csv> --out <dir> --mode plan
    python -m src.subjective_randomness.ablation.eliminate --final run1 --data <pooled.csv> --out <dir> --mode worker --task <i>

The rule (docs/sr_ablation.md, fixed before any fit; the RSA consolidation's,
docs/auto_rsa/CONSOLIDATION.md §2, on one final at a time):

1. Start from the final with every mechanism on (`src.subjective_randomness.ablation.finals`).
2. Each round, score the current model and every model with one mechanism
   removed by ELPD from CV grouped by stimulus pair (`src.model_comparison.cv`:
   both presentations of a pair in one fold; each held-out trial scored by its
   log posterior predictive density). Only removals that leave a valid model
   are candidates (`variants.removals`): person-level variation goes before
   the parameter it varies.
3. Of the converged candidates, take the one with the highest ELPD-CV. Its
   cost is the current model's ELPD-CV minus its own; its SE is clustered by
   stimulus pair. If the cost is at most 0 or below ``threshold_dse`` (2) SEs,
   that mechanism goes, and the next round starts from the smaller model.
4. Stop when every removal costs at least ``threshold_dse`` SEs (or none
   converges, or nothing is left to remove): the current model is the lean model.

The starting model must converge (on every fold); the elimination raises
otherwise (the RSA run costed its first drop against an unconverged start).
Converged means the repository's gate on every population-level parameter
(``gate="population"``; person-level vectors are reported, not gated: on the
pooled data one person's posterior can be two-humped, scripts/paper/sr_figure_data.py).

The work is one Slurm task per (model, fold) (`scripts/subjective_randomness/slurm/ablation.sbatch`):
``plan`` replays the rule over the saved fold results and writes ``next.json``
with the (model, fold) fits the next round still needs, or, when none is
missing, ``elimination.json``, ``summary.md`` and ``lean/``. ``worker`` fits
task ``task`` of ``next.json`` at the first of ``seeds`` that converges
within ``time_limit_hours`` and saves its held-out log densities
(``cv/<model>/fold<k>.npz``) and every attempt (``fold<k>.json``). A fold
that converged at no seed is unconverged; adding seeds and planning again
retries it at the new ones only.

The rule, ``plan`` and ``worker`` are `src.ablation.eliminate`'s; this module
names the finals, the CV unit (the stimulus pair) and the SE's clusters (the
stimulus pair).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Sequence, Tuple

import numpy as np
import tyro

from src.ablation import eliminate as E
from src.ablation.eliminate import Scored, _fold_needed  # noqa: F401 (this module's names)
from src.subjective_randomness.ablation import variants as V

THRESHOLD_DSE = E.THRESHOLD_DSE
SEEDS = (20261018, 6)


def _stimulus_clusters(rows):
    from src.models.clustered_se import stimulus_clusters

    return stimulus_clusters(rows)


ABLATION = E.Ablation(variants=V.VARIANTS, cv_group="stimulus", units=_stimulus_clusters)


def eliminate(final: str, start: Sequence[str], score: Callable[[List[Tuple[str, ...]]], Dict[Tuple[str, ...], Scored]],
              units: np.ndarray, threshold_dse: float = THRESHOLD_DSE) -> dict:
    """The rule above. ``score`` maps mechanism sets to their CV."""
    return E.eliminate(V.VARIANTS, final, start, score, units, threshold_dse)


@dataclass
class Args(E.Args):
    final: Literal["run1", "run2", "run3"]
    data: Path
    """All runs' pooled data (`src.subjective_randomness.ablation.pool`)."""
    seeds: List[int] = field(default_factory=lambda: list(SEEDS))
    """worker: sampler seeds, tried in order until a fit converges (not part of the config: may grow)."""


def plan(args: Args) -> dict:
    """Replay the rule over the saved fold results: the next round's missing fits, or the result."""
    return E.plan(ABLATION, args)


def summary_markdown(record: dict) -> str:
    return E.summary_markdown(V.VARIANTS, record)


def worker(args: Args) -> dict:
    """Fit task ``task`` of ``next.json`` and save its held-out log densities."""
    return E.worker(ABLATION, args)


def main(args: Args):
    return plan(args) if args.mode == "plan" else worker(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
