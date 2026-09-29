"""CLI: lazy batched vs exact greedy joint-EIG selection on a finished design.

The design picks its stimuli by greedy joint EIG over every same-length H/T
pair (43,434 at lengths 2-8). Exact greedy re-scores the whole pool at every
pick; lazy batched greedy (``src/models/eig_selection.py``) re-scores only the
best-ranked candidates between exact full passes, and is an approximation
because joint EIG is not submodular. This script measures the approximation
on the inputs of a real later-experiment design, rebuilt without refitting:

1. It copies the design's inputs out of ``--cell`` (read-only; nothing is
   written there) into ``--work-dir``: ``experiment<N>/cognitive_models``,
   ``experiment<N>/design/_fit_cache`` (the design's posterior fits),
   ``experiment<N>/design/stimuli.json`` and the previous experiment's
   ``model_loop/responses.csv`` and ``model_registry.yaml``.
2. It rebuilds the design's per-draw posterior-predictive ``p_left`` from the
   cached fits (and fails if any fit is missing: it never samples).
3. On a fixed random subsample of the pool (``--subsample`` pairs) it runs
   the design's selection (``eig.select_design_picks``: the n-response
   selection with its noise-floor stop, then the single-response fill) with
   exact greedy in float64 (the reference), exact greedy with other scenarios
   (``--second-seed``: how much exact greedy itself moves), exact greedy in
   float32, and lazy greedy in float32 and float64. With ``--full-pool`` it
   repeats this on the whole pool (exact float64 only with
   ``--full-pool-float64``) and also scores the set the sweep chose.
4. Every set is scored identically out of sample: its joint EIG at
   ``--n-responses`` responses per stimulus on ``--fresh-scenarios`` fresh
   scenarios shared by all sets, with the paired Monte Carlo standard error
   of its difference from the reference. Also reported: the in-sample joint
   EIG, the overlap with the reference, and the run time.
5. With ``--timing`` it times one full scoring pass over the whole pool in
   float64 and float32 on 1 thread and on every allocated CPU.

Writes ``validation.json`` and ``validation.md`` into ``--work-dir``.

Usage:
    python scripts/subjective_randomness/validate_lazy_eig.py \\
        --cell /path/to/_runs/cell_1 \\
        --work-dir $SCRATCH/auto-psych/lazy_eig_validation/<name>
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.models.eig_selection import (  # noqa: E402
    _entropy_bits,
    _model_prior,
    scenario_posterior_entropies,
    select_n_joint_eig,
)
from src.pipelines.outer_loop import eig as eig_mod  # noqa: E402

Picks = List[Tuple[int, float, str]]


@dataclass
class Args:
    """Compare lazy and exact greedy joint-EIG selection on a finished design."""

    cell: Path
    """A finished holdout cell (``_runs/cell_1``); read only."""
    work_dir: Path
    """Where the inputs are copied and the results written."""
    experiment: int = 2
    """The design to rebuild (>= 2: a posterior design)."""
    lengths: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    """Sequence lengths of the design pool (the outer loop's default)."""
    n_select: int = 64
    """Stimuli per design."""
    n_responses: int = 40
    """Responses per stimulus (participants per experiment)."""
    n_samples: int = 200
    """Posterior draws per model, as in the design."""
    n_scenarios: int = 1000
    """Selection scenarios, as in the design."""
    selection_seed: int = 42
    """Seed of the selection scenarios."""
    second_seed: int = 43
    """Seed of a second exact-greedy run: exact greedy's own spread."""
    subsample: int = 5000
    """Pairs in the random subsample of the pool."""
    subsample_seed: int = 0
    """Seed of the subsample."""
    fresh_scenarios: int = 50000
    """Scenarios for scoring every set out of sample."""
    fresh_seed: int = 20260927
    """Seed of the out-of-sample scenarios (distinct from the selection's)."""
    full_pool: bool = True
    """Also compare on the whole pool."""
    full_pool_float64: bool = True
    """Run exact float64 greedy on the whole pool (the slowest run)."""
    timing: bool = True
    """Time one full scoring pass in float64/float32 on 1 and on all CPUs."""
    threads: Optional[int] = None
    """Threads for the selections (None: every allocated CPU)."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def stage_inputs(cell: Path, work_dir: Path, experiment: int) -> Path:
    """Copy the design's inputs from ``cell`` into ``work_dir/inputs``.

    Copies are made once; a later run reuses them. Raises when ``work_dir``
    lies inside ``cell`` (the cell must stay untouched) or an input is missing.
    """
    cell, work_dir = Path(cell).resolve(), Path(work_dir).resolve()
    if work_dir == cell or cell in work_dir.parents:
        raise ValueError(f"--work-dir {work_dir} is inside the cell {cell}; it must not be.")
    if experiment < 2:
        raise ValueError("Only a posterior design (experiment >= 2) has fits to rebuild from.")
    staged = work_dir / "inputs"
    exp, prev = f"experiment{experiment}", f"experiment{experiment - 1}"
    copies = [
        (f"{exp}/cognitive_models", True),
        (f"{exp}/design/_fit_cache", True),
        (f"{exp}/design/stimuli.json", False),
        (f"{prev}/model_loop/responses.csv", False),
        (f"{prev}/model_registry.yaml", False),
    ]
    for rel, is_dir in copies:
        source, target = cell / rel, staged / rel
        if not source.exists():
            raise FileNotFoundError(f"The design input {source} is missing.")
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if is_dir:
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    return staged


def design_draws(
    staged: Path, experiment: int, rows: List[Dict[str, Any]], n_samples: int, cache: Path
) -> Dict[str, np.ndarray]:
    """The design's per-draw posterior-predictive ``p_left`` over ``rows``.

    Rebuilt from the copied fit cache exactly as the design built them
    (``eig._posterior_p_left_draws``), and cached in ``cache`` (.npz). Raises
    when rebuilding would have sampled: a new file in the fit cache means a
    fit was missing, and this script compares searches, never refits.
    """
    if cache.exists():
        with np.load(cache) as saved:
            return {name: saved[name] for name in saved.files}
    exp = staged / f"experiment{experiment}"
    fit_cache = exp / "design" / "_fit_cache"
    before = set(fit_cache.iterdir())
    names = eig_mod._load_model_names(exp / "cognitive_models")
    names, unbindable = eig_mod._screen_usable_models(names, exp / "cognitive_models", rows[0])
    draws, undefined = eig_mod._posterior_p_left_draws(
        names,
        exp / "cognitive_models",
        rows,
        responses_csv=staged / f"experiment{experiment - 1}" / "model_loop" / "responses.csv",
        fit_cache_dir=fit_cache,
        max_draws=n_samples,
        seed=42,
    )
    sampled = sorted(p.name for p in set(fit_cache.iterdir()) - before)
    if sampled:
        raise RuntimeError(
            f"Rebuilding the design sampled new fits {sampled}: the copied fit cache "
            "does not hold the design's fits, so these draws would not be the design's."
        )
    if unbindable or undefined:
        print(f"  [inputs] screened out as in the design: {unbindable + undefined}", flush=True)
    np.savez(cache, **draws)
    return draws


def pool_index_of(pool: Sequence[Dict[str, str]], stimuli: Sequence[Dict[str, Any]]) -> List[int]:
    """Pool indices of ``stimuli`` (the sweep's chosen pairs), in their order."""
    index = {(p["sequence_a"], p["sequence_b"]): i for i, p in enumerate(pool)}
    missing = [s for s in stimuli if (s["sequence_a"], s["sequence_b"]) not in index]
    if missing:
        raise ValueError(f"{len(missing)} chosen stimuli are not in the pool, e.g. {missing[0]}.")
    return [index[(s["sequence_a"], s["sequence_b"])] for s in stimuli]


# ---------------------------------------------------------------------------
# Selections, scores, timings
# ---------------------------------------------------------------------------


def run_selection(
    draws: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]],
    args: Args,
    *,
    lazy: bool,
    dtype: str,
    seed: int,
    threads: int,
) -> Dict[str, Any]:
    """The design's selection under one search setting, timed."""
    start = time.perf_counter()
    picks: Picks = eig_mod.select_design_picks(
        draws,
        args.n_select,
        model_weights=weights,
        n_scenarios=args.n_scenarios,
        seed=seed,
        n_responses=args.n_responses,
        lazy=lazy,
        dtype=dtype,
        n_threads=threads,
    )
    seconds = time.perf_counter() - start
    n_response_picks = [bits for _, bits, source in picks if source == "eig"]
    return {
        "lazy": lazy,
        "dtype": dtype,
        "seed": seed,
        "seconds": round(seconds, 1),
        "indices": [int(i) for i, _, _ in picks],
        "sources": [source for _, _, source in picks],
        "n_response_picks": len(n_response_picks),
        "in_sample_bits": round(n_response_picks[-1], 4) if n_response_picks else None,
    }


def score_sets(
    draws: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]],
    sets: Dict[str, Dict[str, Any]],
    reference: str,
    args: Args,
) -> None:
    """Add out-of-sample joint EIG, its paired difference from ``reference``
    and the overlap with it to every entry of ``sets``, in place.

    Every set is scored on the same fresh scenarios (same models and draws;
    the responses differ with the stimuli), at ``args.n_responses`` responses
    per stimulus, for the whole set and for its n-response picks alone.
    """
    h_prior = float(_entropy_bits(_model_prior(list(draws), weights)))

    def entropies(indices: Sequence[int]) -> np.ndarray:
        return scenario_posterior_entropies(
            draws,
            indices,
            model_weights=weights,
            n_scenarios=args.fresh_scenarios,
            seed=args.fresh_seed,
            n_responses=args.n_responses,
        )

    ref = sets[reference]
    ref_h = entropies(ref["indices"])
    for entry in sets.values():
        h = entropies(entry["indices"])
        diff = ref_h - h  # > 0 where this set leaves less uncertainty
        entry["fresh_bits"] = round(h_prior - float(h.mean()), 4)
        entry["fresh_diff_vs_reference"] = round(float(diff.mean()), 4)
        entry["fresh_diff_se"] = round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 4)
        entry["overlap_with_reference"] = len(set(entry["indices"]) & set(ref["indices"]))
        if entry["n_response_picks"]:
            prefix = entry["indices"][: entry["n_response_picks"]]
            entry["fresh_bits_n_response_picks"] = round(h_prior - float(entropies(prefix).mean()), 4)
        else:
            entry["fresh_bits_n_response_picks"] = None


def time_full_pass(
    draws: Dict[str, np.ndarray], args: Args, *, dtype: str, threads: int
) -> float:
    """Seconds for one exact full scoring pass over the pool (one pick)."""
    start = time.perf_counter()
    select_n_joint_eig(
        draws,
        1,
        n_scenarios=args.n_scenarios,
        seed=args.selection_seed,
        n_responses=args.n_responses,
        dtype=dtype,
        n_threads=threads,
    )
    return round(time.perf_counter() - start, 1)


def compare(
    label: str,
    draws: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]],
    args: Args,
    threads: int,
    *,
    exact_float64: bool,
    extra_sets: Optional[Dict[str, Tuple[List[int], List[Dict[str, Any]]]]] = None,
) -> Dict[str, Any]:
    """Every search setting on one pool, scored against exact float64 greedy
    (or exact float32 greedy when ``exact_float64`` is off). ``extra_sets``
    are sets chosen elsewhere (``name -> (pool indices, stimuli.json
    entries)``), scored alongside."""
    runs = [
        ("exact_float32", False, "float32", args.selection_seed),
        ("exact_float32_other_scenarios", False, "float32", args.second_seed),
        ("lazy_float32", True, "float32", args.selection_seed),
        ("lazy_float64", True, "float64", args.selection_seed),
    ]
    if exact_float64:
        runs.insert(0, ("exact_float64", False, "float64", args.selection_seed))
    sets: Dict[str, Dict[str, Any]] = {}
    for name, lazy, dtype, seed in runs:
        print(f"[{label}] {name} ...", flush=True)
        sets[name] = run_selection(draws, weights, args, lazy=lazy, dtype=dtype, seed=seed, threads=threads)
        print(f"[{label}] {name}: {sets[name]['seconds']} s", flush=True)
    for name, (indices, stimuli) in (extra_sets or {}).items():
        n_response = [s["joint_eig_bits"] for s in stimuli if s["source"] == "eig"]
        sets[name] = {
            "lazy": None, "dtype": None, "seed": None, "seconds": None,
            "indices": indices,
            "sources": [s["source"] for s in stimuli],
            "n_response_picks": len(n_response),
            "in_sample_bits": n_response[-1] if n_response else None,
        }
    reference = "exact_float64" if exact_float64 else "exact_float32"
    score_sets(draws, weights, sets, reference, args)
    return {"n_pairs": int(next(iter(draws.values())).shape[1]), "reference": reference, "sets": sets}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def markdown(results: Dict[str, Any]) -> str:
    lines = [f"# Lazy vs exact greedy joint-EIG selection: {results['cell']}", ""]
    lines.append(
        f"Experiment {results['experiment']} design, {results['n_models']} models, "
        f"{results['n_responses']} responses per stimulus, {results['n_select']} picks, "
        f"{results['fresh_scenarios']:,d} fresh scenarios for scoring, "
        f"{results['threads']} threads."
    )
    for key in ("subsample", "full_pool"):
        if key not in results:
            continue
        block = results[key]
        lines += [
            "",
            f"## {key.replace('_', ' ')} ({block['n_pairs']:,d} pairs; reference {block['reference']})",
            "",
            "| set | seconds | n-response picks | in-sample bits | fresh bits (all) "
            "| fresh bits (n-response picks) | diff vs ref ± SE | overlap |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, s in block["sets"].items():
            lines.append(
                f"| {name} | {s['seconds']} | {s['n_response_picks']} | {s['in_sample_bits']} "
                f"| {s['fresh_bits']} | {s['fresh_bits_n_response_picks']} "
                f"| {s['fresh_diff_vs_reference']:+.4f} ± {s['fresh_diff_se']:.4f} "
                f"| {s['overlap_with_reference']} |"
            )
    if "timing" in results:
        lines += ["", "## One full scoring pass over the pool (seconds)", "", "| dtype | 1 thread | all threads |", "|---|---|---|"]
        for dtype, row in results["timing"].items():
            lines.append(f"| {dtype} | {row['1']} | {row['all']} |")
    return "\n".join(lines) + "\n"


def main(args: Args) -> None:
    from src.models.pymc_inference import allocated_cpus

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_inputs(args.cell, work_dir, args.experiment)
    from src.subjective_randomness.stimulus_design import enumerate_all_pairs

    pool = enumerate_all_pairs(list(args.lengths), same_length_only=True)
    rows = [eig_mod._raw_row(item) for item in pool]
    draws = design_draws(staged, args.experiment, rows, args.n_samples, work_dir / "draws.npz")
    weights = eig_mod._load_model_weights(
        staged / f"experiment{args.experiment - 1}" / "model_registry.yaml"
    ) or None
    threads = args.threads or allocated_cpus()
    results: Dict[str, Any] = {
        "cell": str(args.cell),
        "experiment": args.experiment,
        "n_models": len(draws),
        "n_responses": args.n_responses,
        "n_select": args.n_select,
        "fresh_scenarios": args.fresh_scenarios,
        "threads": threads,
    }

    def save() -> None:
        (work_dir / "validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (work_dir / "validation.md").write_text(markdown(results), encoding="utf-8")

    subsample = np.sort(
        np.random.default_rng(args.subsample_seed).choice(len(pool), args.subsample, replace=False)
    )
    results["subsample"] = compare(
        "subsample", {n: a[:, subsample] for n, a in draws.items()}, weights, args, threads,
        exact_float64=True,
    )
    save()
    if args.timing:
        results["timing"] = {
            dtype: {
                "1": time_full_pass(draws, args, dtype=dtype, threads=1),
                "all": time_full_pass(draws, args, dtype=dtype, threads=threads),
            }
            for dtype in ("float32", "float64")
        }
        save()
    if args.full_pool:
        stimuli = json.loads(
            (staged / f"experiment{args.experiment}" / "design" / "stimuli.json").read_text()
        )
        results["full_pool"] = compare(
            "full pool", draws, weights, args, threads,
            exact_float64=args.full_pool_float64,
            extra_sets={"sweep_design": (pool_index_of(pool, stimuli), stimuli)},
        )
        save()
    print(markdown(results), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
