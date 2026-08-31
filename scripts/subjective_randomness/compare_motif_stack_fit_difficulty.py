"""Compare NUTS fitting difficulty: motif_stack (Viterbi/max) vs its softmax twin.

Both models are fit to the SAME synthetic dataset — generated from the Viterbi
pure-Python twin at its default parameters — at identical sampler settings, so
any difference in sampler behaviour is attributable to the likelihood geometry.
The Viterbi model's double max (Viterbi over hidden paths, argmax over the four
production methods) puts non-differentiable ridges into the log-density;
``motif_stack_softmax`` replaces both maxes with their log-sum-exp counterparts,
which marginalises those ridges away and should be smoother for NUTS.

For each (model, target_accept) it reports the hardware-independent difficulty
metrics that the models' own source comments already use as the yardstick —
leapfrog steps per iteration and tree depth — plus divergences, effective
samples per 1000 leapfrog steps (an efficiency measure independent of CPU
speed), wall-clock, min ESS, and max R-hat.

Usage (on a Sherlock dev node, NOT the login node):
    eltest_venv3/bin/python \\
        scripts/subjective_randomness/compare_motif_stack_fit_difficulty.py
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tyro
from pyprojroot import here

from src.models.pymc_inference import fit_model
from src.subjective_randomness.model_families import motif_stack as viterbi_twin
from src.subjective_randomness.model_families import motif_stack_softmax as softmax_twin

PYMC_MODEL_DIR = here() / "src" / "subjective_randomness" / "pymc_model_families"

# The two adapters share every parameter name, so ESS/R-hat range over the same
# set for both. p_left is a deterministic and is excluded.
PARAM_NAMES = [
    "delta",
    "alpha",
    "repetition_weight",
    "mirror_share",
    "complement_share",
    "beta",
    "side_bias",
]

MODELS = ["motif_stack", "motif_stack_softmax"]


@dataclass
class Args:
    """Fit both motif_stack adapters to one synthetic dataset and compare NUTS cost."""

    n_participants: int = 15
    """Participants; each answers every stimulus pair once. Rows = participants x pairs."""
    target_accepts: List[float] = field(default_factory=lambda: [0.9, 0.99])
    """Fit both models at each of these target_accept values (0.9 = the Viterbi
    model's declared workaround; 0.99 = the production default it lowered from)."""
    draws: int = 400
    """Posterior draws per chain."""
    tune: int = 800
    """Tuning (warm-up) steps per chain."""
    chains: int = 4
    """Chains. Run sequentially (cores=1) so wall-clock is comparable under a
    1-CPU dev allocation."""
    gen_seed: int = 0
    """Seed for the Bernoulli response draws (the generating params are fixed)."""
    fit_seed: int = 42
    """Seed handed to pm.sample for every fit."""
    out_dir: Path = Path("data/analysis/motif_stack_softmax_fit_difficulty")
    """Directory (repo-relative) for the responses CSV, metrics CSV, and JSON."""


def _length8_stimuli() -> List[Dict[str, str]]:
    """A length-8 same-length pair bank that fires every production method.

    Strided enumeration for coverage, plus hand-picked sequences that trigger
    duplication / mirror / complement, so the four production methods (whose
    combination is exactly what the two models score differently) all matter.
    Fails loudly if any memory flag never fires — an equivalence of geometry
    over stimuli that exercise only the repetition method would prove little.
    """
    words = ["".join("HT"[(i >> bit) & 1] for bit in range(8)) for i in range(256)]
    chosen = words[::11]  # ~24 evenly spaced sequences
    flaggy = [
        "HHTTHHTT",  # duplication: first half == second half
        "HTHTHTHT",  # duplication
        "HTTHHTTH",  # mirror: suffix == reverse(prefix)
        "HHHTTTTT",  # complement-ish material
        "TTHHTTHH",  # duplication
        "HTHHTHHT",
    ]
    seqs = list(dict.fromkeys(chosen + flaggy))
    stimuli = [
        {"sequence_a": a, "sequence_b": seqs[(i + 1) % len(seqs)]}
        for i, a in enumerate(seqs)
    ]
    stimuli = [s for s in stimuli if s["sequence_a"] != s["sequence_b"]]

    fired = {"mirror": 0, "complement": 0, "duplication": 0}
    for stim in stimuli:
        for side in ("sequence_a", "sequence_b"):
            for flag, matched in viterbi_twin._memory_patterns(stim[side]).items():
                fired[flag] += int(matched)
    if not all(fired.values()):
        raise ValueError(f"Stimulus bank does not fire every memory flag: {fired}")
    return stimuli


def _generate_responses_csv(
    stimuli: List[Dict[str, str]], n_participants: int, seed: int, path: Path
) -> int:
    """Sample chose_left ~ Bernoulli(viterbi_twin.predict_left) and write a CSV.

    The Viterbi twin is the generating model on purpose: fitting the softmax
    adapter to it is a mild, conservative misspecification (if anything it should
    make the softmax model look HARDER, not easier), so a softmax win here is not
    an artefact of fitting each model to its own data.
    """
    rng = np.random.default_rng(seed)
    p_left = np.array(
        [viterbi_twin.predict_left(stim) for stim in stimuli], dtype="float64"
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["participant_id", "sequence_a", "sequence_b", "chose_left"]
        )
        writer.writeheader()
        n_rows = 0
        for participant in range(n_participants):
            for stim, p in zip(stimuli, p_left):
                writer.writerow(
                    {
                        "participant_id": f"p{participant}",
                        "sequence_a": stim["sequence_a"],
                        "sequence_b": stim["sequence_b"],
                        "chose_left": int(rng.random() < p),
                    }
                )
                n_rows += 1
    return n_rows


def _difficulty_metrics(idata: Any, wall_clock_s: float) -> Dict[str, float]:
    """NUTS difficulty diagnostics from a fitted idata.

    Leads with the hardware-independent measures (leapfrog steps/iteration, tree
    depth, divergences, ESS per 1000 leapfrog steps); wall-clock is secondary and
    only comparable because the fits run back to back on the same allocation.
    """
    import arviz as az

    stats = idata.sample_stats
    steps = np.asarray(stats["n_steps"].values, dtype=float)
    depth = np.asarray(stats["tree_depth"].values, dtype=float)
    n_div = int(np.asarray(stats["diverging"].values).sum())
    total_steps = float(steps.sum())

    ess = az.ess(idata, var_names=PARAM_NAMES)
    min_ess = min(float(ess[v].min()) for v in ess.data_vars)
    rhat = az.rhat(idata, var_names=PARAM_NAMES)
    max_rhat = max(float(rhat[v].max()) for v in rhat.data_vars)

    return {
        "mean_leapfrog_steps": float(steps.mean()),
        "median_leapfrog_steps": float(np.median(steps)),
        "max_leapfrog_steps": float(steps.max()),
        "mean_tree_depth": float(depth.mean()),
        "max_tree_depth": float(depth.max()),
        "divergences": float(n_div),
        "mean_step_size": float(np.asarray(stats["step_size"].values).mean()),
        "min_ess": min_ess,
        "ess_per_1000_leapfrogs": 1000.0 * min_ess / total_steps,
        "max_rhat": max_rhat,
        "total_leapfrog_steps": total_steps,
        "wall_clock_s": wall_clock_s,
    }


def _fit_and_measure(
    model_name: str, responses_path: Path, target_accept: float, args: Args
) -> Dict[str, Any]:
    start = time.perf_counter()
    fitted = fit_model(
        model_name,
        PYMC_MODEL_DIR,
        responses_path,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=1,  # sequential chains: comparable wall-clock on a 1-CPU dev node
        random_seed=args.fit_seed,
        target_accept=target_accept,  # explicit -> overrides each model's own declaration
    )
    wall = time.perf_counter() - start
    metrics = _difficulty_metrics(fitted.idata, wall)
    return {"model": model_name, "target_accept": target_accept, **metrics}


REPORT_COLUMNS = [
    ("mean_leapfrog_steps", "steps/iter", "{:.1f}"),
    ("mean_tree_depth", "tree depth", "{:.2f}"),
    ("divergences", "diverg.", "{:.0f}"),
    ("ess_per_1000_leapfrogs", "ESS/1k steps", "{:.1f}"),
    ("min_ess", "min ESS", "{:.0f}"),
    ("max_rhat", "max R-hat", "{:.3f}"),
    ("wall_clock_s", "wall (s)", "{:.1f}"),
]


def _print_report(rows: List[Dict[str, Any]]) -> None:
    print("\n=== motif_stack (Viterbi/max) vs motif_stack_softmax: NUTS fit difficulty ===")
    header = f"{'model':<22}{'t_accept':>9}" + "".join(
        f"{label:>14}" for _, label, _ in REPORT_COLUMNS
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['model']:<22}{row['target_accept']:>9.2f}"
        for key, _, fmt in REPORT_COLUMNS:
            line += f"{fmt.format(row[key]):>14}"
        print(line)

    # Headline: the steps/iteration ratio at each target_accept.
    print("\nSpeedup (Viterbi steps/iter ÷ softmax steps/iter), lower softmax = easier:")
    by_key = {(r["model"], r["target_accept"]): r for r in rows}
    for ta in sorted({r["target_accept"] for r in rows}):
        v = by_key.get(("motif_stack", ta))
        s = by_key.get(("motif_stack_softmax", ta))
        if v and s and s["mean_leapfrog_steps"] > 0:
            ratio = v["mean_leapfrog_steps"] / s["mean_leapfrog_steps"]
            print(
                f"  target_accept={ta:.2f}: "
                f"Viterbi {v['mean_leapfrog_steps']:.1f} vs softmax "
                f"{s['mean_leapfrog_steps']:.1f}  ->  {ratio:.2f}x fewer steps"
            )


def main(args: Args) -> None:
    out_dir = here() / args.out_dir if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stimuli = _length8_stimuli()
    responses_path = out_dir / "synthetic_responses.csv"
    n_rows = _generate_responses_csv(
        stimuli, args.n_participants, args.gen_seed, responses_path
    )
    print(
        f"Generated {n_rows} rows ({args.n_participants} participants x "
        f"{len(stimuli)} length-8 pairs) from the Viterbi twin -> {responses_path}"
    )

    rows: List[Dict[str, Any]] = []
    for target_accept in args.target_accepts:
        for model_name in MODELS:
            print(f"\nFitting {model_name} at target_accept={target_accept} ...")
            rows.append(_fit_and_measure(model_name, responses_path, target_accept, args))

    _print_report(rows)

    csv_path = out_dir / "fit_difficulty.csv"
    metric_keys = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_keys)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "fit_difficulty.json").write_text(
        json.dumps(
            {"n_rows": n_rows, "n_pairs": len(stimuli), "settings": vars(args) | {"out_dir": str(args.out_dir)}, "results": rows},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote metrics to {csv_path} and {out_dir / 'fit_difficulty.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
