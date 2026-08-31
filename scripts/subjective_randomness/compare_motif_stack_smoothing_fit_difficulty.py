"""Does gentle smoothing of motif_stack buy a fitting benefit at low prediction drift?

Ties the two axes together: for several smoothing settings it fits the model to
the SAME synthetic data (from the Viterbi twin) at target_accept=0.99 and reports
both (a) the NUTS difficulty (leapfrog steps/iter, ESS per 1000 steps, wall) and
(b) the prediction drift from the hard-max motif_stack (mean |Δp_left| over the
exhaustive pool). The settings span the hard max, gentle temperatures, softening
only the production-method max, and the full marginal.

Tempered adapters at arbitrary temperatures are materialised by rewriting the
two module-level temperature constants of the shipped
``pymc_model_families/motif_stack_tempered.py`` into a scratch models dir, so no
per-setting model file is checked in.

Usage (Sherlock dev node, NOT login):
    eltest_venv3/bin/python \\
        scripts/subjective_randomness/compare_motif_stack_smoothing_fit_difficulty.py
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tyro
from pyprojroot import here

from src.models.pymc_inference import fit_model
from src.subjective_randomness.model_families import motif_stack as viterbi_twin
from src.subjective_randomness.model_families import motif_stack_tempered as tempered_twin

REPO_MODEL_DIR = here() / "src" / "subjective_randomness" / "pymc_model_families"
PARAM_NAMES = [
    "delta", "alpha", "repetition_weight", "mirror_share",
    "complement_share", "beta", "side_bias",
]

# (label, path_temperature, method_temperature). None/None = use an existing
# checked-in adapter verbatim (the two endpoints).
CONFIGS: List[Tuple[str, Any, Any]] = [
    ("max (Viterbi)", None, None),          # motif_stack
    ("both tau=0.25", 0.25, 0.25),
    ("both tau=0.35", 0.35, 0.35),
    ("method-only", 0.05, 1.0),             # ~Viterbi paths, full method mixture
    ("full softmax", None, None),           # motif_stack_softmax
]
ENDPOINT_MODEL = {"max (Viterbi)": "motif_stack", "full softmax": "motif_stack_softmax"}


@dataclass
class Args:
    n_participants: int = 15
    target_accept: float = 0.99
    draws: int = 400
    tune: int = 800
    chains: int = 4
    gen_seed: int = 0
    fit_seed: int = 42
    lengths: List[int] = field(default_factory=lambda: [4, 5, 6, 7, 8])
    out_dir: Path = Path("data/analysis/motif_stack_smoothing_fit_difficulty")


def _length8_stimuli() -> List[Dict[str, str]]:
    words = ["".join("HT"[(i >> bit) & 1] for bit in range(8)) for i in range(256)]
    chosen = words[::11]
    flaggy = ["HHTTHHTT", "HTHTHTHT", "HTTHHTTH", "HHHTTTTT", "TTHHTTHH", "HTHHTHHT"]
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


def _generate_responses_csv(stimuli, n_participants, seed, path) -> int:
    rng = np.random.default_rng(seed)
    p_left = np.array([viterbi_twin.predict_left(s) for s in stimuli], dtype="float64")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["participant_id", "sequence_a", "sequence_b", "chose_left"]
        )
        writer.writeheader()
        n = 0
        for participant in range(n_participants):
            for stim, p in zip(stimuli, p_left):
                writer.writerow({
                    "participant_id": f"p{participant}",
                    "sequence_a": stim["sequence_a"],
                    "sequence_b": stim["sequence_b"],
                    "chose_left": int(rng.random() < p),
                })
                n += 1
    return n


def _materialise_tempered_adapter(path_t: float, method_t: float, dest_dir: Path, name: str) -> None:
    """Write a tempered adapter with the two temperature constants substituted."""
    source = (REPO_MODEL_DIR / "motif_stack_tempered.py").read_text(encoding="utf-8")
    source, n1 = re.subn(r"(?m)^PATH_TEMPERATURE = .*$", f"PATH_TEMPERATURE = {path_t}", source)
    source, n2 = re.subn(r"(?m)^METHOD_TEMPERATURE = .*$", f"METHOD_TEMPERATURE = {method_t}", source)
    if n1 != 1 or n2 != 1:
        raise ValueError(f"Expected exactly one temperature line each; got {n1}, {n2}.")
    (dest_dir / f"{name}.py").write_text(source, encoding="utf-8")


def _prediction_drift(path_t: float, method_t: float, lengths: List[int]) -> float:
    """mean |Δp_left| vs Viterbi over the exhaustive same-length pool, default params."""
    params = dict(viterbi_twin.DEFAULT_PARAMS)
    beta, sb = params["beta"], params["side_bias"]
    parts = []
    for length in lengths:
        seqs = ["".join("HT"[(i >> b) & 1] for b in range(length)) for i in range(2**length)]
        pairs = np.array(list(combinations(range(len(seqs)), 2)), dtype=np.int64)
        ia, ib = pairs[:, 0], pairs[:, 1]
        vs = np.array([viterbi_twin.score_sequence(s, params) for s in seqs])
        ts = np.array([
            tempered_twin.score_sequence(s, params, path_temperature=path_t, method_temperature=method_t)
            for s in seqs
        ])
        vp = 1 / (1 + np.exp(-(beta * (vs[ia] - vs[ib]) + sb)))
        tp = 1 / (1 + np.exp(-(beta * (ts[ia] - ts[ib]) + sb)))
        parts.append(np.abs(tp - vp))
    return float(np.mean(np.concatenate(parts)))


def _difficulty(idata, wall_s: float) -> Dict[str, float]:
    import arviz as az

    stats = idata.sample_stats
    steps = np.asarray(stats["n_steps"].values, dtype=float)
    total = float(steps.sum())
    ess = az.ess(idata, var_names=PARAM_NAMES)
    min_ess = min(float(ess[v].min()) for v in ess.data_vars)
    rhat = az.rhat(idata, var_names=PARAM_NAMES)
    max_rhat = max(float(rhat[v].max()) for v in rhat.data_vars)
    return {
        "mean_leapfrog_steps": float(steps.mean()),
        "divergences": float(int(np.asarray(stats["diverging"].values).sum())),
        "min_ess": min_ess,
        "ess_per_1000_leapfrogs": 1000.0 * min_ess / total,
        "max_rhat": max_rhat,
        "wall_clock_s": wall_s,
    }


def main(args: Args) -> None:
    out_dir = here() / args.out_dir if not args.out_dir.is_absolute() else args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_models = Path(os.environ.get("CLAUDE_JOB_DIR", str(out_dir))) / "tempered_models"
    if scratch_models.exists():
        shutil.rmtree(scratch_models)
    scratch_models.mkdir(parents=True, exist_ok=True)

    stimuli = _length8_stimuli()
    responses = out_dir / "synthetic_responses.csv"
    n_rows = _generate_responses_csv(stimuli, args.n_participants, args.gen_seed, responses)
    print(f"Generated {n_rows} rows from the Viterbi twin -> {responses}\n")

    rows: List[Dict[str, Any]] = []
    for label, path_t, method_t in CONFIGS:
        if path_t is None:
            model_name, models_dir = ENDPOINT_MODEL[label], REPO_MODEL_DIR
            drift = 0.0 if model_name == "motif_stack" else _prediction_drift(1.0, 1.0, args.lengths)
        else:
            model_name = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            _materialise_tempered_adapter(path_t, method_t, scratch_models, model_name)
            models_dir = scratch_models
            drift = _prediction_drift(path_t, method_t, args.lengths)

        print(f"Fitting '{label}' ({model_name}) at target_accept={args.target_accept} ...")
        start = time.perf_counter()
        fitted = fit_model(
            model_name, models_dir, responses,
            draws=args.draws, tune=args.tune, chains=args.chains, cores=1,
            random_seed=args.fit_seed, target_accept=args.target_accept,
        )
        wall = time.perf_counter() - start
        rows.append({"config": label, "mean_abs_dp_vs_max": drift, **_difficulty(fitted.idata, wall)})

    print("\n=== Smoothing: prediction drift vs fitting difficulty (target_accept="
          f"{args.target_accept}) ===")
    header = (f"{'config':<16}{'mean|Δp| vs max':>16}{'steps/iter':>12}{'ESS/1k steps':>14}"
              f"{'min ESS':>9}{'diverg':>8}{'R-hat':>8}{'wall(s)':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['config']:<16}{r['mean_abs_dp_vs_max']:>16.4f}{r['mean_leapfrog_steps']:>12.1f}"
              f"{r['ess_per_1000_leapfrogs']:>14.1f}{r['min_ess']:>9.0f}{r['divergences']:>8.0f}"
              f"{r['max_rhat']:>8.3f}{r['wall_clock_s']:>9.1f}")

    csv_path = out_dir / "smoothing_fit_difficulty.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
