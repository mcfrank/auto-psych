"""Discriminating-stimulus probe for a finished holdout-recovery run.

The holdout-recovery harness reports how well the recovered model predicts the
held-out ground truth on a *random* pool of stimuli. A high correlation there
can still hide a structurally different model that merely *mimics* the ground
truth over the tested regime. This probe stresses exactly that: for each
held-out ground truth it mines a large in-distribution candidate pool for the
stimuli where the recovered winning model and the ground truth *disagree most*,
then re-scores agreement on that adversarial set against a matched random
control drawn from the same pool.

How to read the result:

* If even the worst-case disagreement stays small and the adversarial Pearson r
  stays high, the recovered model is a genuine functional equivalent of the
  ground truth over this stimulus space — strong, structure-independent
  recovery.
* If the adversarial set exposes large ``|Δp_left|`` and the correlation there
  collapses, the random-pool recovery was regime-specific mimicry: the two
  models diverge somewhere the random pool never happened to probe.

Only the single best (argmax-posterior) model is probed — it is the model the
loop would actually report — so the cost is one MCMC fit per ground truth. All
scoring after that is pure array indexing on the pool predictions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.models.pymc_inference import fit_model
from src.subjective_randomness.holdout_recovery import (
    _eval_prediction,
    _participant_ids_in,
    _unordered_pair,
    collect_trained_pairs,
)
from src.subjective_randomness.model_recovery import feature_rows, p_left_fixed_params
from src.subjective_randomness.recover import pearson_r
from src.subjective_randomness.stimulus_design import generate_candidate_pool


def select_by_disagreement(
    gt_p: Sequence[float], winner_p: Sequence[float], k: int
) -> List[int]:
    """Indices of the ``k`` stimuli where ``|gt_p - winner_p|`` is largest.

    Pure and deterministic: results are sorted by disagreement descending, and
    ties break by original index (stable sort). Raises if the vectors differ in
    length or ``k`` is outside ``[1, n]``.
    """
    gt = np.asarray(gt_p, dtype="float64")
    win = np.asarray(winner_p, dtype="float64")
    if gt.shape != win.shape:
        raise ValueError(
            f"Prediction vectors differ in length: {gt.shape} vs {win.shape}"
        )
    if not 1 <= k <= gt.size:
        raise ValueError(f"k must be in [1, {gt.size}], got {k}")
    disagreement = np.abs(gt - win)
    order = np.argsort(-disagreement, kind="stable")
    return order[:k].tolist()


def _agreement_scores(
    gt_p: np.ndarray, winner_p: np.ndarray, idx: np.ndarray
) -> Dict[str, Optional[float]]:
    """Pearson r, RMSE, and disagreement magnitude on the selected stimuli."""
    g = gt_p[idx]
    w = winner_p[idx]
    delta = np.abs(g - w)
    return {
        "n": int(idx.size),
        "pearson_r": pearson_r(g.tolist(), w.tolist()),
        "rmse": float(np.sqrt(np.mean((g - w) ** 2))),
        "mean_abs_disagreement": float(np.mean(delta)),
        "max_abs_disagreement": float(np.max(delta)),
    }


def probe_gt_run(
    gt_run: Mapping[str, Any],
    *,
    seed_models_dir: Path,
    n_experiments: int,
    cache_dir: Optional[Path],
    fit_kwargs: Mapping[str, Any],
    pool_size: int,
    k: int,
    lengths: Sequence[int],
    pool_seed: int,
    control_seed: int,
) -> Dict[str, Any]:
    """Run the discriminating-stimulus probe for one held-out ground truth.

    Fits the run's final best model on its pooled responses, predicts ``p_left``
    for both that model and the fixed-param ground truth across a large
    holdout-clean pool, then contrasts agreement on the top-``k`` most
    disagreeing stimuli with a matched random control and the whole pool.
    """
    import sys
    import time

    def _log(msg: str) -> None:
        print(f"  [probe:{gt_run['gt_model']}] {msg}", file=sys.stderr, flush=True)

    gt_model = gt_run["gt_model"]
    gt_params = gt_run["params"]
    run_root = Path(gt_run["run_root"])

    # 1. A large in-distribution pool, excluding every pair used in training so
    #    the probe stays as held-out as the original eval pool.
    t0 = time.time()
    pool = generate_candidate_pool(pool_size, lengths=tuple(lengths), seed=pool_seed)
    trained = collect_trained_pairs(run_root, n_experiments)
    pool = [
        stim
        for stim in pool
        if _unordered_pair(stim["sequence_a"], stim["sequence_b"]) not in trained
    ]
    if len(pool) < 2 * k:
        raise ValueError(
            f"{gt_model}: only {len(pool)} pool stimuli remain after holdout, "
            f"need at least 2*k={2 * k}; enlarge pool_size or lower k."
        )
    _log(f"pool ready: {len(pool)} stimuli ({time.time() - t0:.1f}s)")

    # 2. Ground-truth and recovered-winner p_left across the whole pool.
    t1 = time.time()
    gt_p = p_left_fixed_params(gt_model, seed_models_dir, pool, gt_params)
    _log(f"ground-truth forward pass done ({time.time() - t1:.1f}s)")
    loop_dir = run_root / f"experiment{n_experiments}" / "model_loop"
    history = json.loads((loop_dir / "history.json").read_text(encoding="utf-8"))
    if not history:
        raise ValueError(f"Empty inner-loop history at {loop_dir / 'history.json'}")
    best_model = history[-1]["best_model"]
    t2 = time.time()
    fitted = fit_model(
        best_model,
        loop_dir / "models",
        loop_dir / "responses.csv",
        cache_dir=cache_dir,
        **dict(fit_kwargs),
    )
    _log(f"refit winner {best_model!r} ({time.time() - t2:.1f}s)")
    participant_ids = _participant_ids_in(loop_dir / "responses.csv")
    winner_p = np.asarray(
        _eval_prediction(fitted, feature_rows(pool), participant_ids=participant_ids),
        dtype="float64",
    )
    _log("winner predictions done")

    # 3. Adversarial (max-disagreement) vs. matched random control.
    adv_idx = select_by_disagreement(gt_p, winner_p, k)
    rng = np.random.default_rng(control_seed)
    # Draw the matched control from the NON-adversarial stimuli so the random
    # control cannot overlap the max-disagreement set it is meant to contrast
    # (len(pool) >= 2*k is enforced upstream, so there are always >= k eligible).
    adv_set = {int(i) for i in adv_idx}
    eligible = np.array([i for i in range(len(pool)) if i not in adv_set])
    ctrl_idx = sorted(int(i) for i in rng.choice(eligible, size=k, replace=False))

    return {
        "gt_model": gt_model,
        "final_best_model": best_model,
        "n_pool": len(pool),
        "k": k,
        "pool": _agreement_scores(gt_p, winner_p, np.arange(len(pool))),
        "control": _agreement_scores(gt_p, winner_p, np.array(ctrl_idx)),
        "adversarial": _agreement_scores(gt_p, winner_p, np.array(adv_idx)),
        # Full pool predictions kept for the scatter plot; the CLI strips these
        # underscore keys before writing the summary JSON.
        "_plot": {
            "gt_p": gt_p.tolist(),
            "winner_p": winner_p.tolist(),
            "adv_idx": adv_idx,
        },
    }


def run_probe(
    result: Mapping[str, Any],
    *,
    cache_dir: Optional[Path],
    seed_models_dir: Optional[Path] = None,
    pool_size: int = 8000,
    k: int = 300,
    pool_seed: int = 101,
    control_seed: int = 202,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Probe every held-out ground truth in a holdout-recovery result dict.

    ``fit_kwargs`` overrides the MCMC settings used to refit each winner. It
    defaults to the run's own settings, but the probe only needs a stable
    posterior-mean ``p_left``, so a lighter setting (fewer chains/draws) gives
    the same answer far faster.
    """
    seed_models_dir = Path(seed_models_dir or result["seed_models_dir"])
    n_experiments = result["n_experiments"]
    if fit_kwargs is None:
        fit_kwargs = result["fit"] if "fit" in result else result["fit_kwargs"]
    lengths = result["eval_pool"]["lengths"]

    gt_runs = [
        probe_gt_run(
            gt_run,
            seed_models_dir=seed_models_dir,
            n_experiments=n_experiments,
            cache_dir=cache_dir,
            fit_kwargs=fit_kwargs,
            pool_size=pool_size,
            k=k,
            lengths=lengths,
            pool_seed=pool_seed,
            control_seed=control_seed,
        )
        for gt_run in result["gt_runs"]
    ]
    return {"pool_size": pool_size, "k": k, "gt_runs": gt_runs}


def _fmt(value: Optional[float]) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def probe_summary_text(probe: Mapping[str, Any]) -> str:
    """One aligned block per ground truth contrasting control vs. adversarial."""
    lines = [
        f"Discriminating-stimulus probe — pool {probe['pool_size']}, "
        f"top-k {probe['k']} by |Δp_left|",
        "(control = matched random subset; adversarial = max-disagreement subset)",
    ]
    for run in probe["gt_runs"]:
        lines.append("")
        lines.append(
            f"{run['gt_model']}  (recovered winner: {run['final_best_model']}, "
            f"pool n={run['n_pool']})"
        )
        header = ["set", "pearson_r", "rmse", "mean|Δp|", "max|Δp|"]
        lines.append("  " + "  ".join(f"{h:>11}" for h in header))
        for label in ("pool", "control", "adversarial"):
            s = run[label]
            row = [
                label,
                _fmt(s["pearson_r"]),
                _fmt(s["rmse"]),
                _fmt(s["mean_abs_disagreement"]),
                _fmt(s["max_abs_disagreement"]),
            ]
            lines.append("  " + "  ".join(f"{v:>11}" for v in row))
    return "\n".join(lines)


def plot_probe(probe: Mapping[str, Any], out_path: Path) -> None:
    """Scatter recovered-winner vs. ground-truth ``p_left``, one panel per GT.

    Faint points are the whole pool; highlighted points are the top-``k``
    max-disagreement stimuli. Tight clustering on the identity line everywhere
    means functional equivalence; a highlighted cloud peeling away from the
    diagonal is the regime where the structurally different winner and the
    ground truth actually diverge.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = probe["gt_runs"]
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.6), squeeze=False)
    for ax, run in zip(axes[0], runs):
        gt_p = np.asarray(run["_plot"]["gt_p"])
        winner_p = np.asarray(run["_plot"]["winner_p"])
        adv = np.asarray(run["_plot"]["adv_idx"])
        ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
        ax.scatter(gt_p, winner_p, s=6, alpha=0.2, color="#4878CF", label="pool")
        ax.scatter(
            gt_p[adv],
            winner_p[adv],
            s=12,
            alpha=0.7,
            color="#D65F5F",
            label=f"top-{run['k']} disagreement",
        )
        adv_s = run["adversarial"]
        ax.set_title(
            f"{run['gt_model']}\n"
            f"winner={run['final_best_model']}\n"
            f"adv r={_fmt(adv_s['pearson_r'])}, max|Δp|={adv_s['max_abs_disagreement']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("ground-truth p_left")
        ax.set_ylabel("recovered-winner p_left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Discriminating-stimulus probe — recovered winner vs. held-out ground truth"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
