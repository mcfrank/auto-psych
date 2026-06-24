"""Mega-analytic model comparison: fit one population-level PyMC model per
theory on the human data and score each on ELPD-LOO, RMSE/Brier, and R^2.

This is the compute half of notebook 02 ("Best-fitting theories"). It reuses the
project's inference harness verbatim:
  - src.subjective_randomness.features.featurize_responses_csv  (raw -> features)
  - src.models.pymc_inference.fit_model / FittedModel           (MCMC + ELPD + p_left)

Models (mega_models.json): the 4 canonical seed theories + the 4 overall run
winners (run 2 tied -> two models). Each .py is copied verbatim into a working
models_dir (preserving any module-level `compute_features`).

Two schemes (run both by default):

  mega     One population-level fit per model on ALL pooled data (11,520 trials).
           Scored in-sample with ELPD-LOO + RMSE/R^2. This is the headline
           "collective fit" comparison.

  heldout  Removes selection optimism: each winner was DISCOVERED on one run, so
           we refit it (and the seeds) only on the OTHER two runs. For each
           held-out run r we report (a) the in-sample fit on the two training
           runs (ELPD-LOO + RMSE/R^2), and (b) genuine out-of-sample prediction
           of the held-out run r (RMSE/R^2/log-density). Seeds, tuned on no run,
           are refit per training pair so every comparison is on identical data.

Outputs (to --out-dir, default analysis/behavioral/data):
  mega_model_metrics.csv          one row per (scheme, held_out_run, eval, model)
  mega_model_predictions.csv      tidy long per-trial posterior-mean p_left + choice
  mega_model_posterior_means.csv  tidy long posterior mean+sd of each fitted
                                  parameter, one row per (scheme, held_out_run,
                                  model, parameter); vector params unpacked into
                                  name[0], name[1], ...

Usage:
    python analysis/behavioral/fit_mega_models.py            # both schemes
    python analysis/behavioral/fit_mega_models.py --scheme mega
    python analysis/behavioral/fit_mega_models.py --quick    # tiny MCMC smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

HUMAN_ROOT = REPO_ROOT / "data" / "results" / "human_experiment"
MANIFEST = Path(__file__).resolve().parent / "mega_models.json"


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------
def pooled_rows(runs: Optional[set] = None) -> List[Dict[str, Any]]:
    """Every trial across runs × rounds, in a fixed order. Optionally restrict
    to `runs` (a set of run integers). `participant_uid` is globally unique."""
    files = sorted(
        HUMAN_ROOT.glob("run*/subjective_randomness/experiment*/data/responses.csv")
    )
    if not files:
        raise FileNotFoundError(f"no responses.csv under {HUMAN_ROOT}")

    out: List[Dict[str, Any]] = []
    for path in files:
        run = int(re.search(r"run(\d+)", str(path)).group(1))
        if runs is not None and run not in runs:
            continue
        rnd = int(re.search(r"experiment(\d+)", str(path)).group(1))
        with path.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                out.append({
                    "run": run, "round": rnd,
                    "participant_uid": f"r{run}_e{rnd}_p{r['participant_id']}",
                    "trial_index": int(r["trial_index"]),
                    "sequence_a": r["sequence_a"], "sequence_b": r["sequence_b"],
                    "chose_left": int(r["chose_left"]),
                })
    return out


def write_featurized(rows: List[Dict[str, Any]], dest: Path) -> List[Dict[str, str]]:
    """Write a pooled raw responses.csv, featurize it, and return the featurized
    rows (same order as `rows`, ready to bind as stim data)."""
    from src.subjective_randomness.features import featurize_responses_csv

    raw = dest.with_suffix(".raw.csv")
    cols = ["participant_id", "trial_index", "sequence_a", "sequence_b",
            "chose_left", "chose_right", "model"]
    with raw.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({
                "participant_id": r["participant_uid"], "trial_index": r["trial_index"],
                "sequence_a": r["sequence_a"], "sequence_b": r["sequence_b"],
                "chose_left": r["chose_left"], "chose_right": 1 - r["chose_left"],
                "model": "",
            })
    featurize_responses_csv(raw, dest)
    with dest.open(encoding="utf-8", newline="") as f:
        feat = list(csv.DictReader(f))
    assert len(feat) == len(rows), "row alignment broke during featurization"
    return feat


def assemble_models_dir(manifest: dict, work: Path) -> List[dict]:
    """Copy each manifest model .py into `work` under its canonical name."""
    work.mkdir(parents=True, exist_ok=True)
    for m in manifest["models"]:
        src = REPO_ROOT / m["source"]
        if not src.exists():
            raise FileNotFoundError(f"model source missing: {src}")
        shutil.copyfile(src, work / f"{m['name']}.py")
    return manifest["models"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _r2(y, yhat) -> float:
    import numpy as np

    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def point_metrics(p_left, observed) -> Dict[str, float]:
    import numpy as np

    p = np.clip(np.asarray(p_left, float), 1e-6, 1 - 1e-6)
    y = np.asarray(observed, float)
    brier = float(np.mean((p - y) ** 2))
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    pbar = float(y.mean())
    ll_null = float(np.sum(y * math.log(pbar) + (1 - y) * math.log(1 - pbar))) if 0 < pbar < 1 else float("nan")
    return {
        "n_eval": int(len(y)),
        "rmse": math.sqrt(brier), "brier": brier,
        "accuracy": float(np.mean((p >= 0.5) == (y >= 0.5))),
        "r2_trial": _r2(y, p),
        "logpd": ll,
        "pseudo_r2_deviance": (1.0 - ll / ll_null) if ll_null not in (0, float("nan")) and not math.isnan(ll_null) else float("nan"),
    }


def stimulus_r2(meta, p_left, observed) -> float:
    """R^2 of predicted vs empirical choice proportion per distinct ORDERED
    stimulus (sequence_a, sequence_b), one unweighted point per stimulus."""
    agg: Dict[tuple, List[List[float]]] = {}
    for m, p, y in zip(meta, p_left, observed):
        key = (m["sequence_a"], m["sequence_b"])
        agg.setdefault(key, [[], []])
        agg[key][0].append(p)
        agg[key][1].append(y)
    obs = [sum(ys) / len(ys) for _, ys in agg.values()]
    pred = [sum(ps) / len(ps) for ps, _ in agg.values()]
    return _r2(obs, pred)


def count_free_params(name: str, models_dir: Path) -> int:
    from src.models.pymc_inference import load_pymc_model

    return len(load_pymc_model(name, models_dir).free_RVs)


def posterior_param_means(idata, param_names) -> List[Dict[str, Any]]:
    """One ``{param, posterior_mean, posterior_sd}`` record per scalar parameter.

    ``param_names`` must be the model's fitted (free) parameters — pass
    ``[rv.name for rv in fitted.model.free_RVs]`` so the per-trial Deterministic
    ``p_left`` (an 11,520-long vector, also stored in the posterior) is excluded.
    Vector-valued parameters (e.g. a length-3 Dirichlet ``weights``) are unpacked
    by ArviZ into ``weights[0]``, ``weights[1]``, ... so every record is one
    scalar quantity. ``kind="stats"`` keeps this to the posterior mean/sd without
    recomputing R-hat/ESS.
    """
    import arviz as az

    summary = az.summary(idata, var_names=list(param_names), kind="stats")
    return [
        {
            "param": str(param),
            "posterior_mean": float(row["mean"]),
            "posterior_sd": float(row["sd"]),
        }
        for param, row in summary.iterrows()
    ]


# ---------------------------------------------------------------------------
# Fit a set of models on one training set; score on one or more eval sets.
# ---------------------------------------------------------------------------
def fit_group(models, train_meta, work, featurized_dir, fit_kwargs, cache_dir,
              evals, scheme, held_out_run):
    """Fit each model in `models` on `train_meta`; score on each eval set.

    `evals` is a list of (eval_tag, eval_meta, is_train) tuples. ELPD-LOO is
    attached only to the eval whose data IS the training set (is_train=True),
    since PSIS-LOO cross-validates the fitted points. Returns
    (metric_rows, pred_rows, param_rows, idatas) where idatas maps model name ->
    InferenceData (for the in-train az.compare done by the caller) and param_rows
    holds one posterior-mean record per fitted parameter (per fit, not per eval).
    """
    from src.models.pymc_inference import fit_model, make_stim_data

    train_feat = write_featurized(train_meta, featurized_dir / "train.csv")
    train_csv = featurized_dir / "train.csv"

    eval_feat = {}
    for tag, em, is_train in evals:
        eval_feat[tag] = train_feat if is_train else write_featurized(em, featurized_dir / f"eval_{tag}.csv")

    metric_rows, pred_rows, param_rows, idatas = [], [], [], {}
    for m in models:
        name = m["name"]
        print(f"[mega] {scheme} ho={held_out_run} fit {name} "
              f"(train n={len(train_meta)}) ...", flush=True)
        fitted = fit_model(name, work, train_csv, cache_dir=cache_dir, **fit_kwargs)
        idatas[name] = fitted.idata
        elpd_loo = fitted.elpd_loo()

        # Posterior mean (+ sd) of each fitted parameter. Tied to the fit (the
        # training set), so recorded once here rather than per eval set.
        free_param_names = [rv.name for rv in fitted.model.free_RVs]
        for pm_rec in posterior_param_means(fitted.idata, free_param_names):
            param_rows.append({
                "scheme": scheme, "held_out_run": held_out_run,
                "name": name, "kind": m["kind"], "source_run": m.get("source_run"),
                "label": m["label"], **pm_rec,
            })

        for tag, em, is_train in evals:
            feat = eval_feat[tag]
            stim = make_stim_data(fitted.model, feat)
            p_left = fitted.predict_p_left(stim, seed=fit_kwargs["random_seed"], max_draws=1000)
            obs = [r["chose_left"] for r in em]
            pm = point_metrics(p_left, obs)
            rec = {
                "scheme": scheme, "held_out_run": held_out_run, "eval": tag,
                "name": name, "kind": m["kind"], "source_run": m.get("source_run"),
                "label": m["label"], "n_params": count_free_params(name, work),
                "elpd_loo": elpd_loo if is_train else None,
                "r2_stimulus": stimulus_r2(em, p_left, obs),
                **pm,
            }
            metric_rows.append(rec)
            for mm, p in zip(em, p_left):
                pred_rows.append({
                    "scheme": scheme, "held_out_run": held_out_run, "eval": tag,
                    "run": mm["run"], "round": mm["round"],
                    "participant_uid": mm["participant_uid"], "trial_index": mm["trial_index"],
                    "sequence_a": mm["sequence_a"], "sequence_b": mm["sequence_b"],
                    "chose_left": mm["chose_left"], "model": name, "p_left": float(p),
                })
    return metric_rows, pred_rows, param_rows, idatas


def attach_compare(metric_rows, idatas, scheme, held_out_run):
    """Attach az.compare columns (rank, se, elpd_diff, dse, weight) to the
    in-train eval rows for one fit group (models sharing a training set)."""
    import arviz as az

    cmp = az.compare(idatas, ic="loo")
    for rec in metric_rows:
        if rec["scheme"] != scheme or rec["held_out_run"] != held_out_run:
            continue
        if rec["elpd_loo"] is None:  # out-of-sample eval: LOO not applicable
            continue
        c = cmp.loc[rec["name"]]
        rec.update({
            "elpd_se": float(c.get("se", float("nan"))),
            "elpd_rank": int(c["rank"]),
            "elpd_diff": float(c["elpd_diff"]),
            "elpd_dse": float(c.get("dse", float("nan"))),
            "stacking_weight": float(c["weight"]),
            "loo_unreliable": bool(c.get("warning", False)),
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "analysis" / "behavioral" / "data")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--scheme", choices=["mega", "heldout", "both"], default="both")
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="draws=200 tune=200 chains=2")
    args = ap.parse_args()

    if args.quick:
        args.draws, args.tune, args.chains, args.cores = 200, 200, 2, 2

    import arviz  # noqa: F401  (fail early if env is broken)

    manifest = json.loads(MANIFEST.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="mega_models_"))
    models = assemble_models_dir(manifest, work)
    all_runs = sorted({int(re.search(r"run(\d+)", str(p)).group(1))
                       for p in HUMAN_ROOT.glob("run*/subjective_randomness")})

    fit_kwargs = dict(draws=args.draws, tune=args.tune, chains=args.chains,
                      cores=args.cores, random_seed=args.seed)

    all_metrics, all_preds, all_params = [], [], []

    if args.scheme in ("mega", "both"):
        meta = pooled_rows()
        fdir = work / "feat_mega"; fdir.mkdir()
        mrows, prows, parrows, idatas = fit_group(
            models, meta, work, fdir, fit_kwargs, args.cache_dir,
            evals=[("in_sample", meta, True)], scheme="mega", held_out_run=None)
        attach_compare(mrows, idatas, "mega", None)
        all_metrics += mrows; all_preds += prows; all_params += parrows

    if args.scheme in ("heldout", "both"):
        for r in all_runs:
            grp = [m for m in models if m["kind"] == "seed" or m.get("source_run") == r]
            if not any(m.get("source_run") == r for m in grp):
                continue  # no winner came from this run -> nothing to de-bias
            train_meta = pooled_rows(runs=set(all_runs) - {r})
            oos_meta = pooled_rows(runs={r})
            fdir = work / f"feat_ho{r}"; fdir.mkdir()
            mrows, prows, parrows, idatas = fit_group(
                grp, train_meta, work, fdir, fit_kwargs, args.cache_dir,
                evals=[("heldout_train", train_meta, True),
                       ("heldout_oos", oos_meta, False)],
                scheme="heldout", held_out_run=r)
            attach_compare(mrows, idatas, "heldout", r)
            all_metrics += mrows; all_preds += prows; all_params += parrows

    # --- write outputs ------------------------------------------------------
    metric_cols = ["scheme", "held_out_run", "eval", "name", "kind", "source_run",
                   "label", "n_params", "elpd_loo", "elpd_se", "elpd_rank",
                   "elpd_diff", "elpd_dse", "stacking_weight", "loo_unreliable",
                   "n_eval", "rmse", "brier", "accuracy", "r2_trial", "r2_stimulus",
                   "logpd", "pseudo_r2_deviance"]
    metrics_path = args.out_dir / "mega_model_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metric_cols, lineterminator="\n")
        w.writeheader()
        for rec in all_metrics:
            w.writerow({k: rec.get(k) for k in metric_cols})

    preds_path = args.out_dir / "mega_model_predictions.csv"
    with preds_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_preds[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(all_preds)

    param_cols = ["scheme", "held_out_run", "name", "kind", "source_run", "label",
                  "param", "posterior_mean", "posterior_sd"]
    params_path = args.out_dir / "mega_model_posterior_means.csv"
    with params_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=param_cols, lineterminator="\n")
        w.writeheader()
        for rec in all_params:
            w.writerow({k: rec.get(k) for k in param_cols})

    shutil.rmtree(work, ignore_errors=True)
    print(f"[mega] wrote {metrics_path} ({len(all_metrics)} rows)")
    print(f"[mega] wrote {preds_path} ({len(all_preds)} rows)")
    print(f"[mega] wrote {params_path} ({len(all_params)} rows)")


if __name__ == "__main__":
    main()
