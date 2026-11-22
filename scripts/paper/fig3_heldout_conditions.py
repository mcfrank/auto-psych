"""Figure 3's rows (a) and (b) on Figure 2's split: per condition, people's data and the best discovered model's
and best seed's held-out predictions, as the figure data's ``conditions``, plus each model's summed held-out log
predictive density over the condition's trials (``<role>_lpd``), whose total is Figure 2's score.

- Subjective randomness (``subjective-randomness``): leave-one-out. Each trial's PSIS-LOO predictive probability
  of choosing the left sequence, from the pooled fits Figure 2 scored (all runs' final data, 23,104 trials; read
  from the figure data's fit cache, never sampled). For a binary choice it is the trial's LOO density of the
  observed choice, exp(``loo_i``), or one minus it if the right sequence was chosen, so the conditions'
  ``<role>_lpd`` sum to Figure 2's ELPD-LOO. Before writing, the trial alignment is checked: the in-sample
  P(left) from the same fits' log-likelihood, per condition, must match the figure data's in-sample conditions.
- Number game (``number-game``): the Bigelow & Piantadosi responses whose (set, target) item no run drew
  (Figure 2's test set, ``number_game_unseen_items.py``), each predicted by the pooled full-data fit (read from
  the figure data's fit cache, never sampled) with that script's draws and seed; the items' ``<role>_lpd`` must
  sum to its record's (``data/paper/number_game_unseen_items.json``). The discovered model is
  ``--model`` (Figure 3's ``COLUMN_MODEL``, run 3's winner), the seed Figure 2's best seed on that test.

Run on Sherlock, where the fits are (``--work-dir``: the figure data's work directory):

    uv run python -m scripts.paper.fig3_heldout_conditions subjective-randomness --work-dir <sr work dir>
    uv run python -m scripts.paper.fig3_heldout_conditions number-game --work-dir <number-game work dir>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, List, Mapping, Sequence, Union

import numpy as np
import tyro
from pyprojroot import here

FIGURE_DATA = here() / "data" / "paper" / "figure_data"
ROLES = ("best_model", "best_seed")


def loo_p_left(loo_i: np.ndarray, chose_left: np.ndarray) -> np.ndarray:
    """Each trial's leave-one-out P(left) from its LOO log density of the observed (binary) choice."""
    chose_left = np.asarray(chose_left)
    if not np.isin(chose_left, (0, 1)).all():
        raise ValueError("chose_left must be 0 or 1")
    p = np.exp(np.asarray(loo_i, dtype="float64"))
    return np.where(chose_left == 1, p, 1.0 - p)


def in_sample_p_left(log_lik: np.ndarray, chose_left: np.ndarray) -> np.ndarray:
    """Each trial's posterior-mean P(left) from its log-likelihood draws ``(draws, trials)``."""
    p = np.exp(np.asarray(log_lik, dtype="float64"))
    return np.where(np.asarray(chose_left) == 1, p, 1.0 - p).mean(axis=0)


def cell_sums(keys: Sequence[Hashable], values: np.ndarray) -> Dict[Hashable, float]:
    """``values`` (one per trial) summed per condition key."""
    out: Dict[Hashable, float] = {}
    for k, v in zip(keys, values):
        out[k] = out.get(k, 0.0) + float(v)
    return out


def sr_key(row: Mapping[str, Any]) -> tuple:
    """A subjective-randomness trial's condition: its unordered pair, as the figure data writes it."""
    return tuple(sorted((row["sequence_a"], row["sequence_b"])))


def ng_key(row: Mapping[str, Any]) -> tuple:
    return (row["number_set"], int(row["target"]))


def with_lpd(conditions: List[Dict[str, Any]], key, lpd: Dict[str, Dict[Hashable, float]]) -> List[Dict[str, Any]]:
    """Each condition with every role's summed log predictive density (``<role>_lpd``)."""
    return [{**c, **{f"{role}_lpd": sums[key(c)] for role, sums in lpd.items()}} for c in conditions]


@dataclass
class SubjectiveRandomness:
    work_dir: Path
    """The SR figure data's work directory: ``pooled/`` holds the models, responses and fit cache."""
    figure_data: Path = FIGURE_DATA / "subjective_randomness.json"
    out: Path = FIGURE_DATA / "subjective_randomness_heldout_conditions.json"
    in_sample_tolerance: float = 0.01
    """Largest per-condition difference allowed between the in-sample P(left) from the fits' log-likelihood
    and the figure data's (posterior-predictive, thinned draws): the trial-alignment check."""


@dataclass
class NumberGame:
    work_dir: Path
    """The number-game figure data's work directory: ``pooled/`` holds the models, responses and fit cache."""
    model: str = "exception_weighted_rule_rejection"
    """The discovered model (Figure 3's ``COLUMN_MODEL``)."""
    figure_data: Path = FIGURE_DATA / "number_game.json"
    unseen: Path = here() / "data" / "paper" / "number_game_unseen_items.json"
    """Figure 2's number-game scores on the unseen items, which these predictions must reproduce."""
    out: Path = FIGURE_DATA / "number_game_heldout_conditions.json"
    chunk_rows: int = 5000


def subjective_randomness(args: SubjectiveRandomness) -> None:
    from scripts.paper import sr_figure_data as sr
    from src.models.loo_reliability import _pointwise_log_likelihood
    from src.models.pymc_inference import load_cached_fit

    figure = json.loads(args.figure_data.read_text())
    pooled = figure["pooled"]
    work = args.work_dir / sr.POOLED_DIR
    responses = work / "responses.csv"
    rows = sr._read_rows(responses)
    if len(rows) != pooled["n_trials"]:
        raise RuntimeError(f"{responses} has {len(rows)} trials, the figure data {pooled['n_trials']}")
    chose_left = np.array([int(r["chose_left"]) for r in rows])
    keys = [sr_key(r) for r in rows]
    models = {role: pooled[role] for role in ROLES}
    loo_p, in_p, lpd, fingerprints = {}, {}, {}, {}
    for role, name in models.items():
        fitted = load_cached_fit(name, work / "models", responses, cache_dir=work / "fit_cache",
                                 **pooled["fit_settings"])
        loo_i = np.asarray(fitted.loo_diagnostics().loo.loo_i, dtype="float64").reshape(-1)
        if not np.isclose(loo_i.sum(), pooled["elpd"][name], rtol=0, atol=1e-6):
            raise RuntimeError(f"{name}: ELPD-LOO {loo_i.sum()} is not the figure data's {pooled['elpd'][name]}")
        loo_p[role] = loo_p_left(loo_i, chose_left)
        in_p[role] = in_sample_p_left(_pointwise_log_likelihood(fitted.idata), chose_left)
        lpd[role] = cell_sums(keys, loo_i)
        fingerprints[name] = fitted.fingerprint
        print(f"[sr] {name}: fit {fitted.fingerprint}, ELPD-LOO {loo_i.sum():.3f}", flush=True)

    check = sr.condition_table(rows, in_p)
    expected = figure["conditions"]
    if [(c["pair"], c["n"]) for c in check] != [(c["pair"], c["n"]) for c in expected]:
        raise RuntimeError("the pooled responses' conditions are not the figure data's")
    worst = max(abs(a[r] - b[r]) for a, b in zip(check, expected) for r in ROLES)
    if worst > args.in_sample_tolerance:
        raise RuntimeError(f"in-sample P(left) from the log-likelihood differs from the figure data's by {worst}")
    print(f"[sr] trial alignment checked: in-sample conditions within {worst:.4f} of the figure data's", flush=True)

    conditions = with_lpd(sr.condition_table(rows, loo_p), lambda c: tuple(c["pair"]), lpd)
    args.out.write_text(json.dumps({
        "domain": "subjective_randomness",
        "split": "leave-one-out (PSIS-LOO) on all runs' final data pooled, Figure 2's ELPD-LOO: each trial's "
                 "LOO P(left), averaged over each unordered pair; <role>_lpd: the pair's summed loo_i",
        "models": models,
        "fit_fingerprints": fingerprints,
        "fit_settings": pooled["fit_settings"],
        "in_sample_check_max_abs_diff": worst,
        "conditions": conditions,
    }, indent=1))
    print(f"wrote {args.out}", flush=True)


def number_game(args: NumberGame) -> None:
    from scripts.paper import number_game_figure_data as ng
    from scripts.paper import number_game_unseen_items as unseen_items
    from src.models.pymc_inference import load_cached_fit
    from src.number_game.dataset import _cells

    figure = json.loads(args.figure_data.read_text())
    record = json.loads(args.unseen.read_text())
    settings = figure["pooled_fit_settings"]
    if record["fit_settings"] != settings:
        raise RuntimeError(f"{args.unseen}'s fits {record['fit_settings']} are not the figure data's {settings}")
    test = record["unseen_items"]
    models = {"best_model": args.model, "best_seed": test["best_seed"]}
    if figure["pooled"]["best_seed"] != test["best_seed"]:
        raise RuntimeError(f"the figure data's best seed {figure['pooled']['best_seed']} is not Figure 2's "
                           f"on the unseen items, {test['best_seed']}")
    work = args.work_dir / ng.POOLED_DIR
    responses = work / "responses.csv"
    all_rows = [r for cell in _cells().values() for r in cell]
    rows, _ = unseen_items.unseen_responses(all_rows, unseen_items.read_rows(responses))
    if len(rows) != test["n_responses"]:
        raise RuntimeError(f"{len(rows)} unseen responses, Figure 2's test set has {test['n_responses']}")
    ratings = np.array([r["rating"] for r in rows])
    keys = [ng_key(r) for r in rows]
    p_yes, lpd, fingerprints = {}, {}, {}
    for role, name in models.items():
        fitted = load_cached_fit(name, work / "models", responses, cache_dir=work / "fit_cache", **settings)
        problems = fitted.convergence_problems()
        if problems:
            raise RuntimeError(f"{name}'s pooled fit did not converge: {problems}")
        draws = unseen_items.predict(fitted, rows, draws=record["prediction_draws"], chunk_rows=args.chunk_rows)
        point = unseen_items.pointwise_lpd(draws, ratings)
        if not np.isclose(point.sum(), test["models"][name]["lpd"], rtol=1e-9, atol=0):
            raise RuntimeError(f"{name}: lpd {point.sum()} on the unseen items is not Figure 2's "
                               f"{test['models'][name]['lpd']}")
        p_yes[role], lpd[role] = draws.mean(axis=0), cell_sums(keys, point)
        fingerprints[name] = fitted.fingerprint
        print(f"[ng] {name}: fit {fitted.fingerprint}, lpd {point.sum():.3f} (Figure 2's reproduced)", flush=True)
        del draws

    conditions = with_lpd(ng.condition_table(rows, p_yes), lambda c: (c["number_set"], c["target"]), lpd)
    if len(conditions) != test["n_items"]:
        raise RuntimeError(f"{len(conditions)} items, Figure 2's test set has {test['n_items']}")
    args.out.write_text(json.dumps({
        "domain": "number_game",
        "split": "the Bigelow & Piantadosi responses whose (set, target) item no run drew (Figure 2's test set): "
                 "posterior-mean P(yes) of the pooled full-data fit, averaged over each item's responses; "
                 "<role>_lpd: the item's summed log predictive density",
        "models": models,
        "fit_fingerprints": fingerprints,
        "fit_settings": settings,
        "prediction_draws": record["prediction_draws"],
        "conditions": conditions,
    }, separators=(",", ":")))
    print(f"wrote {args.out}", flush=True)


def main(args: Union[SubjectiveRandomness, NumberGame]) -> None:
    (subjective_randomness if isinstance(args, SubjectiveRandomness) else number_game)(args)


if __name__ == "__main__":
    main(tyro.cli(Union[SubjectiveRandomness, NumberGame]))
