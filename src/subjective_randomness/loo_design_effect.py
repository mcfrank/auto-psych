"""The LOO design effect: what unit should the inner loop prune on?

The inner loop prunes a model when it is statistically distinguishable from
the best on the current data: ``elpd_diff > 2·dse`` in ``az.compare``'s
PSIS-LOO table (``model_zoo._prune_losers``). That ``dse`` is
``sqrt(n · var(diff_i))`` over the *trials* — every participant's response to
every stimulus counted as an independent observation. But a holdout cell's
40 participants all answer the same 64 stimuli, and the models differ
*by stimulus*: whether model A beats model B on a trial is mostly decided by
which stimulus it is, so the pointwise differences within a stimulus are
strongly correlated and the trial-level ``dse`` understates the uncertainty of
the difference. How much is a design effect, and it is measured here rather
than assumed.

This module re-reads a finished holdout cell (its archived run record and its
``mcmc_cache/``; **no new MCMC**) and, for every scoring step the loop
recorded, rebuilds the comparison the loop made and reports it under three
units:

* **trial** — the loop's own numbers (``elpd_diff``, ``dse``), reproduced from
  the cached fits and checked against the recorded ELPDs and ledger margins,
  so the mapping from cache file to decision is verified, not assumed;
* **cluster** — the same trial-level ELPD difference with a standard error
  that treats each stimulus as one cluster: the pointwise differences are
  summed within stimulus and ``dse_cluster = sqrt(G · var(sum_g))`` over the
  ``G`` stimuli. ``dse_cluster / dse_trial`` is the design effect on the
  standard-error scale (its square is the classic variance design effect);
  the estimand is unchanged;
* **grouped** — a leave-one-stimulus-out PSIS-LOO: the log-likelihood draws
  are summed within stimulus and PSIS-LOO is run over the stimuli, giving a
  different estimand (predicting all responses to a *new* stimulus) with its
  own ``elpd_diff``, ``dse`` and Pareto-k reliability verdict.

Each unit yields a prune set under the loop's rule (``prune_set`` replicates
``_prune_losers``: non-protected, PSIS-LOO-reliable, ``elpd_diff >
multiplier·dse``, and nothing at all when the rank-0 baseline is
unreliable); the per-cell summary counts how many of the loop's decisions
would flip. The comparison set at each step is held as archived (the models
the step scored plus those it pruned): a model kept under another unit would
have stayed in later comparisons, which this offline reading cannot replay.

A stimulus is an ordered ``(sequence_a, sequence_b)`` pair as presented; the
mirrored pair is a different presentation and a different cluster.

Nothing here changes pruning. The recommendation is the memo's, for the user.
"""

from __future__ import annotations

import csv
import json
import re
import tarfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from src.models.loo_reliability import LooDiagnostics, loo_diagnostics
from src.models.pymc_inference import (
    cached_fit_path,
    fit_fingerprint,
    resolve_fit_settings,
)
from src.pipelines.inner_loop.model_zoo import (
    DEFAULT_PRUNE_DSE_MULTIPLIER,
    parse_prune_margin,
)
from src.subjective_randomness.incumbent import (
    histories_from_run_tree,
    starting_model_set,
)

STIMULUS_COLUMNS = ("sequence_a", "sequence_b")

# The run-record members an analysis needs, relative to ``experiment<n>/``.
_RUN_MEMBER_RE = re.compile(
    r"^(?P<prefix>.*?)experiment(?P<num>\d+)/model_loop/"
    r"(responses\.csv|history\.json|attempted_hypotheses\.jsonl"
    r"|models/[^/]+\.py|models/pruned/[^/]+\.py)$"
)

# A recorded ELPD is rounded to four decimals; a recomputed one must agree to
# well within that, or the cached fit is not the fit the loop scored.
_ELPD_MATCH_TOLERANCE = 0.01
# A ledger margin is rounded to one decimal.
_MARGIN_MATCH_TOLERANCE = 0.06


# ─────────────────────────────────────────────
# The units
# ─────────────────────────────────────────────


def stimulus_groups(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """One integer per row: rows presenting the same ordered
    ``(sequence_a, sequence_b)`` pair share an id; ids are numbered in order
    of first appearance. Raises ``KeyError`` naming a missing stimulus column."""
    ids: Dict[tuple, int] = {}
    out = np.empty(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        for column in STIMULUS_COLUMNS:
            if column not in row:
                raise KeyError(
                    f"responses row {i} lacks the stimulus column {column!r}; "
                    f"stimulus grouping needs {STIMULUS_COLUMNS}."
                )
        key = tuple(str(row[column]) for column in STIMULUS_COLUMNS)
        out[i] = ids.setdefault(key, len(ids))
    return out


def grouped_log_likelihood(log_lik: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Sum the trailing (trial) axis of ``log_lik`` within each group: the
    log-likelihood of every response to a stimulus, per draw."""
    groups = np.asarray(groups)
    if log_lik.shape[-1] != groups.shape[0]:
        raise ValueError(
            f"log-likelihood has {log_lik.shape[-1]} trials but {groups.shape[0]} "
            "group labels."
        )
    n_groups = int(groups.max()) + 1
    one_hot = np.zeros((groups.shape[0], n_groups))
    one_hot[np.arange(groups.shape[0]), groups] = 1.0
    return np.asarray(log_lik, dtype="float64") @ one_hot


def elpd_difference(best_i: np.ndarray, other_i: np.ndarray) -> tuple:
    """``(elpd_diff, dse)`` of ``other`` behind ``best`` from their pointwise
    ELPDs, exactly as ``az.compare`` computes them (``sqrt(n · var)``, ddof 0)."""
    diff = np.asarray(best_i, dtype="float64") - np.asarray(other_i, dtype="float64")
    return float(diff.sum()), float(np.sqrt(diff.shape[0] * np.var(diff)))


def cluster_dse(best_i: np.ndarray, other_i: np.ndarray, groups: np.ndarray) -> float:
    """The standard error of the same ELPD difference with each group (stimulus)
    as one observation: ``sqrt(G · var(sum_g diff_i))``."""
    diff = np.asarray(best_i, dtype="float64") - np.asarray(other_i, dtype="float64")
    sums = np.bincount(np.asarray(groups), weights=diff)
    return float(np.sqrt(sums.shape[0] * np.var(sums)))


def comparison_models(step: Mapping[str, Any]) -> List[str]:
    """The models a recorded step compared: the survivors it scored plus the
    models it pruned (``_prune_losers`` runs before the entry is written)."""
    return sorted(set(step["elpd_loo"]) | set(step.get("pruned") or []))


def prune_set(
    table: Mapping[str, Mapping[str, Any]],
    *,
    protected: Collection[str],
    dse_multiplier: float = DEFAULT_PRUNE_DSE_MULTIPLIER,
) -> List[str]:
    """The loop's pruning rule over a comparison table (``model_zoo._prune_losers``):
    nothing when the rank-0 baseline's PSIS-LOO is unreliable; otherwise every
    non-protected, reliable model with ``dse > 0`` and
    ``elpd_diff > dse_multiplier·dse``. Sorted by name."""
    baseline = min(table, key=lambda name: table[name]["rank"])
    if table[baseline]["unreliable"]:
        return []
    return sorted(
        name
        for name, row in table.items()
        if name not in protected
        and not row["unreliable"]
        and row["dse"] > 0
        and row["elpd_diff"] > dse_multiplier * row["dse"]
    )


def compare_unit(diagnostics: Mapping[str, LooDiagnostics]) -> Dict[str, Dict[str, Any]]:
    """``az.compare`` over the diagnostics' pointwise ELPDs, one row per model:
    ``rank``, ``elpd_loo``, ``elpd_diff``, ``dse`` and the reliability verdict."""
    import arviz as az

    cmp = az.compare({name: d.loo for name, d in diagnostics.items()}, ic="loo")
    table: Dict[str, Dict[str, Any]] = {}
    for rank, (name, row) in enumerate(cmp.iterrows()):
        diag = diagnostics[name]
        table[name] = {
            "rank": rank,
            "elpd_loo": float(row["elpd_loo"]),
            "elpd_diff": float(row["elpd_diff"]),
            "dse": float(row["dse"]),
            "unreliable": bool(diag.unreliable),
            "frac_bad_k": float(diag.frac_bad_k),
            "n_bad_k": int(diag.n_bad_k),
        }
    return table


# ─────────────────────────────────────────────
# One cached fit under every unit
# ─────────────────────────────────────────────


@dataclass(frozen=True)
class FitUnits:
    """One cached fit's PSIS-LOO at trial level and leave-one-stimulus-out."""

    name: str
    trial: LooDiagnostics
    grouped: LooDiagnostics


def load_fit_units(nc_path: Path, groups: np.ndarray, *, name: str) -> FitUnits:
    """Read a cached ``.nc`` (posterior and log-likelihood groups only) and score
    it under both PSIS-LOO units. The posterior group is kept so ``az.loo``
    derives the relative efficiency exactly as the loop's scoring did."""
    import arviz as az
    import xarray as xr

    nc_path = Path(nc_path)
    posterior = xr.open_dataset(nc_path, group="posterior", engine="h5netcdf").load()
    log_lik = xr.open_dataset(nc_path, group="log_likelihood", engine="h5netcdf").load()
    variables = list(log_lik.data_vars)
    if len(variables) != 1:
        raise ValueError(
            f"{nc_path}: expected exactly one log_likelihood variable, got {variables}."
        )
    [variable] = variables
    trial_arr = log_lik[variable]
    if trial_arr.shape[-1] != groups.shape[0]:
        raise ValueError(
            f"{nc_path}: the fit has {trial_arr.shape[-1]} trials but the "
            f"responses file has {groups.shape[0]} rows."
        )
    grouped_arr = grouped_log_likelihood(trial_arr.values, groups)
    grouped = xr.Dataset(
        {variable: (("chain", "draw", "stimulus"), grouped_arr)},
        coords={
            "chain": log_lik.chain.values,
            "draw": log_lik.draw.values,
            "stimulus": np.arange(grouped_arr.shape[-1]),
        },
    )
    with warnings.catch_warnings():
        # arviz's blanket any-k>0.7 warning is the verdict loo_diagnostics replaces.
        warnings.simplefilter("ignore")
        trial_diag = loo_diagnostics(az.InferenceData(posterior=posterior, log_likelihood=log_lik))
        grouped_diag = loo_diagnostics(az.InferenceData(posterior=posterior, log_likelihood=grouped))
    return FitUnits(name=name, trial=trial_diag, grouped=grouped_diag)


# ─────────────────────────────────────────────
# Reading a cell's run record
# ─────────────────────────────────────────────


def _materialise_run_root(cell_dir: Path, work_dir: Path) -> Path:
    """The ``_runs/<gt>/`` root holding ``experiment<n>/model_loop/...``: the kept
    repo copy when the cell has one, else the needed members extracted from
    ``agent_runs.tar.gz`` into ``work_dir`` (never into the sweep)."""
    archive = cell_dir / "agent_runs.tar.gz"
    if archive.exists():
        prefixes = set()
        work_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            members = []
            for member in tar.getmembers():
                match = _RUN_MEMBER_RE.match(member.name)
                if match and member.isfile():
                    prefixes.add(match.group("prefix"))
                    members.append(member)
            if len(prefixes) != 1:
                raise ValueError(
                    f"{archive} holds run records under {sorted(prefixes)}; a cell "
                    "archive must hold exactly one run."
                )
            # The data filter (3.12+, 3.11.4+) refuses members that escape the
            # target directory; older interpreters extract without it.
            extract_kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
            tar.extractall(work_dir, members=members, **extract_kwargs)
        return work_dir / prefixes.pop()
    run_roots = sorted(
        path.parent.parent.parent
        for path in cell_dir.glob("repo/_runs/*/experiment1/model_loop/history.json")
    )
    if len(run_roots) != 1:
        raise FileNotFoundError(
            f"{cell_dir} has neither agent_runs.tar.gz nor exactly one kept repo "
            f"copy under repo/_runs/<gt>/ (found {[str(r) for r in run_roots]})."
        )
    return run_roots[0]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no response rows.")
    return rows


def _ledger_prune_margins(ledger_path: Path, experiment: int) -> Dict[tuple, float]:
    """``(iteration, model) -> nats behind the baseline`` for every prune the
    ledger records in this experiment (its lines from earlier experiments are
    inherited and skipped)."""
    margins: Dict[tuple, float] = {}
    context_re = re.compile(rf"^experiment{experiment} round (?P<iteration>\d+)$")
    for line in Path(ledger_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry["outcome"] != "pruned":
            continue
        match = context_re.match(entry["context"])
        if match is None:
            continue
        margins[(int(match.group("iteration")), entry["name"])] = parse_prune_margin(
            entry["detail"]
        )
    return margins


def _cell_fit_kwargs(cell_dir: Path, explicit: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The sampler kwargs the cell's fits were cached under: ``holdout.json``'s
    record when the cell finished, else the caller's; both must agree."""
    recorded: Optional[Dict[str, Any]] = None
    holdout = cell_dir / "holdout.json"
    if holdout.exists():
        record = json.loads(holdout.read_text(encoding="utf-8"))
        if "fit_kwargs" not in record:
            raise KeyError(f"{holdout} records no fit_kwargs.")
        recorded = dict(record["fit_kwargs"])
    if recorded is not None and explicit is not None and dict(explicit) != recorded:
        raise ValueError(
            f"{holdout} records fit_kwargs {recorded} but {dict(explicit)} were given."
        )
    if recorded is not None:
        return recorded
    if explicit is not None:
        return dict(explicit)
    raise FileNotFoundError(
        f"{cell_dir} has no holdout.json to read fit_kwargs from and none were "
        "given; the cache fingerprints cannot be computed."
    )


def _locate_fit(
    name: str,
    models_dir: Path,
    responses: Path,
    cache_dir: Path,
    fit_kwargs: Mapping[str, Any],
) -> Path:
    """The cached fit of ``name`` on ``responses``: its source is in the
    experiment's ``models/`` (a survivor) or ``models/pruned/``."""
    candidates = [models_dir / f"{name}.py", models_dir / "pruned" / f"{name}.py"]
    sources = [path for path in candidates if path.exists()]
    if len(sources) != 1:
        raise FileNotFoundError(
            f"Expected exactly one source for model {name!r} among "
            f"{[str(p) for p in candidates]}, found {[str(p) for p in sources]}."
        )
    [source] = sources
    settings = resolve_fit_settings(name, source.parent, dict(fit_kwargs))
    nc_path = cached_fit_path(cache_dir, name, fit_fingerprint(name, source.parent, responses, settings))
    if not nc_path.exists():
        raise FileNotFoundError(
            f"No cached fit for model {name!r} on {responses} under {cache_dir}: "
            f"expected {nc_path.name}. The loop scored this model, so its fit "
            "must be in the cache; a missing file means the cache and the run "
            "record disagree."
        )
    return nc_path


# ─────────────────────────────────────────────
# One step, one experiment, one cell
# ─────────────────────────────────────────────


def step_record(
    step: Mapping[str, Any],
    fits: Mapping[str, FitUnits],
    groups: np.ndarray,
    *,
    protected: Collection[str],
    dse_multiplier: float,
) -> Dict[str, Any]:
    """The step's comparison under every unit, with each unit's prune set."""
    names = comparison_models(step)
    trial = compare_unit({name: fits[name].trial for name in names})
    grouped = compare_unit({name: fits[name].grouped for name in names})
    best_trial = min(trial, key=lambda name: trial[name]["rank"])
    best_grouped = min(grouped, key=lambda name: grouped[name]["rank"])
    best_i = np.asarray(fits[best_trial].trial.loo.loo_i)
    cluster = {
        name: {
            **trial[name],
            "dse": (
                0.0
                if name == best_trial
                else cluster_dse(best_i, np.asarray(fits[name].trial.loo.loo_i), groups)
            ),
        }
        for name in names
    }
    grouped_reliable = {
        name: {**row, "unreliable": False} for name, row in grouped.items()
    }
    pruned = {
        "trial": prune_set(trial, protected=protected, dse_multiplier=dse_multiplier),
        "cluster": prune_set(cluster, protected=protected, dse_multiplier=dse_multiplier),
        "grouped": prune_set(grouped, protected=protected, dse_multiplier=dse_multiplier),
        "grouped_ignoring_reliability": prune_set(
            grouped_reliable, protected=protected, dse_multiplier=dse_multiplier
        ),
    }

    def ratio(numerator: float, denominator: float) -> Optional[float]:
        return None if denominator <= 0 else numerator / denominator

    rows = []
    for name in names:
        t, c, g = trial[name], cluster[name], grouped[name]
        rows.append(
            {
                "name": name,
                "protected": name in protected,
                "rank_trial": t["rank"],
                "elpd_loo_trial": t["elpd_loo"],
                "elpd_diff_trial": t["elpd_diff"],
                "dse_trial": t["dse"],
                "unreliable_trial": t["unreliable"],
                "frac_bad_k_trial": t["frac_bad_k"],
                "prune_trial": name in pruned["trial"],
                "dse_cluster": c["dse"],
                "ratio_cluster": ratio(c["dse"], t["dse"]),
                "prune_cluster": name in pruned["cluster"],
                "rank_grouped": g["rank"],
                "elpd_loo_grouped": g["elpd_loo"],
                "elpd_diff_grouped": g["elpd_diff"],
                "dse_grouped": g["dse"],
                "unreliable_grouped": g["unreliable"],
                "frac_bad_k_grouped": g["frac_bad_k"],
                "ratio_grouped": ratio(g["dse"], t["dse"]),
                "prune_grouped": name in pruned["grouped"],
                "prune_grouped_ignoring_reliability": name
                in pruned["grouped_ignoring_reliability"],
            }
        )
    return {
        "step": int(step["step"]),
        "iteration": step["iteration"],
        "best_model": step["best_model"],
        "best_trial": best_trial,
        "best_grouped": best_grouped,
        "archived_pruned": sorted(step.get("pruned") or []),
        "pruned": pruned,
        "rows": rows,
    }


def _check_step_against_record(
    record: Mapping[str, Any],
    step: Mapping[str, Any],
    margins: Mapping[tuple, float],
    *,
    where: str,
) -> None:
    """The reproduced trial-level comparison must be the loop's own: recorded
    ELPDs agree at every step, and at a round step the archived prune set and
    the ledger margins agree too — or the cached fits are not the fits the
    loop scored."""
    by_name = {row["name"]: row for row in record["rows"]}
    for name, recorded in step["elpd_loo"].items():
        recomputed = by_name[name]["elpd_loo_trial"]
        if abs(recomputed - float(recorded)) > _ELPD_MATCH_TOLERANCE:
            raise ValueError(
                f"{where}: recomputed ELPD-LOO of {name!r} is {recomputed:.4f} but "
                f"history.json recorded {recorded}; the cached fit is not the fit "
                "the loop scored."
            )
    if step["iteration"] is None:
        # The seed step scores the set as carried and never prunes, however far
        # behind a carried model is; the rule's verdict there is reported but
        # is neither a decision nor something the record can disagree with.
        if record["archived_pruned"]:
            raise ValueError(
                f"{where}: a seed step recorded pruned models "
                f"{record['archived_pruned']}; the loop prunes only after a round."
            )
        return
    if record["pruned"]["trial"] != record["archived_pruned"]:
        raise ValueError(
            f"{where}: the loop's rule over the reproduced comparison prunes "
            f"{record['pruned']['trial']} but the run record pruned "
            f"{record['archived_pruned']}."
        )
    for name in record["archived_pruned"]:
        key = (int(step["iteration"]), name)
        if key not in margins:
            raise ValueError(
                f"{where}: history.json pruned {name!r} but the ledger records "
                "no prune margin for it."
            )
        recomputed = by_name[name]["elpd_diff_trial"]
        if abs(recomputed - margins[key]) > _MARGIN_MATCH_TOLERANCE:
            raise ValueError(
                f"{where}: recomputed elpd_diff of {name!r} is {recomputed:.2f} "
                f"but the ledger recorded {margins[key]} nats."
            )


def _experiment_dirs(run_root: Path, n_experiments: int) -> List[Path]:
    dirs = [run_root / f"experiment{n}" for n in range(1, n_experiments + 1)]
    for path in dirs:
        if not (path / "model_loop" / "responses.csv").exists():
            raise FileNotFoundError(f"{path / 'model_loop' / 'responses.csv'} is missing.")
    return dirs


def analyze_experiment(
    exp_dir: Path,
    history: Sequence[Mapping[str, Any]],
    cache_dir: Path,
    *,
    experiment: int,
    protected: Collection[str],
    fit_kwargs: Mapping[str, Any],
    dse_multiplier: float,
) -> Dict[str, Any]:
    """Every scoring step of one experiment under every unit."""
    loop = Path(exp_dir) / "model_loop"
    responses = loop / "responses.csv"
    rows = _read_rows(responses)
    groups = stimulus_groups(rows)
    margins = _ledger_prune_margins(loop / "attempted_hypotheses.jsonl", experiment)
    needed = sorted(set().union(*(comparison_models(step) for step in history)))
    fits: Dict[str, FitUnits] = {}
    for name in needed:
        nc_path = _locate_fit(name, loop / "models", responses, cache_dir, fit_kwargs)
        fits[name] = load_fit_units(nc_path, groups, name=name)
    steps = []
    for step in history:
        record = step_record(
            step, fits, groups, protected=protected, dse_multiplier=dse_multiplier
        )
        _check_step_against_record(
            record, step, margins, where=f"{exp_dir} step {step['step']}"
        )
        steps.append(record)
    n_stimuli = int(groups.max()) + 1
    return {
        "experiment": experiment,
        "n_trials": len(rows),
        "n_stimuli": n_stimuli,
        "trials_per_stimulus": len(rows) / n_stimuli,
        "fits": {
            name: {
                "trial": _diagnostic_summary(units.trial),
                "grouped": _diagnostic_summary(units.grouped),
            }
            for name, units in fits.items()
        },
        "steps": steps,
    }


def _diagnostic_summary(diag: LooDiagnostics) -> Dict[str, Any]:
    return {
        "elpd_loo": diag.elpd_loo,
        "n_points": diag.n_points,
        "n_exact": diag.n_exact,
        "n_bad_k": diag.n_bad_k,
        "frac_bad_k": diag.frac_bad_k,
        "unreliable": diag.unreliable,
    }


def _quantiles(values: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None], dtype="float64")
    if arr.size == 0:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarise_cell(experiments: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Per-cell counts: the loop's prune decisions (every non-protected model at
    a round step), how many each unit prunes, how many flip, and the
    distribution of the dse ratios over every non-best comparison row."""
    decisions = [
        row
        for experiment in experiments
        for step in experiment["steps"]
        if step["iteration"] is not None
        for row in step["rows"]
        if not row["protected"]
    ]
    non_best = [
        row
        for experiment in experiments
        for step in experiment["steps"]
        for row in step["rows"]
        if row["rank_trial"] != 0
    ]
    archived_pruned = sum(
        len(step["archived_pruned"]) for experiment in experiments for step in experiment["steps"]
    )

    def count(key: str) -> int:
        return sum(1 for row in decisions if row[key])

    return {
        "n_decisions": len(decisions),
        "n_pruned_archived": archived_pruned,
        "n_prune_trial": count("prune_trial"),
        "n_prune_cluster": count("prune_cluster"),
        "n_prune_grouped": count("prune_grouped"),
        "n_prune_grouped_ignoring_reliability": count("prune_grouped_ignoring_reliability"),
        "n_flip_trial_vs_cluster": sum(
            1 for row in decisions if row["prune_trial"] != row["prune_cluster"]
        ),
        "n_flip_trial_vs_grouped": sum(
            1 for row in decisions if row["prune_trial"] != row["prune_grouped"]
        ),
        "n_flip_trial_vs_grouped_ignoring_reliability": sum(
            1
            for row in decisions
            if row["prune_trial"] != row["prune_grouped_ignoring_reliability"]
        ),
        "n_grouped_unreliable_decisions": count("unreliable_grouped"),
        "ratio_cluster": _quantiles(row["ratio_cluster"] for row in non_best),
        "ratio_grouped": _quantiles(row["ratio_grouped"] for row in non_best),
        "ratio_cluster_decisions": _quantiles(row["ratio_cluster"] for row in decisions),
        "ratio_grouped_decisions": _quantiles(row["ratio_grouped"] for row in decisions),
    }


def analyze_cell(
    cell_dir: Path,
    work_dir: Path,
    *,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
    dse_multiplier: float = DEFAULT_PRUNE_DSE_MULTIPLIER,
) -> Dict[str, Any]:
    """The full record for one holdout cell ``run<r>/<gt>/``: every experiment,
    every step, every unit, plus the per-cell summary. Reads the cell's run
    record and ``mcmc_cache/``; writes only under ``work_dir``."""
    cell_dir = Path(cell_dir)
    label = f"{cell_dir.parent.name}/{cell_dir.name}"
    cache_dir = cell_dir / "mcmc_cache"
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"{cell_dir} has no mcmc_cache/ to read fits from.")
    kwargs = _cell_fit_kwargs(cell_dir, fit_kwargs)
    run_root = _materialise_run_root(cell_dir, Path(work_dir) / label.replace("/", "__"))
    histories = histories_from_run_tree(run_root)
    if not histories:
        raise FileNotFoundError(f"{run_root} holds no experiment*/model_loop/history.json.")
    protected = sorted(starting_model_set(histories[0]))
    experiments = [
        analyze_experiment(
            exp_dir,
            history,
            cache_dir,
            experiment=number,
            protected=protected,
            fit_kwargs=kwargs,
            dse_multiplier=dse_multiplier,
        )
        for number, (exp_dir, history) in enumerate(
            zip(_experiment_dirs(run_root, len(histories)), histories), start=1
        )
    ]
    return {
        "cell": label,
        "gt": cell_dir.name,
        "archived": (cell_dir / "agent_runs.tar.gz").exists(),
        "fit_kwargs": kwargs,
        "dse_multiplier": dse_multiplier,
        "protected": protected,
        "experiments": experiments,
        "summary": summarise_cell(experiments),
    }
