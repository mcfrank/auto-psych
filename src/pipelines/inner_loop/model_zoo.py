"""Model-set ("zoo") helpers for the PyMC inner loop.

Functions that manage the surviving model set: manifest I/O, candidate naming,
seeding, admission, pruning, and the novelty gate. Extracted from
``pymc_orchestrator.py`` — behaviour is identical; only the file location changed.
"""

from __future__ import annotations

import csv
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from src.models.model_manifest import manifest_path, read_manifest_entries
from src.models.data_binding import MissingStimulusColumns, make_stim_data
from src.models.model_loading import load_pymc_model, pm_data_inputs
from src.models.pymc_inference import (
    convergence_problems_of,
    evict_fit_cache,
    fit_model,
    fit_models_to_cache,
    model_logp_is_finite,
)
from src.model_comparison.likelihood import log_likelihood
from src.model_comparison.posterior import compare_table
from src.pipelines.inner_loop.hypothesis_ledger import (
    HypothesisLedger,
    LedgerEntry,
    collapse_whitespace,
)
from src.pipelines.inner_loop.import_gate import (
    CANDIDATE_IMPORT_ALLOWLIST,
    check_forbidden_imports,
)


class AllCandidatesNoFileError(RuntimeError):
    """Every candidate slot in a round produced no ``candidate.py``.

    This is the signature of a configuration bug (a missing opencode ``write``
    permission, a cwd outside the worktree, etc.), not a run of bad luck.
    Continuing would silently produce a sweep whose model set never grew.
    """


_NO_FILE_DETAIL = "no candidate.py written"

MAX_EMPTY_ROUND_RETRIES = 1


def _is_all_no_file_round(round_results: list[dict]) -> bool:
    """True when a non-empty round produced zero candidates and every slot is no-file.

    ``round_results`` is a list of dicts with ``outcome`` and ``detail`` keys,
    one per candidate slot (including spawn failures recorded as
    ``outcome='spawn_failed'``).
    """
    if not round_results:
        return False
    n_admitted = sum(1 for r in round_results if r["outcome"] == "admitted")
    if n_admitted > 0:
        return False
    no_file_reasons = {_NO_FILE_DETAIL, "agent process failed"}
    return all(
        r["detail"] in no_file_reasons or r["outcome"] == "spawn_failed"
        for r in round_results
    )


# ─────────────────────────────────────────────
# Slot roles and the lens walk
# ─────────────────────────────────────────────

# A round's candidate slots have roles. Exploratory slots walk the lens
# battery — breadth: a new mechanism per slot. Refinement slots improve a
# model already proposed — depth: two refine the incumbent (the best model of
# the latest scoring step, named in the brief) and one refines a
# non-incumbent model of the agent's choosing, live or pruned, from a menu.
# Before this the loop had three breadth mechanisms (the novelty gate, the
# ledger's "do not re-propose", pruning) and no depth mechanism: in the
# September 2026 sweep the exported best model never changed from its
# starting seed (0 of 27 scoring steps), and a partially correct mechanism,
# once pruned, could not be revived by design. A role is the slot's
# *assignment*: it shapes the prompt and the ledger context. Which model the
# agent actually refined is stated in its hypothesis, in prose; nothing in
# the pipeline parses it or branches on it. See the decision record.
SLOT_EXPLORE = "explore"
SLOT_REFINE_INCUMBENT = "refine incumbent"
SLOT_REFINE_CHOSEN = "refine chosen"
SLOT_ROLES = (SLOT_EXPLORE, SLOT_REFINE_INCUMBENT, SLOT_REFINE_CHOSEN)


def slot_roles(candidate_count: int) -> List[str]:
    """The role of each of a round's ``candidate_count`` slots, in slot order.

    At four or more slots: ``candidate_count - 3`` exploratory slots, then
    two incumbent-refinement slots, then one agent-chosen refinement slot.
    Below four the refinement slots are given up one at a time — the second
    incumbent slot first, then the chosen slot, then the last incumbent slot
    — and the exploratory slot never is: ``[explore]`` at one slot,
    ``[explore, refine incumbent]`` at two, ``[explore, refine incumbent,
    refine chosen]`` at three. Zero slots is a round that only scores.
    """
    if candidate_count < 0:
        raise ValueError(f"candidate_count cannot be negative; got {candidate_count}.")
    n_incumbent = 2 if candidate_count >= 4 else min(1, max(candidate_count - 1, 0))
    n_chosen = 1 if candidate_count >= 3 else 0
    n_explore = candidate_count - n_incumbent - n_chosen
    return (
        [SLOT_EXPLORE] * n_explore
        + [SLOT_REFINE_INCUMBENT] * n_incumbent
        + [SLOT_REFINE_CHOSEN] * n_chosen
    )


def exploratory_slots_per_round(candidate_count: int) -> int:
    """How many of a round's slots walk the lens battery (see ``slot_roles``)."""
    return slot_roles(candidate_count).count(SLOT_EXPLORE)


def _lens_offset(exp_num: int, *, max_iterations: int, candidate_count: int) -> int:
    """The lens-schedule position at which experiment ``exp_num`` starts.

    Only exploratory slots walk the battery, so experiment k spends
    ``max_iterations * exploratory_slots_per_round(candidate_count)`` lenses
    and experiment k+1 continues the walk where k stopped.
    """
    if exp_num < 1:
        raise ValueError(f"Experiment numbers start at 1; got {exp_num}.")
    return (exp_num - 1) * max_iterations * exploratory_slots_per_round(candidate_count)


def _lens_index(
    lens_offset: int,
    iteration: int,
    exploratory_per_round: int,
    exploratory_idx: int,
    n_lenses: int,
) -> int:
    """Which lens the ``exploratory_idx``-th exploratory slot of round ``iteration`` works."""
    if n_lenses < 1:
        raise ValueError("The lens battery is empty.")
    return (lens_offset + iteration * exploratory_per_round + exploratory_idx) % n_lenses


# ─────────────────────────────────────────────
# Model-set ("zoo") helpers
# ─────────────────────────────────────────────


def _manifest_entries(models_dir: Path) -> List[Dict[str, str]]:
    """Full manifest entries (``name`` + ``rationale``) for the zoo directory.

    A missing manifest reads as an empty model set here — the zoo's manifest
    only appears once the seed set has been copied in, and several helpers
    below run against a directory that is still being built up.
    """
    return read_manifest_entries(models_dir, missing_ok=True)


def _manifest_names(models_dir: Path) -> List[str]:
    """Ordered model names from the zoo's manifest."""
    return [e["name"] for e in _manifest_entries(models_dir)]


def _write_manifest(models_dir: Path, entries: List[Dict[str, str]]) -> None:
    """Overwrite the zoo's ``models_manifest.yaml`` with ``entries``."""
    manifest_path(models_dir).write_text(
        yaml.safe_dump({"models": entries}, sort_keys=False), encoding="utf-8"
    )


# Agent-chosen model names: short snake_case slugs. The auto pattern and the
# export names are reserved so agent names never collide with pipeline
# machinery (the carried-manifest validator rejects zoo names outright).
_MODEL_NAME_RE = re.compile(r"[a-z][a-z0-9_]{2,40}")
_ZOO_NAME_RE = re.compile(r"iter\d+_candidate\d+")
_RESERVED_MODEL_NAMES = frozenset({"inner_loop_model", "best_model"})


def _resolve_candidate_name(
    candidate_dir: Path, models_dir: Path, *, fallback: str
) -> str:
    """The admitted name for a candidate: the agent's slug or the auto fallback.

    The agent writes ``model_name.txt`` (snake_case) alongside ``hypothesis.md``
    so discovered models carry meaningful, run-unique identifiers instead of
    ``iterN_candidateM`` (which collided across runs and carries no meaning). A
    missing or invalid name falls back to the auto name with a loud log — a bad
    name never sinks an otherwise good candidate. A name already in the model
    set is uniquified with a numeric suffix.
    """
    name_path = Path(candidate_dir) / "model_name.txt"
    if not name_path.exists():
        print(
            f"  [name] {name_path} not written — admitting as {fallback!r}",
            flush=True,
        )
        return fallback
    raw = name_path.read_text(encoding="utf-8").strip()
    if (
        not _MODEL_NAME_RE.fullmatch(raw)
        or _ZOO_NAME_RE.fullmatch(raw)
        or raw in _RESERVED_MODEL_NAMES
    ):
        print(
            f"  [name] invalid model name {raw!r} (need a short snake_case slug, "
            f"not a reserved or auto-generated name) — admitting as {fallback!r}",
            flush=True,
        )
        return fallback
    existing = set(_manifest_names(models_dir))
    if raw in existing:
        suffix = 2
        while f"{raw}_{suffix}" in existing:
            suffix += 1
        unique = f"{raw}_{suffix}"
        print(
            f"  [name] {raw!r} is already in the model set — admitting as "
            f"{unique!r}",
            flush=True,
        )
        return unique
    return raw


def _seed_model_set(seed_models_dir: Path, models_dir: Path) -> List[Dict[str, str]]:
    """Copy every model listed in ``seed_models_dir``'s manifest into ``models_dir``.

    Returns the manifest entries (name + rationale) that were carried over.
    Fails loudly if a listed model file is missing — we never silently drop a
    seed model.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, str]] = []
    for entry in read_manifest_entries(seed_models_dir):
        name = entry["name"]
        src = seed_models_dir / f"{name}.py"
        if not src.exists():
            raise FileNotFoundError(f"Seed model {name!r} has no file at {src}")
        shutil.copyfile(src, models_dir / f"{name}.py")
        entries.append(
            {"name": name, "rationale": entry.get("rationale", "") or "Seed model."}
        )

    if not entries:
        raise ValueError(f"No seed models found in {manifest_path(seed_models_dir)}")
    _write_manifest(models_dir, entries)
    return entries


def _record(
    ledger: Optional[HypothesisLedger],
    *,
    name: str,
    outcome: str,
    detail: str,
    hypothesis: str,
    context: str,
) -> None:
    """Append one event to the hypothesis ledger (a no-op without a ledger)."""
    if ledger is None:
        return
    ledger.append(
        LedgerEntry(
            name=name,
            outcome=outcome,
            detail=collapse_whitespace(detail),
            hypothesis=collapse_whitespace(hypothesis),
            context=context,
        )
    )


def _drop_unfittable_models(
    models_dir: Path,
    responses_path: Path,
    *,
    ledger: Optional[HypothesisLedger] = None,
    ledger_context: str = "",
) -> None:
    """Remove from the manifest any seed model that cannot be MCMC-fit.

    A seed/theory model whose logp is non-finite on the data (e.g. a
    numerically unsafe construct that NaNs in PyTensor) would otherwise crash
    ``pm.sample`` at its start-value check and abort the whole run. We drop such
    models from the manifest with a loud warning rather than let one bad model
    kill a long agentic run. Fails loudly only if **no** model survives. Each
    drop is recorded in the ledger so a carried model that vanishes here is
    still accounted for.
    """
    keep: List[Dict[str, str]] = []
    for entry in _manifest_entries(models_dir):
        name = entry["name"]
        fittable, reason = model_logp_is_finite(name, models_dir, responses_path)
        if fittable:
            keep.append(entry)
        else:
            print(f"  [drop] seed model {name!r} cannot be fit — {reason}", flush=True)
            _record(
                ledger,
                name=name,
                outcome="dropped",
                detail=f"cannot be fit on this experiment's data — {reason}",
                hypothesis=entry.get("rationale") or "",
                context=ledger_context,
            )
    if not keep:
        raise ValueError(
            f"No fittable seed models remain in {models_dir} — every seed model's "
            "logp was non-finite on the data."
        )
    _write_manifest(models_dir, keep)


def _drop_nonfinite_elpd_models(
    models_dir: Path,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    ledger: Optional[HypothesisLedger] = None,
    ledger_context: str = "",
) -> None:
    """Remove from the manifest any model whose ELPD-LOO is non-finite on the data.

    ``_drop_unfittable_models`` screens only the *initial-point logp*, and
    ``_admit_candidate`` screens a *fresh candidate*'s ELPD. Neither covers a
    model carried forward from a previous experiment: it scored a finite ELPD on
    that experiment's responses but can yield a NaN/inf ELPD-LOO on this
    experiment's *different* responses (e.g. it now assigns ~0 probability to a
    newly observed outcome). Such a model slips past the logp gate and crashes
    ``model_posterior`` (which refuses to softmax a non-finite ELPD), aborting the
    whole run. Compute each model's ELPD-LOO now and drop the non-finite ones
    with a loud warning. Fails loudly only if **no** model survives.

    This is also the experiment's first MCMC pass: every model in the set meets
    this experiment's responses here for the first time, so the whole set is
    sampled in one batch (``fit_models_to_cache`` — concurrent fits, each
    persisted to ``cache_dir``) and everything downstream (the ELPD calls below,
    scoring, the critique) reuses those fits. A model whose fit fails is dropped
    on that report; it is not fit a second time.
    """
    fit_kwargs = fit_kwargs or {}
    entries = _manifest_entries(models_dir)

    def drop(entry: Dict[str, str], message: str, detail: str) -> None:
        print(f"  [drop] model {entry['name']!r}: {message}; dropping.", flush=True)
        _record(
            ledger,
            name=entry["name"],
            outcome="dropped",
            detail=detail,
            hypothesis=entry.get("rationale") or "",
            context=ledger_context,
        )

    fit_failures = fit_models_to_cache(
        [entry["name"] for entry in entries],
        models_dir,
        responses_path,
        cache_dir=cache_dir,
        **fit_kwargs,
    )
    keep: List[Dict[str, str]] = []
    for entry in entries:
        name = entry["name"]
        if name in fit_failures:
            drop(
                entry,
                f"MCMC fit failed ({fit_failures[name]}) — cannot score it",
                f"MCMC fit failed ({fit_failures[name]})",
            )
            continue
        try:
            elpd = log_likelihood(
                name, responses_path, models_dir, cache_dir=cache_dir, **fit_kwargs
            )
        except Exception as e:  # noqa: BLE001 — any fit/LOO failure means unscorable
            drop(
                entry,
                f"ELPD-LOO computation failed ({type(e).__name__}: {e}) — cannot "
                "score it",
                f"ELPD-LOO computation failed ({type(e).__name__}: {e})",
            )
            continue
        if math.isfinite(elpd):
            keep.append(entry)
        else:
            drop(
                entry,
                f"non-finite ELPD-LOO ({elpd}) on the data — would corrupt the "
                "posterior",
                f"non-finite ELPD-LOO ({elpd}) on this experiment's data",
            )
    if not keep:
        raise ValueError(
            f"No model in {models_dir} has a finite ELPD-LOO on the data — every "
            "model's PSIS-LOO was non-finite."
        )
    _write_manifest(models_dir, keep)


class NoveltyPoolUnbindable(ValueError):
    """A *candidate* binds columns a bare stimulus row never carries.

    ``_min_prediction_rmse`` raises this for the candidate only, and the
    admission gate turns it into a recorded rejection: such a model cannot be
    evaluated on any stimulus pool (the held-out evaluation included). An
    *admitted* model that cannot bind the pool is a set-level inconsistency
    and raises a plain ``RuntimeError`` instead.
    """

    def __init__(self, model_name: str, missing: Sequence[str]):
        self.model_name = model_name
        self.missing = tuple(missing)
        super().__init__(
            f"model {model_name!r} binds {list(self.missing)}, which a bare "
            "stimulus row never carries, so it cannot be compared on the novelty pool"
        )


def _participant_ids_in(responses_path: Path) -> Optional[List[int]]:
    """Distinct participant ids in the responses, sorted; ``None`` without the column."""
    with Path(responses_path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "participant_id" not in (reader.fieldnames or []):
            return None
        ids = sorted({int(row["participant_id"]) for row in reader})
    return ids or None


def _pool_prediction(
    fitted: Any,
    pool_rows: Sequence[Mapping[str, str]],
    *,
    model_name: str,
    participant_ids: Optional[Sequence[int]],
) -> np.ndarray:
    """Posterior-mean ``p_left`` of one fitted model on the pool, population-level.

    The observed-response container is filled with dummies (``p_left`` never
    reads it), as the recovery evaluation does with its pool. A model without
    a participant random effect predicts the rows directly. One that indexes
    ``participant_id`` has only per-participant ``p_left``, so each pool row
    is predicted as every participant the model was fit on and averaged — one
    participant at a time, so the draws x rows array stays one pass wide.
    """
    rows = [{**row, "chose_left": 0} for row in pool_rows]
    if "participant_id" not in pm_data_inputs(fitted.model):
        stim_data = make_stim_data(fitted.model, rows)
        return np.asarray(fitted.predict_p_left(stim_data), dtype="float64")
    if not participant_ids:
        raise ValueError(
            f"Model {model_name!r} indexes a participant_id random effect but the "
            "training responses carry no participant_id to marginalize over."
        )
    total = np.zeros(len(rows), dtype="float64")
    for pid in participant_ids:
        as_participant = [{**row, "participant_id": pid} for row in rows]
        stim_data = make_stim_data(fitted.model, as_participant)
        total += np.asarray(fitted.predict_p_left(stim_data), dtype="float64")
    return total / len(participant_ids)


def _min_prediction_rmse(
    model_name: str,
    models_dir: Path,
    responses_path: Path,
    *,
    pool_rows: Sequence[Mapping[str, str]],
    cache_dir: Optional[Path] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], float]:
    """Min RMSE between ``model_name``'s p_left and each admitted model's on ``pool_rows``.

    Predictions are posterior means on the novelty pool (``novelty_pool_rows``;
    the constants below say why it is not the training stimuli), computed from
    the cached fits on ``responses_path`` — this runs after the admission
    fit-gate and after scoring has fit every admitted model, so no new MCMC
    happens here. Returns the nearest model's name and the RMSE —
    ``(None, inf)`` when the set holds no other model. Raises
    ``NoveltyPoolUnbindable`` when the candidate cannot bind bare stimulus rows.
    """
    pool_rows = list(pool_rows)
    if not pool_rows:
        raise ValueError("The novelty pool is empty.")
    participant_ids = _participant_ids_in(responses_path)

    def posterior_mean_p_left(name: str) -> np.ndarray:
        fitted = fit_model(
            name, models_dir, responses_path, cache_dir=cache_dir, **(fit_kwargs or {})
        )
        return _pool_prediction(
            fitted, pool_rows, model_name=name, participant_ids=participant_ids
        )

    try:
        candidate_p = posterior_mean_p_left(model_name)
    except MissingStimulusColumns as e:
        raise NoveltyPoolUnbindable(model_name, e.missing) from e
    nearest: Optional[str] = None
    nearest_rmse = float("inf")
    for name in _manifest_names(models_dir):
        if name == model_name:
            continue
        try:
            other_p = posterior_mean_p_left(name)
        except MissingStimulusColumns as e:
            raise RuntimeError(
                f"Admitted model {name!r} cannot bind the novelty pool (missing "
                f"{list(e.missing)}); every model in the set must be evaluable "
                "on bare stimulus rows."
            ) from e
        rmse = float(np.sqrt(np.mean((candidate_p - other_p) ** 2)))
        if rmse < nearest_rmse:
            nearest, nearest_rmse = name, rmse
    return nearest, nearest_rmse


# The ledger ``detail`` of a prune is the margin behind the model that won,
# in a fixed form the refinement menu reads back to rank pruned models by how
# narrowly they lost (``LedgerEntry`` cannot gain a numeric field: its key set
# is fixed, so an inherited ledger would become unreadable).
_PRUNE_MARGIN_RE = re.compile(r"^(?P<nats>\d+(?:\.\d+)?) nats behind \S+")


def prune_margin_detail(*, elpd_diff: float, dse: float, baseline: str) -> str:
    """The ledger ``detail`` of a prune: nats behind ``baseline`` and the dse ratio."""
    return f"{elpd_diff:.1f} nats behind {baseline} ({elpd_diff / dse:.1f}× dse)"


def parse_prune_margin(detail: str) -> float:
    """The nats a pruned model lost by, read back from ``prune_margin_detail``'s text."""
    match = _PRUNE_MARGIN_RE.match(detail)
    if match is None:
        raise ValueError(
            "Not a prune margin (expected '<nats> nats behind <model> ...'): "
            f"{detail!r}"
        )
    return float(match.group("nats"))


# Pruning: after each scoring pass, a non-protected model is dropped when it is
# statistically distinguishable from the best (elpd_diff > multiplier·dse among
# PSIS-LOO-reliable rows). The surviving set is the uncertainty set — every
# non-protected survivor is within the margin of the best — which is what the
# outer loop carries into the next experiment. Stacking weight is deliberately
# NOT a criterion: az.compare's weights are ensemble coefficients, not
# plausibility. See the decision record for the empirical evidence.
DEFAULT_PRUNE_DSE_MULTIPLIER = 2.0

# Novelty gate: a candidate whose posterior-mean p_left is within this RMSE of
# an admitted model's — measured on the loop's novelty pool below, not on the
# training stimuli — is a re-skinned duplicate, not a new hypothesis; reject
# it at admission. Calibration (the September 2026 sweep, measured on the 64
# training stimuli at the old 0.02): the 23 archived rejection margins were
# bimodal — about five re-skins at ~0 (0.0000 x2, 0.0001, 0.0002, 0.0004; two
# predicted identically) and about eighteen spread evenly from 0.006 to 0.019,
# distinct mechanisms that happened to agree on the training points. 0.002
# sits in the gap. See the decision record. Set to 0 to disable.
DEFAULT_NOVELTY_RMSE_THRESHOLD = 0.002

# The novelty pool: the stimuli on which a candidate's predictions are compared
# to every admitted model's. The loop generates it from its own seed over the
# design's pair universe (same-length H/T pairs at lengths 4–8), so two
# mechanisms are compared across the stimulus space rather than on the few
# dozen training stimuli the design happened to select. It is deliberately NOT
# the recovery harness's eval pool (``holdout_eval.build_eval_stimuli``, seeded
# from the holdout config's ``eval_pool``): the loop must not select models on
# the stimuli it is later scored against. The orchestrator writes the pool to
# the run tree as ``novelty_pool.json`` so what the gate saw is auditable.
NOVELTY_POOL_SEED = 20260919
NOVELTY_POOL_N_PAIRS = 512
NOVELTY_POOL_LENGTHS = (4, 5, 6, 7, 8)
NOVELTY_POOL_FILENAME = "novelty_pool.json"


def novelty_pool_rows(
    *,
    seed: int = NOVELTY_POOL_SEED,
    n_pairs: int = NOVELTY_POOL_N_PAIRS,
    lengths: Sequence[int] = NOVELTY_POOL_LENGTHS,
) -> List[Dict[str, str]]:
    """The loop's own stimulus pool for the novelty gate (see the constants above).

    Raw ``sequence_a``/``sequence_b`` rows: each model computes its own
    features from them, exactly as it does for the responses CSV.
    """
    # Lazy, like the design stage's import of the same module (``eig.py``):
    # the pipeline's coupling to the research library stays thin and explicit.
    from src.subjective_randomness.stimulus_design import generate_candidate_pool

    return generate_candidate_pool(n_pairs, lengths=tuple(lengths), seed=seed)


def _untrusted(row: Dict[str, Any]) -> bool:
    """An unreliable PSIS-LOO or a fit that failed the convergence gate."""
    return bool(row.get("loo_unreliable") or row.get("not_converged"))


def _prune_losers(
    models_dir: Path,
    responses_path: Path,
    *,
    protected: set[str],
    cache_dir: Optional[Path],
    fit_kwargs: Optional[Dict[str, Any]],
    dse_multiplier: float = DEFAULT_PRUNE_DSE_MULTIPLIER,
    ledger: Optional[HypothesisLedger] = None,
    ledger_context: str = "",
) -> List[str]:
    """Drop non-protected models that have lost; return their names.

    "Lost" means statistically distinguishable from the best on the current
    data: ``elpd_diff > dse_multiplier·dse`` among PSIS-LOO-reliable rows. The
    survivors are therefore the uncertainty set — every non-protected model
    still within the margin of the best — which is what the outer loop carries
    into the next experiment. ``protected`` names (the project's seed models)
    are never pruned: they are the baselines the run reports against. Pruned
    files move to ``models/pruned/`` (an audit trail, not a deletion), the
    ledger records the margin, and the cached fits are evicted so the
    in-process memory footprint stops growing with dead models.

    Honest framing: within a run, re-scoring a loser is a cache hit, so the
    savings are memory, az.compare size, and a focused existing_hypotheses.md —
    not avoided MCMC.
    """
    if dse_multiplier <= 0:
        return []
    names = _manifest_names(models_dir)
    if len(names) < 2:
        return []
    comparison = compare_table(
        responses_path, models_dir, cache_dir=cache_dir, **(fit_kwargs or {})
    )
    if not comparison:
        return []
    # Reliability gates are deliberately narrow. Every elpd_diff is measured
    # against the rank-0 baseline, so an unreliable baseline poisons every
    # comparison and blocks all pruning. Beyond that, a model is only shielded
    # by its OWN unreliable row — agent-written candidates trip Pareto-k
    # warnings routinely, and one flaky bystander must not switch pruning off
    # wholesale (the active set would then only ever grow).
    baseline = min(comparison, key=lambda name: comparison[name]["rank"])
    if _untrusted(comparison[baseline]):
        print(
            f"  [warn] Skipping model pruning: baseline model {baseline!r} "
            "(rank 0) has an unreliable LOO estimate or a non-converged fit, so "
            "every elpd_diff against it is untrustworthy.",
            file=sys.stderr,
            flush=True,
        )
        return []
    unreliable = sorted(name for name, row in comparison.items() if _untrusted(row))
    if unreliable:
        print(
            "  [warn] Not pruning models with unreliable LOO estimates or "
            "non-converged fits: "
            + ", ".join(unreliable),
            file=sys.stderr,
            flush=True,
        )
    to_prune = [
        name
        for name in names
        if name not in protected
        and name in comparison
        and not _untrusted(comparison[name])
        and comparison[name]["dse"] > 0
        and comparison[name]["elpd_diff"] > dse_multiplier * comparison[name]["dse"]
    ]
    if not to_prune:
        return []
    details = {}
    for name in to_prune:
        row = comparison[name]
        details[name] = prune_margin_detail(
            elpd_diff=row["elpd_diff"], dse=row["dse"], baseline=baseline
        )
        print(
            f"  [prune] {name}: elpd_diff {row['elpd_diff']:.1f} > "
            f"{dse_multiplier}·dse ({row['dse']:.1f}) — {details[name]}; moved to "
            "models/pruned/.",
            flush=True,
        )
    _retire(models_dir, details, ledger=ledger, ledger_context=ledger_context)
    return to_prune


# The live set carried into the next experiment is at most this many models,
# seeds included (user decision 2026-09-26): pruning alone could let it grow
# without bound, since a model within the margin of the best is never pruned.
MAX_LIVE_MODELS = 8


def _cap_live_set(
    models_dir: Path,
    responses_path: Path,
    *,
    protected: set[str],
    cache_dir: Optional[Path],
    fit_kwargs: Optional[Dict[str, Any]],
    cap: int = MAX_LIVE_MODELS,
    ledger: Optional[HypothesisLedger] = None,
    ledger_context: str = "",
) -> List[str]:
    """Retire non-protected models until at most ``cap`` remain; return them.

    Models that cannot be trusted (an unreliable PSIS-LOO or a non-converged
    fit; they can never be exported) retire first, then the lowest by ELPD-LOO.
    Protected seeds are never retired. Retired models go where pruned ones do:
    ``models/pruned/``, and the ledger as ``pruned`` with the reason.
    """
    names = _manifest_names(models_dir)
    excess = len(names) - cap
    if excess <= 0:
        return []
    comparison = compare_table(
        responses_path, models_dir, cache_dir=cache_dir, **(fit_kwargs or {})
    )
    retirable = [name for name in names if name not in protected]
    # Untrusted first, then worst rank first.
    order = sorted(
        retirable,
        key=lambda name: (not _untrusted(comparison[name]), -comparison[name]["rank"]),
    )
    to_retire = order[:excess]
    if len(to_retire) < excess:
        print(
            f"  [warn] live set cap {cap}: {len(names)} models, but only "
            f"{len(retirable)} are not protected seeds; keeping "
            f"{len(names) - len(to_retire)}.",
            file=sys.stderr,
            flush=True,
        )
    details = {}
    for name in to_retire:
        row = comparison[name]
        if _untrusted(row):
            why = "its fit cannot be trusted (unreliable PSIS-LOO or no convergence)"
        else:
            why = (
                f"ELPD-LOO rank {row['rank'] + 1} of {len(names)}, "
                f"{row['elpd_diff']:.1f} nats behind the best"
            )
        details[name] = f"retired to keep the live set at {cap} models: {why}"
        print(f"  [cap] {name}: {details[name]}; moved to models/pruned/.", flush=True)
    _retire(models_dir, details, ledger=ledger, ledger_context=ledger_context)
    return to_retire


def _retire(
    models_dir: Path,
    details: Dict[str, str],
    *,
    ledger: Optional[HypothesisLedger],
    ledger_context: str,
) -> None:
    """Move models out of the live set: files to ``models/pruned/`` (an audit
    trail, not a deletion), fits evicted, ledger ``pruned`` with each
    model's reason, manifest rewritten without them."""
    hypotheses = {
        e["name"]: (e.get("rationale") or "") for e in _manifest_entries(models_dir)
    }
    pruned_dir = models_dir / "pruned"
    pruned_dir.mkdir(exist_ok=True)
    for name, detail in details.items():
        for suffix in (".py", ".hypothesis.md"):
            src = models_dir / f"{name}{suffix}"
            if src.exists():
                shutil.move(str(src), str(pruned_dir / f"{name}{suffix}"))
        evict_fit_cache(name)
        _record(
            ledger,
            name=name,
            outcome="pruned",
            detail=detail,
            hypothesis=hypotheses.get(name, ""),
            context=ledger_context,
        )
    _write_manifest(
        models_dir,
        [e for e in _manifest_entries(models_dir) if e["name"] not in details],
    )


@dataclass(frozen=True)
class Admission:
    """The verdict on one candidate: admitted, or rejected with the reason.

    ``reason`` is the rejection message exactly as the ledger records it — a
    repair attempt injects it verbatim into the agent's prompt (see the
    orchestrator) — and empty when the candidate was admitted.
    """

    admitted: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if self.admitted and self.reason:
            raise ValueError(f"an admitted candidate has no rejection reason; got {self.reason!r}")
        if not self.admitted and not self.reason:
            raise ValueError("a rejected candidate needs a rejection reason")


def _admit_candidate(
    candidate_file: Path,
    models_dir: Path,
    model_name: str,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    novelty_rmse_threshold: float = DEFAULT_NOVELTY_RMSE_THRESHOLD,
    novelty_pool: Optional[Sequence[Mapping[str, str]]] = None,
    ledger: Optional[HypothesisLedger] = None,
    ledger_context: str = "",
) -> bool:
    """``_admit_candidate_with_reason`` for callers that only need the verdict."""
    return _admit_candidate_with_reason(
        candidate_file,
        models_dir,
        model_name,
        responses_path,
        cache_dir=cache_dir,
        fit_kwargs=fit_kwargs,
        novelty_rmse_threshold=novelty_rmse_threshold,
        novelty_pool=novelty_pool,
        ledger=ledger,
        ledger_context=ledger_context,
    ).admitted


def _admit_candidate_with_reason(
    candidate_file: Path,
    models_dir: Path,
    model_name: str,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    novelty_rmse_threshold: float = DEFAULT_NOVELTY_RMSE_THRESHOLD,
    novelty_pool: Optional[Sequence[Mapping[str, str]]] = None,
    ledger: Optional[HypothesisLedger] = None,
    ledger_context: str = "",
) -> Admission:
    """Validate a candidate and, if valid, admit it to the model set.

    Every outcome — admitted, or rejected for any of the reasons below — is
    recorded in ``ledger`` (when given) with the candidate's hypothesis, so the
    next round's briefs can list what was already tried. The returned
    ``Admission`` carries the rejection reason so the orchestrator can hand it
    to the slot's one repair attempt.

    A candidate is admitted only when it ships **both**:

    - ``candidate.py`` that loads as a module-level ``model: pm.Model`` (via
      ``load_pymc_model``), evaluates to a finite logp on the data, **and
      actually completes an MCMC fit**, and
    - ``hypothesis.md`` next to it stating, in natural language, the single
      cognitive hypothesis the model implements.

    The hypothesis text becomes the model's manifest rationale and is copied to
    ``models/<name>.hypothesis.md`` so every model in the set carries the
    hypothesis it tests. Candidates missing either file, or whose logp is
    non-finite on the data, or whose MCMC sampling raises, are skipped (a
    rejected ``Admission``, with a loud message) so one bad agent output does
    not abort the round.

    The finite-logp check only inspects the initial point, so a candidate can
    pass it yet NaN once NUTS jitters off that point. Such a candidate, if merely
    admitted, would crash the post-admission scoring pass and take down the whole
    run — so admission ends with a real fit (its result is cached and reused by
    scoring, adding no extra MCMC), containing any sampling failure to this one
    candidate.

    The novelty gate compares the candidate's posterior-mean ``p_left`` with
    every admitted model's on ``novelty_pool`` (``None`` ⇒ the loop's default
    pool, ``novelty_pool_rows()``; the orchestrator passes the pool it recorded
    in the run tree so every candidate of a run is gated on the same stimuli).
    """
    hypothesis_file = candidate_file.parent / "hypothesis.md"
    hypothesis = (
        hypothesis_file.read_text(encoding="utf-8").strip()
        if hypothesis_file.exists()
        else ""
    )

    def reject(reason: str) -> Admission:
        print(f"  [reject] {model_name}: {reason}", flush=True)
        _record(
            ledger,
            name=model_name,
            outcome="rejected",
            detail=reason,
            hypothesis=hypothesis,
            context=ledger_context,
        )
        return Admission(admitted=False, reason=reason)

    if not candidate_file.exists():
        return reject("no candidate.py written")
    if not hypothesis:
        return reject(
            "no hypothesis.md — every model must state one cognitive hypothesis "
            "before it can be admitted"
        )

    source = candidate_file.read_text(encoding="utf-8")
    forbidden = check_forbidden_imports(source)
    if forbidden:
        return reject(
            f"forbidden import: {', '.join(forbidden)} — candidates may only "
            f"import from {sorted(CANDIDATE_IMPORT_ALLOWLIST)}"
        )

    staged = models_dir / f"{model_name}.py"
    shutil.copyfile(candidate_file, staged)
    try:
        load_pymc_model(model_name, models_dir)
    except Exception as e:
        staged.unlink(missing_ok=True)
        return reject(f"candidate.py is not a loadable PyMC model: {e}")

    fittable, reason = model_logp_is_finite(model_name, models_dir, responses_path)
    if not fittable:
        staged.unlink(missing_ok=True)
        return reject(f"model cannot be fit — {reason}")

    # Real-fit gate: the logp check above only covers the initial point, so a
    # candidate can pass it yet diverge/NaN once NUTS jitters off it. Fit it now
    # (cached, so scoring reuses this exact fit) to contain such a failure here
    # instead of letting it abort the round's scoring pass.
    try:
        fitted = fit_model(
            model_name,
            models_dir,
            responses_path,
            cache_dir=cache_dir,
            **(fit_kwargs or {}),
        )
    except Exception as e:
        staged.unlink(missing_ok=True)
        return reject(
            f"MCMC sampling failed ({type(e).__name__}: {e}); dropping it so it "
            "cannot abort scoring."
        )

    # Convergence gate: a fit with divergences, poor R-hat or low ESS has an
    # untrustworthy ELPD and p_left, and could otherwise prune rivals and be
    # exported. PSIS-LOO reliability does not catch it (it measures influential
    # trials, not mixing).
    problems = convergence_problems_of(fitted)
    if problems:
        staged.unlink(missing_ok=True)
        return reject(
            f"MCMC did not converge ({'; '.join(problems)}). Reparameterise the "
            "model (e.g. non-centred parameters, tighter priors, no hard "
            "thresholds in the likelihood), or declare smaller NUTS steps with a "
            "module-level SAMPLER_SETTINGS = {\"target_accept\": 0.95}."
        )

    # ELPD-LOO gate: a model can sample cleanly yet still assign ~0 probability to
    # an observed outcome at some posterior draws, giving a non-finite PSIS-LOO.
    # That NaN/inf would later crash model_posterior (which refuses to build a
    # posterior a non-finite ELPD would corrupt). Compute it now from the cached
    # fit (no extra sampling) and drop the candidate here instead.
    try:
        elpd = log_likelihood(
            model_name,
            responses_path,
            models_dir,
            cache_dir=cache_dir,
            **(fit_kwargs or {}),
        )
    except Exception as e:
        staged.unlink(missing_ok=True)
        return reject(
            f"ELPD-LOO computation failed ({type(e).__name__}: {e}); dropping it "
            "so it cannot abort scoring."
        )
    if not math.isfinite(elpd):
        staged.unlink(missing_ok=True)
        return reject(
            f"non-finite ELPD-LOO ({elpd}); a model that assigns ~0 probability "
            "to an observed outcome would corrupt the posterior — dropping it."
        )

    # Novelty gate: a candidate that predicts like an existing model is a
    # re-skinned duplicate under a new name — it would split posterior mass
    # with its twin in every later comparison, and being statistically tied
    # with it, pruning would never remove it. Measured on the novelty pool, not
    # the training stimuli. Uses the cached fits from the gates above and prior
    # scoring, so this adds no MCMC.
    if novelty_rmse_threshold > 0:
        pool = list(novelty_pool) if novelty_pool is not None else novelty_pool_rows()
        try:
            nearest, rmse = _min_prediction_rmse(
                model_name,
                models_dir,
                responses_path,
                pool_rows=pool,
                cache_dir=cache_dir,
                fit_kwargs=fit_kwargs,
            )
        except NoveltyPoolUnbindable as e:
            staged.unlink(missing_ok=True)
            return reject(
                f"model binds {list(e.missing)} — response-row bookkeeping a bare "
                "stimulus never carries — so it cannot be evaluated on a stimulus "
                "pool (the novelty gate now, the held-out evaluation later); "
                "compute everything from sequence_a and sequence_b."
            )
        if nearest is not None and rmse < novelty_rmse_threshold:
            staged.unlink(missing_ok=True)
            return reject(
                f"predicts like existing model {nearest!r} (p_left RMSE "
                f"{rmse:.5f} < {novelty_rmse_threshold} on the {len(pool)}-stimulus "
                f"novelty pool) — a near-duplicate of {nearest}, not a new "
                "hypothesis."
            )

    shutil.copyfile(hypothesis_file, models_dir / f"{model_name}.hypothesis.md")
    _record(
        ledger,
        name=model_name,
        outcome="admitted",
        detail="",
        hypothesis=hypothesis,
        context=ledger_context,
    )

    # Rebuild the manifest, preserving every existing entry's rationale (its
    # hypothesis) and recording this candidate's hypothesis as its rationale.
    entries: List[Dict[str, str]] = []
    seen = set()
    for entry in _manifest_entries(models_dir):
        name = entry["name"]
        if name in seen:
            continue
        seen.add(name)
        entries.append(dict(entry))
    if model_name in seen:
        for e in entries:
            if e["name"] == model_name:
                e["rationale"] = hypothesis
    else:
        entries.append({"name": model_name, "rationale": hypothesis})
    _write_manifest(models_dir, entries)
    return Admission(admitted=True)
