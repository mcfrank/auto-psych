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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.models.model_manifest import manifest_path, read_manifest_entries
from src.models.data_binding import make_stim_data
from src.models.model_loading import load_pymc_model
from src.models.pymc_inference import (
    evict_fit_cache,
    fit_model,
    model_logp_is_finite,
)
from src.model_comparison.likelihood import log_likelihood
from src.model_comparison.posterior import compare_table
from src.pipelines.inner_loop.hypothesis_ledger import (
    HypothesisLedger,
    LedgerEntry,
    one_line,
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


def _lens_offset(exp_num: int, *, max_iterations: int, candidate_count: int) -> int:
    """The lens-schedule position at which experiment ``exp_num`` starts.

    Experiment k spends ``max_iterations * candidate_count`` candidate slots, so
    experiment k+1 continues the walk through the lens battery where k stopped.
    """
    if exp_num < 1:
        raise ValueError(f"Experiment numbers start at 1; got {exp_num}.")
    return (exp_num - 1) * max_iterations * candidate_count


def _lens_index(
    lens_offset: int, iteration: int, candidate_count: int, candidate_idx: int, n_lenses: int
) -> int:
    """Which lens candidate ``candidate_idx`` of round ``iteration`` works."""
    if n_lenses < 1:
        raise ValueError("The lens battery is empty.")
    return (lens_offset + iteration * candidate_count + candidate_idx) % n_lenses


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
            detail=one_line(detail),
            hypothesis=one_line(hypothesis),
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
    whole run. Compute each model's ELPD-LOO now (reusing cached fits — no extra
    MCMC) and drop the non-finite ones with a loud warning. Fails loudly only if
    **no** model survives.
    """
    fit_kwargs = fit_kwargs or {}
    keep: List[Dict[str, str]] = []
    for entry in _manifest_entries(models_dir):
        name = entry["name"]
        try:
            elpd = log_likelihood(
                name, responses_path, models_dir, cache_dir=cache_dir, **fit_kwargs
            )
        except Exception as e:  # noqa: BLE001 — any fit/LOO failure means unscorable
            print(
                f"  [drop] model {name!r}: ELPD-LOO computation failed "
                f"({type(e).__name__}: {e}) — cannot score it; dropping.",
                flush=True,
            )
            _record(
                ledger,
                name=name,
                outcome="dropped",
                detail=f"ELPD-LOO computation failed ({type(e).__name__}: {e})",
                hypothesis=entry.get("rationale") or "",
                context=ledger_context,
            )
            continue
        if math.isfinite(elpd):
            keep.append(entry)
        else:
            print(
                f"  [drop] model {name!r}: non-finite ELPD-LOO ({elpd}) on the data "
                "— would corrupt the posterior; dropping.",
                flush=True,
            )
            _record(
                ledger,
                name=name,
                outcome="dropped",
                detail=f"non-finite ELPD-LOO ({elpd}) on this experiment's data",
                hypothesis=entry.get("rationale") or "",
                context=ledger_context,
            )
    if not keep:
        raise ValueError(
            f"No model in {models_dir} has a finite ELPD-LOO on the data — every "
            "model's PSIS-LOO was non-finite."
        )
    _write_manifest(models_dir, keep)


def _min_prediction_rmse(
    model_name: str,
    models_dir: Path,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], float]:
    """Min RMSE between ``model_name``'s p_left and each admitted model's.

    Predictions are posterior means on the observed stimuli, computed from the
    cached fits (this runs after the admission fit-gate and after scoring has
    fit every admitted model, so no new MCMC happens here). Returns the
    nearest model's name and the RMSE — ``(None, inf)`` when the set holds no
    other model.
    """
    with Path(responses_path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def posterior_mean_p_left(name: str) -> np.ndarray:
        fitted = fit_model(
            name, models_dir, responses_path, cache_dir=cache_dir, **(fit_kwargs or {})
        )
        stim_data = make_stim_data(fitted.model, rows)
        return np.asarray(fitted.predict_p_left(stim_data), dtype="float64")

    candidate_p = posterior_mean_p_left(model_name)
    nearest: Optional[str] = None
    nearest_rmse = float("inf")
    for name in _manifest_names(models_dir):
        if name == model_name:
            continue
        rmse = float(np.sqrt(np.mean((candidate_p - posterior_mean_p_left(name)) ** 2)))
        if rmse < nearest_rmse:
            nearest, nearest_rmse = name, rmse
    return nearest, nearest_rmse


# Pruning: after each scoring pass, a non-protected model is dropped when it is
# statistically distinguishable from the best (elpd_diff > multiplier·dse among
# PSIS-LOO-reliable rows). The surviving set is the uncertainty set — every
# non-protected survivor is within the margin of the best — which is what the
# outer loop carries into the next experiment. Stacking weight is deliberately
# NOT a criterion: az.compare's weights are ensemble coefficients, not
# plausibility. See the decision record for the empirical evidence.
DEFAULT_PRUNE_DSE_MULTIPLIER = 2.0

# Novelty gate: a candidate whose posterior-mean p_left is within this RMSE of
# an admitted model's (on the observed stimuli) is a re-skinned duplicate, not
# a new hypothesis — reject it at admission. See the decision record for how
# this threshold was calibrated. Set to 0 to disable.
DEFAULT_NOVELTY_RMSE_THRESHOLD = 0.02


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
    if comparison[baseline].get("loo_unreliable"):
        print(
            f"  [warn] Skipping model pruning: baseline model {baseline!r} "
            "(rank 0) has an unreliable LOO estimate, so every elpd_diff "
            "against it is untrustworthy.",
            file=sys.stderr,
            flush=True,
        )
        return []
    unreliable = sorted(
        name for name, row in comparison.items() if row.get("loo_unreliable")
    )
    if unreliable:
        print(
            "  [warn] Not pruning models with unreliable LOO estimates: "
            + ", ".join(unreliable),
            file=sys.stderr,
            flush=True,
        )
    to_prune = [
        name
        for name in names
        if name not in protected
        and name in comparison
        and not comparison[name].get("loo_unreliable")
        and comparison[name]["dse"] > 0
        and comparison[name]["elpd_diff"] > dse_multiplier * comparison[name]["dse"]
    ]
    if not to_prune:
        return []

    hypotheses = {
        e["name"]: (e.get("rationale") or "") for e in _manifest_entries(models_dir)
    }
    pruned_dir = models_dir / "pruned"
    pruned_dir.mkdir(exist_ok=True)
    for name in to_prune:
        row = comparison[name]
        for suffix in (".py", ".hypothesis.md"):
            src = models_dir / f"{name}{suffix}"
            if src.exists():
                shutil.move(str(src), str(pruned_dir / f"{name}{suffix}"))
        evict_fit_cache(name)
        margin = (
            f"{row['elpd_diff']:.1f} nats behind {baseline} "
            f"({row['elpd_diff'] / row['dse']:.1f}× dse)"
        )
        print(
            f"  [prune] {name}: elpd_diff {row['elpd_diff']:.1f} > "
            f"{dse_multiplier}·dse ({row['dse']:.1f}) — {margin}; moved to "
            "models/pruned/.",
            flush=True,
        )
        _record(
            ledger,
            name=name,
            outcome="pruned",
            detail=margin,
            hypothesis=hypotheses.get(name, ""),
            context=ledger_context,
        )
    remaining = set(names) - set(to_prune)
    _write_manifest(
        models_dir,
        [e for e in _manifest_entries(models_dir) if e["name"] in remaining],
    )
    return to_prune


def _admit_candidate(
    candidate_file: Path,
    models_dir: Path,
    model_name: str,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    novelty_rmse_threshold: float = DEFAULT_NOVELTY_RMSE_THRESHOLD,
    ledger: Optional[HypothesisLedger] = None,
    ledger_context: str = "",
) -> bool:
    """Validate a candidate and, if valid, admit it to the model set.

    Every outcome — admitted, or rejected for any of the reasons below — is
    recorded in ``ledger`` (when given) with the candidate's hypothesis, so the
    next round's briefs can list what was already tried.

    A candidate is admitted only when it ships **both**:

    - ``candidate.py`` that loads as a module-level ``model: pm.Model`` (via
      ``load_pymc_model``), evaluates to a finite logp on the data, **and
      actually completes an MCMC fit**, and
    - ``hypothesis.md`` next to it stating, in natural language, the single
      cognitive hypothesis the model implements.

    The hypothesis text becomes the model's manifest rationale and is copied to
    ``models/<name>.hypothesis.md`` so every model in the set carries the
    hypothesis it tests. Candidates missing either file, or whose logp is
    non-finite on the data, or whose MCMC sampling raises, are skipped (returns
    False, with a loud message) so one bad agent output does not abort the round.

    The finite-logp check only inspects the initial point, so a candidate can
    pass it yet NaN once NUTS jitters off that point. Such a candidate, if merely
    admitted, would crash the post-admission scoring pass and take down the whole
    run — so admission ends with a real fit (its result is cached and reused by
    scoring, adding no extra MCMC), containing any sampling failure to this one
    candidate.
    """
    hypothesis_file = candidate_file.parent / "hypothesis.md"
    hypothesis = (
        hypothesis_file.read_text(encoding="utf-8").strip()
        if hypothesis_file.exists()
        else ""
    )

    def reject(reason: str) -> bool:
        print(f"  [reject] {model_name}: {reason}", flush=True)
        _record(
            ledger,
            name=model_name,
            outcome="rejected",
            detail=reason,
            hypothesis=hypothesis,
            context=ledger_context,
        )
        return False

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
        fit_model(
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
    # with its twin in every later comparison. Uses the cached fits from the
    # gates above and prior scoring, so this adds no MCMC.
    if novelty_rmse_threshold > 0:
        nearest, rmse = _min_prediction_rmse(
            model_name,
            models_dir,
            responses_path,
            cache_dir=cache_dir,
            fit_kwargs=fit_kwargs,
        )
        if nearest is not None and rmse < novelty_rmse_threshold:
            staged.unlink(missing_ok=True)
            return reject(
                f"predicts like existing model {nearest!r} (p_left RMSE "
                f"{rmse:.4f} < {novelty_rmse_threshold}) — a near-duplicate of "
                f"{nearest}, not a new hypothesis."
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
    return True
