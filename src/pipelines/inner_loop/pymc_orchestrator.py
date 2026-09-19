"""PyMC-native inner model loop.

The inner loop improves a cognitive model by repeatedly asking a coding agent
to write a new candidate **PyMC model** (`candidate.py` with a module-level
`with pm.Model() as model:` block), fitting every surviving model to the
observed responses via MCMC, and scoring them by ELPD-LOO. The Bayesian
posterior over models (softmax of ELPD-LOO, optionally with a complexity prior)
selects the incumbent; the best model is exported for the outer loop.

This replaces the earlier maximum-likelihood + BIC tournament: models are now
probabilistic programs fit by inference, not callables fit by external
optimization.

Layout under ``results_dir``::

    results_dir/
        models/                 # the surviving model set (the "zoo")
            <name>.py           # one PyMC model per surviving candidate
            models_manifest.yaml
            pruned/             # models that lost (audit trail, still readable)
        attempted_hypotheses.jsonl  # ledger: every candidate/prune event
        iter_0/candidate_0/     # per-candidate agent working dirs
        model_posterior.json    # ELPD-LOO posterior over models/
        history.json            # best model + posterior after every scoring step
        best_model.py           # copy of the exported (best reliable) model
        report.md
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.pipelines.inner_loop.hypothesis_ledger import (
    LEDGER_FILENAME,
    HypothesisLedger,
)
from src.pipelines.inner_loop.candidate_agent import (
    DEFAULT_CANDIDATE_HINTS,
    _spawn_candidate_agent,
    _write_candidate_context,
)
from src.pipelines.inner_loop.model_zoo import (
    DEFAULT_NOVELTY_RMSE_THRESHOLD,
    DEFAULT_PRUNE_DSE_MULTIPLIER,
    MAX_EMPTY_ROUND_RETRIES,
    AllCandidatesNoFileError,
    _NO_FILE_DETAIL,
    _admit_candidate,
    _drop_nonfinite_elpd_models,
    _drop_unfittable_models,
    _is_all_no_file_round,
    _lens_index,
    _manifest_names,
    _prune_losers,
    _record,
    _resolve_candidate_name,
    _seed_model_set,
)

from src.pipelines.inner_loop.critique_round import (
    CRITIQUE_N_PROPOSALS,
    CRITIQUE_PPC_REPLICATES,
    CRITIQUE_SIGNIFICANCE_ALPHA,
    _run_critique_round,
)
from src.pipelines.inner_loop.scoring import (
    DEFAULT_COMPLEXITY_PRIOR_CONST,
    _compare,
    _export,
    _record_history_step,
    _resolve_protected_names,
    _score,
)


# ─────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────


def run_pymc_inner_loop(
    responses_path: Path,
    results_dir: Path,
    *,
    seed_models_dir: Path,
    max_iterations: int = 3,
    candidate_count: int = 3,
    complexity_prior_const: float = DEFAULT_COMPLEXITY_PRIOR_CONST,
    cache_dir: Optional[Path] = None,
    agent_timeout_sec: int = 900,
    backend: Optional[str] = None,
    agent_model: Optional[str] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    enable_critique: bool = True,
    n_critique_proposals: int = CRITIQUE_N_PROPOSALS,
    critique_significance_alpha: float = CRITIQUE_SIGNIFICANCE_ALPHA,
    n_critique_replicates: int = CRITIQUE_PPC_REPLICATES,
    candidate_hints: Optional[List[str]] = None,
    novelty_rmse_threshold: float = DEFAULT_NOVELTY_RMSE_THRESHOLD,
    prune_dse_multiplier: float = DEFAULT_PRUNE_DSE_MULTIPLIER,
    candidate_parallelism: Optional[int] = None,
    protected_names: Optional[Iterable[str]] = None,
    ledger_context: str = "",
    lens_offset: int = 0,
    agent_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the PyMC inner model loop and export the best model.

    Parameters
    ----------
    responses_path
        Preprocessed responses CSV. Its columns are the `pm.Data` inputs that
        candidate models read (one container per column they use).
    results_dir
        Output directory (created if absent).
    seed_models_dir
        Directory with the starting model set (`<name>.py` + manifest), e.g. the
        previous experiment's `cognitive_models/`. An ``attempted_hypotheses.jsonl``
        beside the manifest (the ledger a previous experiment carried) seeds
        this run's ledger.
    protected_names
        Models that are never pruned — the project's seed models, the
        baselines a run reports against. ``None`` protects every model in
        ``seed_models_dir`` (the right default when that directory *is* the
        seed set). The outer loop passes the project seeds explicitly so that a
        model carried from a previous experiment can lose and leave the set.
        Names not in the seed set are ignored (a seed held out of this run);
        a non-empty set with no member in the seed set raises.
    ledger_context
        Prefix for the ledger's ``context`` field (e.g. ``"experiment2"``).
    max_iterations
        Number of candidate-generation rounds. ``0`` only fits/compares the seed
        set (no agent is spawned).
    candidate_count
        Candidate models proposed per round.
    complexity_prior_const
        Passed through to ``model_posterior`` (negative penalises complex models).
    fit_kwargs
        Extra kwargs for MCMC (e.g. ``{"draws": 500, "tune": 500, "chains": 2}``).
    enable_critique
        When True, run a CriticAL posterior-predictive critique of the incumbent
        (best) model before each candidate round and feed the resulting
        ``critiques.md`` to the candidate agents (see ``src/critique/ppc.py``).
    n_critique_proposals, critique_significance_alpha, n_critique_replicates
        Test statistics the critique agent proposes per round, the raw p-value
        threshold for a significant discrepancy, and the posterior-predictive replicates
        forming each statistic's null distribution.
    candidate_hints
        Exploration lenses cycled across a round's candidates (``None`` ⇒
        ``DEFAULT_CANDIDATE_HINTS``). With ``candidate_count <= len(hints)``
        every candidate in a round works a distinct lens.
    novelty_rmse_threshold
        Reject a candidate whose posterior-mean ``p_left`` is within this RMSE
        of an admitted model's on the observed stimuli (``0`` disables).
    prune_dse_multiplier
        After each scoring pass, drop non-protected models that are
        statistically distinguishable from the best
        (``elpd_diff > multiplier·dse``); ``0`` disables pruning.
    candidate_parallelism
        Concurrent candidate agents per round (``None`` ⇒ all of the round's
        candidates at once; ``1`` ⇒ sequential). Agents are CLI subprocesses,
        so this is a pure wall-clock lever; admission is always sequential in
        candidate order, keeping runs deterministic.
    lens_offset
        Starting position in the lens battery. The outer loop
        passes ``_lens_offset(exp_num, ...)`` so experiment k+1 continues the
        walk where experiment k stopped.

    Returns a dict with ``best_model``, ``posteriors``, ``elpd_loo``,
    ``live_models`` (the surviving zoo, in manifest order) and paths.
    """
    responses_path = Path(responses_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_dir / "models"

    seeded_entries = _seed_model_set(Path(seed_models_dir), models_dir)
    seeded_names = {e.get("name") for e in seeded_entries if e.get("name")}
    protected = _resolve_protected_names(protected_names, seeded_names)
    # The loop's memory: every hypothesis tried, continuing the ledger the
    # previous experiment carried beside its model set.
    ledger = HypothesisLedger.create(
        results_dir / LEDGER_FILENAME,
        inherit_from=Path(seed_models_dir) / LEDGER_FILENAME,
    )
    _drop_unfittable_models(
        models_dir, responses_path, ledger=ledger, ledger_context=ledger_context
    )
    fit_kwargs = fit_kwargs or {}
    # A carried-forward model can score a finite ELPD on a prior experiment's data
    # yet NaN on this one's; drop those before scoring so a single one can't crash
    # model_posterior and abort the whole run.
    _drop_nonfinite_elpd_models(
        models_dir,
        responses_path,
        cache_dir=cache_dir,
        fit_kwargs=fit_kwargs,
        ledger=ledger,
        ledger_context=ledger_context,
    )
    n_lenses = len(candidate_hints) if candidate_hints is not None else len(DEFAULT_CANDIDATE_HINTS)
    if max_iterations > 0 and n_lenses < 1:
        raise ValueError(
            "The lens battery is empty — pass at least one exploration lens "
            "via candidate_hints, or use the defaults."
        )

    posterior = _score(
        responses_path, models_dir, complexity_prior_const, cache_dir, fit_kwargs
    )
    # The comparison table (ELPD rank + PSIS-LOO reliability) is what selects
    # the best model at every step; it reuses the fits _score just made.
    comparison = _compare(responses_path, models_dir, cache_dir, fit_kwargs)
    history: List[Dict[str, Any]] = []
    _record_history_step(history, results_dir, posterior, comparison, iteration=None)

    rounds_abandoned = 0
    for iteration in range(max_iterations):
        round_dir = results_dir / f"iter_{iteration}"
        critique_path: Optional[Path] = None
        if enable_critique:
            critique_path = _run_critique_round(
                round_dir,
                responses_path=responses_path,
                models_dir=models_dir,
                posterior=posterior,
                comparison=comparison,
                cache_dir=cache_dir,
                fit_kwargs=fit_kwargs,
                n_proposals=n_critique_proposals,
                significance_alpha=critique_significance_alpha,
                n_replicates=n_critique_replicates,
                agent_timeout_sec=agent_timeout_sec,
                backend=backend,
                agent_model=agent_model,
                agent_root=agent_root,
            )

        round_context = f"{ledger_context} round {iteration}".strip()
        round_results: List[Dict[str, str]] = []

        for attempt in range(1 + MAX_EMPTY_ROUND_RETRIES):
            if attempt == 0:
                attempt_dir = round_dir
            else:
                attempt_dir = results_dir / f"iter_{iteration}_retry_{attempt}"
                print(
                    f"  [retry] {round_context}: all candidates no-file on "
                    f"attempt {attempt}/{1 + MAX_EMPTY_ROUND_RETRIES}, retrying",
                    flush=True,
                )

            candidate_dirs = []
            for idx in range(candidate_count):
                candidate_dir = attempt_dir / f"candidate_{idx}"
                lens = _lens_index(
                    lens_offset, iteration, candidate_count, idx, n_lenses
                )
                docs = _write_candidate_context(
                    candidate_dir,
                    responses_path,
                    models_dir,
                    iteration,
                    idx,
                    candidate_count,
                    posterior,
                    critique_path=critique_path,
                    hints=candidate_hints,
                    ledger=ledger,
                    comparison=comparison,
                    lens_index=lens,
                )
                candidate_dirs.append((idx, candidate_dir, docs, lens))

            def spawn(item) -> bool:
                _, candidate_dir, docs, _lens = item
                return _spawn_candidate_agent(
                    candidate_dir,
                    docs,
                    models_dir=models_dir,
                    responses_path=responses_path,
                    agent_timeout_sec=agent_timeout_sec,
                    backend=backend,
                    agent_model=agent_model,
                    agent_root=agent_root,
                )

            workers = min(candidate_parallelism or candidate_count, candidate_count)
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    spawn_ok = list(pool.map(spawn, candidate_dirs))
            else:
                spawn_ok = [spawn(item) for item in candidate_dirs]

            round_results = []
            for (idx, candidate_dir, _, lens), ok in zip(candidate_dirs, spawn_ok):
                if not ok:
                    round_results.append(
                        {"outcome": "spawn_failed", "detail": "agent process failed"}
                    )
                    continue
                admitted_ok = _admit_candidate(
                    candidate_dir / "candidate.py",
                    models_dir,
                    model_name=_resolve_candidate_name(
                        candidate_dir,
                        models_dir,
                        fallback=f"iter{iteration}_candidate{idx}",
                    ),
                    responses_path=responses_path,
                    cache_dir=cache_dir,
                    fit_kwargs=fit_kwargs,
                    novelty_rmse_threshold=novelty_rmse_threshold,
                    ledger=ledger,
                    ledger_context=f"{round_context} candidate {idx} lens {lens}",
                )
                if admitted_ok:
                    round_results.append({"outcome": "admitted", "detail": ""})
                else:
                    detail = _NO_FILE_DETAIL
                    if (candidate_dir / "candidate.py").exists():
                        detail = "rejected after file written"
                    round_results.append({"outcome": "rejected", "detail": detail})

            if not _is_all_no_file_round(round_results):
                break

        if _is_all_no_file_round(round_results):
            rounds_abandoned += 1
            _record(
                ledger,
                name="__round__",
                outcome="round_abandoned",
                detail=(
                    f"abandoned after {1 + MAX_EMPTY_ROUND_RETRIES} attempts "
                    f"({len(round_results)} slots, all no-file)"
                ),
                hypothesis="",
                context=round_context,
            )
            print(
                f"  [abandon] {round_context}: abandoned after "
                f"{1 + MAX_EMPTY_ROUND_RETRIES} attempts — continuing to the "
                f"next round.",
                flush=True,
            )
            continue

        posterior = _score(
            responses_path, models_dir, complexity_prior_const, cache_dir, fit_kwargs
        )
        pruned = _prune_losers(
            models_dir,
            responses_path,
            protected=protected,
            cache_dir=cache_dir,
            fit_kwargs=fit_kwargs,
            dse_multiplier=prune_dse_multiplier,
            ledger=ledger,
            ledger_context=round_context,
        )
        if pruned:
            posterior = _score(
                responses_path, models_dir, complexity_prior_const, cache_dir, fit_kwargs
            )
        comparison = _compare(responses_path, models_dir, cache_dir, fit_kwargs)
        _record_history_step(
            history, results_dir, posterior, comparison, iteration=iteration, pruned=pruned
        )

    if rounds_abandoned == max_iterations and max_iterations > 0:
        raise AllCandidatesNoFileError(
            f"Every round of the experiment ended with all candidates failing "
            f"to write files ({rounds_abandoned} of {max_iterations} rounds "
            f"abandoned after {1 + MAX_EMPTY_ROUND_RETRIES} attempts each). "
            f"This is a systemic failure, not a transient hiccup."
        )

    result = _export(results_dir, models_dir, posterior, comparison)
    result["history"] = history
    result["history_path"] = str(results_dir / "history.json")
    result["live_models"] = _manifest_names(models_dir)
    result["ledger_path"] = str(ledger.path)
    return result
