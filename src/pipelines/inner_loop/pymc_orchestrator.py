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
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.model_comparison.posterior import compare_table, model_posterior
from src.pipelines.inner_loop.hypothesis_ledger import (
    LEDGER_FILENAME,
    HypothesisLedger,
)
from src.pipelines.inner_loop.candidate_agent import (
    DEFAULT_CANDIDATE_HINTS,
    _build_candidate_prompt,
    _describe_standing,
    _spawn_candidate_agent,
    _write_candidate_context,
    _write_existing_hypotheses,
)
from src.pipelines.inner_loop.model_zoo import (
    AllCandidatesNoFileError,
    DEFAULT_NOVELTY_RMSE_THRESHOLD,
    DEFAULT_PRUNE_DSE_MULTIPLIER,
    _NO_FILE_DETAIL,
    _admit_candidate,
    _check_round_admissions,
    _drop_nonfinite_elpd_models,
    _drop_unfittable_models,
    _lens_index,
    _lens_offset,
    _manifest_entries,
    _manifest_names,
    _prune_losers,
    _resolve_candidate_name,
    _seed_model_set,
)

from src.pipelines.inner_loop.critique_round import (
    CRITIQUE_N_PROPOSALS,
    CRITIQUE_PPC_REPLICATES,
    CRITIQUE_SIGNIFICANCE_ALPHA,
    _run_critique_round,
)

# Occam backstop for model selection: each model's log-prior is this constant
# times its non-comment line count (see ``model_complexity``). Negative ⇒ leaner
# models are preferred when fit is comparable. It is deliberately *gentle* — a
# tie-breaker among hypotheses the data barely distinguishes, not the main guard
# against blended models (the hypothesis-first candidate generation is that). The
# proxy is imperfect: it also nicks a verbose but legitimately single-mechanism
# model (e.g. a full Bayesian model), so keep the magnitude small.
DEFAULT_COMPLEXITY_PRIOR_CONST = -0.05


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
        # Stage 1 — write every candidate's context, then spawn the agents
        # concurrently: each is a CLI subprocess whose latency dominates the
        # round, and they are independent given the shared round context.
        candidate_dirs = []
        for idx in range(candidate_count):
            candidate_dir = round_dir / f"candidate_{idx}"
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

        # Stage 2 — admit sequentially in candidate order: admission mutates
        # the manifest, uniquifies names, and runs MCMC + the novelty gate, so
        # a fixed order keeps runs deterministic (earlier candidates win ties).
        round_context = f"{ledger_context} round {iteration}".strip()
        round_results: List[Dict[str, str]] = []
        for (idx, candidate_dir, _, lens), ok in zip(candidate_dirs, spawn_ok):
            if not ok:
                round_results.append(
                    {"outcome": "spawn_failed", "detail": "agent process failed"}
                )
                continue
            admitted = _admit_candidate(
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
            if admitted:
                round_results.append({"outcome": "admitted", "detail": ""})
            else:
                detail = _NO_FILE_DETAIL
                if (candidate_dir / "candidate.py").exists():
                    detail = "rejected after file written"
                round_results.append({"outcome": "rejected", "detail": detail})
        _check_round_admissions(round_results, round_context=round_context)
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
            # Re-normalize over the surviving set (cached fits — no new MCMC).
            posterior = _score(
                responses_path, models_dir, complexity_prior_const, cache_dir, fit_kwargs
            )
        comparison = _compare(responses_path, models_dir, cache_dir, fit_kwargs)
        _record_history_step(
            history, results_dir, posterior, comparison, iteration=iteration, pruned=pruned
        )

    result = _export(results_dir, models_dir, posterior, comparison)
    result["history"] = history
    result["history_path"] = str(results_dir / "history.json")
    result["live_models"] = _manifest_names(models_dir)
    result["ledger_path"] = str(ledger.path)
    return result


def _resolve_protected_names(
    protected_names: Optional[Iterable[str]], seeded_names: set
) -> set:
    """The subset of the seeded set that pruning must never touch.

    ``None`` protects the whole seeded set. An explicit set is intersected
    with the seeded names (a project seed held out of this run is simply
    absent); an explicit non-empty set that matches nothing is a caller bug
    and raises rather than silently leaving every model prunable.
    """
    if protected_names is None:
        return set(seeded_names)
    requested = set(protected_names)
    protected = requested & set(seeded_names)
    if requested and not protected:
        raise ValueError(
            f"None of the protected model names {sorted(requested)} is in the "
            f"seeded model set {sorted(seeded_names)}."
        )
    return protected


def _best_model(posterior: Dict[str, Any]) -> str:
    """The raw softmax-posterior argmax (a report field, not the selection rule)."""
    return max(posterior["posteriors"], key=lambda m: posterior["posteriors"][m])


def _unreliable_names(comparison: Dict[str, Dict[str, Any]]) -> List[str]:
    """Models whose PSIS-LOO verdict in ``comparison`` is unreliable, sorted."""
    return sorted(
        name
        for name, row in comparison.items()
        if isinstance(row, dict) and row.get("loo_unreliable")
    )


def _best_exportable_model(
    posterior: Dict[str, Any], comparison: Dict[str, Dict[str, Any]]
) -> str:
    """The best model by ELPD-LOO rank among those whose PSIS-LOO is reliable.

    This is the loop's single notion of "best": the export, the per-step
    history and the critique incumbent all use it. Selection follows
    ``comparison[name]["rank"]`` (``az.compare``'s ordering by raw ELPD-LOO),
    restricted to reliable rows. It deliberately does NOT use the softmax
    ``posteriors``: those are rounded to six decimals, so every model more
    than ~14 nats behind the argmax reads 0.0 and ties, and a ``max`` over
    them returned whichever came first in the manifest — in the baseline
    sweeps that exported a seed hundreds of nats behind a reliable agent
    model in 62 of 230 experiments. The posterior (which also carries the
    line-count complexity prior) stays a report field.

    With no comparison table (no reliability or rank information available)
    we fall back to the plain posterior argmax. Every model in the posterior
    must have a comparison row; if NOTHING is reliable there is nothing
    trustworthy to export, and both cases raise rather than guess.
    """
    posteriors = posterior["posteriors"]
    if not comparison:
        return _best_model(posterior)
    missing = sorted(name for name in posteriors if name not in comparison)
    if missing:
        raise ValueError(
            f"Model(s) {missing} are in the posterior but have no comparison row; "
            "the posterior and the az.compare table must cover the same model set."
        )
    reliable = [
        name for name in posteriors if not comparison[name].get("loo_unreliable")
    ]
    if not reliable:
        raise RuntimeError(
            "Cannot export a model: no model has a reliable PSIS-LOO estimate "
            "(every candidate's ELPD-LOO was flagged unreliable — many high "
            "Pareto-k points). Improve the fit or data before selecting a model."
        )
    ranks = {name: int(comparison[name]["rank"]) for name in reliable}
    if len(set(ranks.values())) != len(ranks):
        raise ValueError(
            f"Reliable models share an az.compare rank: {ranks}; ranks must be unique."
        )
    best = min(reliable, key=lambda name: ranks[name])
    elpd = {name: float(comparison[name]["elpd_loo"]) for name in reliable}
    if elpd[best] < max(elpd.values()):
        raise ValueError(
            f"Lowest-rank reliable model {best!r} (ELPD {elpd[best]:.2f}) is not "
            f"the ELPD-best reliable model; the comparison table is inconsistent: "
            f"{elpd}"
        )
    return best


def _record_history_step(
    history: List[Dict[str, Any]],
    results_dir: Path,
    posterior: Dict[str, Any],
    comparison: Dict[str, Dict[str, Any]],
    iteration: Optional[int],
    pruned: Optional[List[str]] = None,
) -> None:
    """Append one scoring step to the history and persist it immediately.

    The file is rewritten after every step so a crashed run still leaves the
    trajectory up to its last completed scoring. ``best_model`` is selected by
    the same rule as the export (``_best_exportable_model``: ELPD rank among
    reliable models), so the trajectory a recovery harness scores from this
    file is the model the loop would carry forward. The raw posterior argmax
    and the models excluded as unreliable are recorded beside it for audit.
    ``pruned`` records any models dropped by the pruning pass this step (the
    posterior and comparison in the entry are over the surviving set).
    """
    entry = {
        "step": len(history),
        "iteration": iteration,
        "best_model": _best_exportable_model(posterior, comparison),
        "argmax_model": _best_model(posterior),
        "excluded_unreliable": _unreliable_names(comparison),
        "posteriors": dict(posterior["posteriors"]),
        "elpd_loo": dict(posterior["elpd_loo"]),
    }
    if pruned:
        entry["pruned"] = list(pruned)
    history.append(entry)
    (results_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


def _score(
    responses_path: Path,
    models_dir: Path,
    complexity_prior_const: float,
    cache_dir: Optional[Path],
    fit_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    return model_posterior(
        responses_path,
        models_dir,
        complexity_prior_const=complexity_prior_const,
        cache_dir=cache_dir,
        **fit_kwargs,
    )


def _compare(
    responses_path: Path,
    models_dir: Path,
    cache_dir: Optional[Path],
    fit_kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """ELPD-LOO distinguishability table (reuses cached fits — no new MCMC)."""
    return compare_table(responses_path, models_dir, cache_dir=cache_dir, **fit_kwargs)


def _export(
    results_dir: Path,
    models_dir: Path,
    posterior: Dict[str, Any],
    comparison: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    comparison = comparison or {}
    argmax_model = _best_model(posterior)
    # Models arviz flagged unreliable are excluded from selection AND from the
    # next design's EIG prior; record them so the exclusion is auditable, not
    # only visible in the prose report.
    excluded_unreliable = _unreliable_names(comparison)

    # Write the posterior + comparison record BEFORE selection: it is a
    # diagnostic record of what the run concluded, not an endorsement of the
    # selection. If _best_exportable_model refuses below (nothing reliable),
    # this file is what explains why.
    payload = {
        **posterior,
        "comparison": comparison,
        "excluded_unreliable": excluded_unreliable,
    }
    (results_dir / "model_posterior.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # Exclude models whose PSIS-LOO estimate is unreliable rather than hard-
    # failing on them; only when NOTHING is reliable does this raise.
    best_model = _best_exportable_model(posterior, comparison)
    argmax_excluded = best_model != argmax_model and comparison.get(
        argmax_model, {}
    ).get("loo_unreliable")

    # Re-record with the exported selection now that it is known: downstream
    # validation/export must key off the model we actually exported (the best
    # reliable one), not re-derive the posterior argmax (which may be an
    # excluded, unreliable model). The earlier write above is the diagnostic
    # that survives the no-reliable-model raise.
    payload["best_model"] = best_model
    (results_dir / "model_posterior.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    shutil.copyfile(models_dir / f"{best_model}.py", results_dir / "best_model.py")

    hypotheses = {
        e["name"]: (e.get("rationale") or "").strip()
        for e in _manifest_entries(models_dir)
    }
    ranked = sorted(posterior["posteriors"].items(), key=lambda kv: kv[1], reverse=True)
    lines = [
        "# Inner Model Loop Report",
        "",
        "Each model below is ONE distinct cognitive hypothesis. The posterior mass "
        "shows which single hypothesis best explains the data — it is **not** a "
        "recipe to combine the top models into a blend.",
        "",
        f"- Best model: **{best_model}** "
        f"(posterior={posterior['posteriors'][best_model]:.3f}, "
        f"elpd_loo={posterior['elpd_loo'][best_model]:.2f})",
        f"- Trials: {posterior['n_trials']}",
        f"- Models compared: {len(ranked)}",
        "",
        "## Posterior over models (ELPD-LOO)",
        "",
        "| model | posterior | elpd_loo |",
        "| --- | --- | --- |",
    ]
    if best_model != argmax_model:
        idx = next(i for i, line in enumerate(lines) if line.startswith("- Best model:"))
        if argmax_excluded:
            note = (
                f"- NOTE: the posterior argmax (**{argmax_model}**) was **excluded "
                "from selection** — its PSIS-LOO estimate is unreliable (many high "
                "Pareto-k points), so the exported model above is the best "
                "*reliable* one by ELPD-LOO rank instead."
            )
        else:
            note = (
                f"- NOTE: the posterior argmax (**{argmax_model}**) differs from the "
                "exported model: the posterior includes the line-count complexity "
                "prior, whereas selection follows the raw ELPD-LOO rank among "
                "reliable models."
            )
        lines.insert(idx + 1, note)
    lines += [
        f"| {name} | {p:.4f} | {posterior['elpd_loo'][name]:.2f} |"
        for name, p in ranked
    ]
    lines += ["", "## Hypotheses", ""]
    for name, _ in ranked:
        lines.append(f"- **{name}**: {hypotheses.get(name) or '(no stated hypothesis)'}")

    if comparison:
        # Ordered by az.compare rank (0 = best by RAW ELPD-LOO — no complexity
        # prior). elpd_diff/dse are relative to that top model; a model is
        # distinguishable only when elpd_diff is large vs dse (~elpd_diff > 2*dse).
        by_rank = sorted(comparison.items(), key=lambda kv: kv[1]["rank"])
        loo_top = by_rank[0][0] if by_rank else None
        # The exported "Best model" is this table's lowest-rank RELIABLE row, so
        # it differs from rank 0 only when higher-ranked rows were excluded as
        # PSIS-LOO-unreliable. Say so rather than let the report contradict itself.
        reconcile = ""
        if loo_top is not None and loo_top != best_model:
            reconcile = (
                f" NOTE: this table's raw-ELPD top model (`{loo_top}`) is not "
                f"the exported **Best model** (`{best_model}`) — higher-ranked "
                "models were excluded as PSIS-LOO-unreliable (see "
                "`excluded_unreliable`), so the export is the best *reliable* model."
            )
        lines += [
            "",
            "## Distinguishability (arviz.compare, PSIS-LOO)",
            "",
            "`elpd_diff` and `dse` are relative to the best model. A model is "
            "only clearly worse than the best when `elpd_diff > 2 * dse`; "
            "models within ~2·dse of the top are statistically indistinguishable. "
            "`LOO reliable` is False when PSIS-LOO flagged this model's estimate as "
            "untrustworthy (many high Pareto-k points) — its row should be read with "
            "caution." + reconcile,
            "",
            "| model | elpd_diff | dse | distinguishable from best | weight | LOO reliable |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, row in by_rank:
            if row["rank"] == 0:
                verdict = "— (best)"
            elif row["dse"] > 0 and row["elpd_diff"] > 2 * row["dse"]:
                verdict = "yes"
            else:
                verdict = "no (within ~2·dse)"
            reliable = "no ⚠" if row.get("loo_unreliable") else "yes"
            marker = " ←selected" if name == best_model else ""
            lines.append(
                f"| {name}{marker} | {row['elpd_diff']:.2f} | {row['dse']:.2f} | "
                f"{verdict} | {row['weight']:.3f} | {reliable} |"
            )

    (results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "best_model": best_model,
        "posteriors": posterior["posteriors"],
        "elpd_loo": posterior["elpd_loo"],
        "comparison": comparison,
        "excluded_unreliable": excluded_unreliable,
        "model_posterior_path": str(results_dir / "model_posterior.json"),
        "best_model_path": str(results_dir / "best_model.py"),
        "report_path": str(results_dir / "report.md"),
    }
