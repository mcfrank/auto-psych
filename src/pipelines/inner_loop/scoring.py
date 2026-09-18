"""Scoring, selection, and export helpers for the PyMC inner model loop.

Functions here score the model set (ELPD-LOO posterior + az.compare table),
select the best exportable model (by ELPD rank among PSIS-LOO-reliable rows),
record the per-step history, and write the final export artifacts
(``model_posterior.json``, ``best_model.py``, ``report.md``).

They were extracted from ``pymc_orchestrator.py`` so that orchestration,
candidate management, and scoring/export live in separate modules.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.model_comparison.posterior import compare_table, model_posterior
from src.pipelines.inner_loop.model_zoo import _manifest_entries

# Occam backstop for model selection: each model's log-prior is this constant
# times its non-comment line count (see ``model_complexity``). Negative ⇒ leaner
# models are preferred when fit is comparable. It is deliberately *gentle* — a
# tie-breaker among hypotheses the data barely distinguishes, not the main guard
# against blended models (the hypothesis-first candidate generation is that). The
# proxy is imperfect: it also nicks a verbose but legitimately single-mechanism
# model (e.g. a full Bayesian model), so keep the magnitude small.
DEFAULT_COMPLEXITY_PRIOR_CONST = -0.05


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
