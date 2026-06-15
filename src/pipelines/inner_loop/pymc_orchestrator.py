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
        iter_0/candidate_0/     # per-candidate agent working dirs
        model_posterior.json    # ELPD-LOO posterior over models/
        history.json            # best model + posterior after every scoring step
        best_model.py           # copy of the argmax-posterior model
        report.md
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.models.pymc_inference import load_pymc_model, model_logp_is_finite
from src.model_comparison.posterior import compare_table, model_posterior

_PKG_DIR = Path(__file__).resolve().parent
_THEORY_PROMPT = _PKG_DIR / "prompts" / "pymc_theory.md"

# Occam backstop for model selection: each model's log-prior is this constant
# times its non-comment line count (see ``model_complexity``). Negative ⇒ leaner
# models are preferred when fit is comparable. It is deliberately *gentle* — a
# tie-breaker among hypotheses the data barely distinguishes, not the main guard
# against blended models (the hypothesis-first candidate generation is that). The
# proxy is imperfect: it also nicks a verbose but legitimately single-mechanism
# model (e.g. a full Bayesian model), so keep the magnitude small.
DEFAULT_COMPLEXITY_PRIOR_CONST = -0.05


# ─────────────────────────────────────────────
# Model-set ("zoo") helpers
# ─────────────────────────────────────────────


def _manifest_entries(models_dir: Path) -> List[Dict[str, str]]:
    """Full manifest entries (``name`` + ``rationale``), normalised to dicts."""
    manifest_path = models_dir / "models_manifest.yaml"
    if not manifest_path.exists():
        return []
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    out: List[Dict[str, str]] = []
    for entry in manifest.get("models") or []:
        if isinstance(entry, dict):
            out.append(entry)
        elif entry:
            out.append({"name": entry})
    return out


def _manifest_names(models_dir: Path) -> List[str]:
    return [e["name"] for e in _manifest_entries(models_dir) if e.get("name")]


def _write_manifest(models_dir: Path, entries: List[Dict[str, str]]) -> None:
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": entries}, sort_keys=False), encoding="utf-8"
    )


def _seed_model_set(seed_models_dir: Path, models_dir: Path) -> List[Dict[str, str]]:
    """Copy every model listed in ``seed_models_dir``'s manifest into ``models_dir``.

    Returns the manifest entries (name + rationale) that were carried over.
    Fails loudly if a listed model file is missing — we never silently drop a
    seed model.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    seed_manifest_path = seed_models_dir / "models_manifest.yaml"
    if not seed_manifest_path.exists():
        raise FileNotFoundError(
            f"seed_models_dir has no models_manifest.yaml: {seed_manifest_path}"
        )
    seed_manifest = yaml.safe_load(seed_manifest_path.read_text(encoding="utf-8")) or {}

    entries: List[Dict[str, str]] = []
    for entry in seed_manifest.get("models") or []:
        name = entry.get("name") if isinstance(entry, dict) else entry
        if not name:
            continue
        src = seed_models_dir / f"{name}.py"
        if not src.exists():
            raise FileNotFoundError(f"Seed model {name!r} has no file at {src}")
        shutil.copyfile(src, models_dir / f"{name}.py")
        rationale = entry.get("rationale", "") if isinstance(entry, dict) else ""
        entries.append({"name": name, "rationale": rationale or "Seed model."})

    if not entries:
        raise ValueError(f"No seed models found in {seed_manifest_path}")
    _write_manifest(models_dir, entries)
    return entries


def _drop_unfittable_models(models_dir: Path, responses_path: Path) -> None:
    """Remove from the manifest any seed model that cannot be MCMC-fit.

    A seed/theory model whose logp is non-finite on the data (e.g. a
    numerically unsafe construct that NaNs in PyTensor) would otherwise crash
    ``pm.sample`` at its start-value check and abort the whole run. We drop such
    models from the manifest with a loud warning rather than let one bad model
    kill a long agentic run. Fails loudly only if **no** model survives.
    """
    keep: List[Dict[str, str]] = []
    for entry in _manifest_entries(models_dir):
        name = entry.get("name")
        if not name:
            continue
        fittable, reason = model_logp_is_finite(name, models_dir, responses_path)
        if fittable:
            keep.append(entry)
        else:
            print(f"  [drop] seed model {name!r} cannot be fit — {reason}", flush=True)
    if not keep:
        raise ValueError(
            f"No fittable seed models remain in {models_dir} — every seed model's "
            "logp was non-finite on the data."
        )
    _write_manifest(models_dir, keep)


def _admit_candidate(
    candidate_file: Path, models_dir: Path, model_name: str, responses_path: Path
) -> bool:
    """Validate a candidate and, if valid, admit it to the model set.

    A candidate is admitted only when it ships **both**:

    - ``candidate.py`` that loads as a module-level ``model: pm.Model`` (via
      ``load_pymc_model``) **and** evaluates to a finite logp on the data, and
    - ``hypothesis.md`` next to it stating, in natural language, the single
      cognitive hypothesis the model implements.

    The hypothesis text becomes the model's manifest rationale and is copied to
    ``models/<name>.hypothesis.md`` so every model in the set carries the
    hypothesis it tests. Candidates missing either file, or whose logp is
    non-finite on the data (NaN/-inf, which would crash MCMC at its start-value
    check and abort the whole run), are skipped (returns False, with a loud
    message) so one bad agent output does not abort the round.
    """
    if not candidate_file.exists():
        print(f"  [reject] {model_name}: no candidate.py written", flush=True)
        return False
    hypothesis_file = candidate_file.parent / "hypothesis.md"
    hypothesis = (
        hypothesis_file.read_text(encoding="utf-8").strip()
        if hypothesis_file.exists()
        else ""
    )
    if not hypothesis:
        print(
            f"  [reject] {model_name}: no hypothesis.md — every model must state "
            "one cognitive hypothesis before it can be admitted",
            flush=True,
        )
        return False

    staged = models_dir / f"{model_name}.py"
    shutil.copyfile(candidate_file, staged)
    try:
        load_pymc_model(model_name, models_dir)
    except Exception as e:
        staged.unlink(missing_ok=True)
        print(
            f"  [reject] {model_name}: candidate.py is not a loadable PyMC model: {e}",
            flush=True,
        )
        return False

    fittable, reason = model_logp_is_finite(model_name, models_dir, responses_path)
    if not fittable:
        staged.unlink(missing_ok=True)
        print(
            f"  [reject] {model_name}: model cannot be fit — {reason}",
            flush=True,
        )
        return False

    shutil.copyfile(hypothesis_file, models_dir / f"{model_name}.hypothesis.md")

    # Rebuild the manifest, preserving every existing entry's rationale (its
    # hypothesis) and recording this candidate's hypothesis as its rationale.
    entries: List[Dict[str, str]] = []
    seen = set()
    for entry in _manifest_entries(models_dir):
        name = entry.get("name")
        if not name or name in seen:
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


# ─────────────────────────────────────────────
# Agent spawning
# ─────────────────────────────────────────────


def _write_existing_hypotheses(
    candidate_dir: Path,
    models_dir: Path,
    current_posterior: Optional[Dict[str, Any]],
) -> None:
    """Write the hypotheses already in the model set + how well each fits.

    Each model's hypothesis is its manifest rationale; its fit is its current
    ELPD-LOO posterior mass. The candidate agent reads this to pick a *distinct*
    or *refined* hypothesis — never to merge the top models into a blend.
    """
    posteriors = (current_posterior or {}).get("posteriors", {})
    elpd = (current_posterior or {}).get("elpd_loo", {})
    blocks: List[str] = []
    for entry in _manifest_entries(models_dir):
        name = entry.get("name")
        if not name:
            continue
        hypothesis = (entry.get("rationale") or "").strip() or "(no stated hypothesis)"
        header = f"## {name}"
        if name in posteriors:
            header += f"  — posterior {posteriors[name]:.3f}"
        if name in elpd:
            header += f", ELPD-LOO {elpd[name]:.2f}"
        blocks.append(f"{header}\n\n{hypothesis}\n")
    body = "\n".join(blocks) if blocks else "(no models yet)\n"
    text = (
        "# Existing hypotheses\n\n"
        "Each model below is ONE cognitive hypothesis, with how well it currently "
        "explains the data. Propose a hypothesis that is genuinely different from "
        "these, or a refinement of a single one of them — never a combination of "
        "several.\n\n" + body
    )
    (candidate_dir / "existing_hypotheses.md").write_text(text, encoding="utf-8")


def _write_candidate_context(
    candidate_dir: Path,
    responses_path: Path,
    models_dir: Path,
    iteration: int,
    candidate_idx: int,
    candidate_count: int,
    current_posterior: Optional[Dict[str, Any]],
) -> None:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    with responses_path.open(encoding="utf-8") as f:
        header = f.readline().strip()
    lines = [
        f"# Inner Loop — round {iteration}, candidate {candidate_idx} of {candidate_count}",
        "",
        f"Responses CSV: `{responses_path}`",
        f"Feature columns available (use these as `pm.Data` names): `{header}`",
        "",
        "Work in two steps:",
        "1. Write `hypothesis.md` — one cognitive hypothesis, in plain English.",
        "2. Write `candidate.py` — a module-level PyMC model implementing only that",
        "   hypothesis.",
        "",
        "`existing_hypotheses.md` lists the hypotheses already in the model set and",
        "how well each fits. Read it so you propose a *distinct* or *refined*",
        "hypothesis — never a blend of several.",
    ]
    (candidate_dir / "CONTEXT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_existing_hypotheses(candidate_dir, models_dir, current_posterior)

    hints = [
        "Refine one existing hypothesis within its single mechanism — e.g. a "
        "different functional form, prior, or normalization. Do NOT graft cues "
        "from other models onto it.",
        "Propose a new single-mechanism hypothesis that the current models cannot "
        "express.",
        "Propose a simpler or higher-variance single-mechanism hypothesis if "
        "progress has stalled.",
    ]
    brief = (
        "# Candidate Brief\n\n"
        f"{hints[candidate_idx % len(hints)]}\n\n"
        "Your candidate must express **exactly one** cognitive hypothesis. Do not "
        "average, weight, or mix cues or mechanisms from several hypotheses into a "
        "single model — a blended mega-model is not a hypothesis.\n"
    )
    (candidate_dir / "CANDIDATE_BRIEF.md").write_text(brief, encoding="utf-8")


def _spawn_candidate_agent(
    candidate_dir: Path, agent_timeout_sec: int, backend: Optional[str]
) -> bool:
    from src.runtime.coding_agent import run_coding_agent

    prompt = (
        f"{_THEORY_PROMPT.read_text(encoding='utf-8')}\n\n"
        "---\n\nRead `CONTEXT.md`, `CANDIDATE_BRIEF.md`, and `existing_hypotheses.md` "
        "in this directory. Then write `hypothesis.md` (your single hypothesis in "
        "plain English), and `candidate.py` (a PyMC model implementing only it).\n"
    )
    log_path = candidate_dir / "agent.jsonl"
    success, _ = run_coding_agent(
        prompt,
        cwd=candidate_dir,
        log_path=log_path,
        allowed_dirs=[candidate_dir],
        timeout_secs=agent_timeout_sec,
        backend=backend,
    )
    return success


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
    fit_kwargs: Optional[Dict[str, Any]] = None,
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
        previous experiment's `cognitive_models/`.
    max_iterations
        Number of candidate-generation rounds. ``0`` only fits/compares the seed
        set (no agent is spawned).
    candidate_count
        Candidate models proposed per round.
    complexity_prior_const
        Passed through to ``model_posterior`` (negative penalises complex models).
    fit_kwargs
        Extra kwargs for MCMC (e.g. ``{"draws": 500, "tune": 500, "chains": 2}``).

    Returns a dict with ``best_model``, ``posteriors``, ``elpd_loo`` and paths.
    """
    responses_path = Path(responses_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = results_dir / "models"

    _seed_model_set(Path(seed_models_dir), models_dir)
    _drop_unfittable_models(models_dir, responses_path)
    fit_kwargs = fit_kwargs or {}

    posterior = _score(
        responses_path, models_dir, complexity_prior_const, cache_dir, fit_kwargs
    )
    history: List[Dict[str, Any]] = []
    _record_history_step(history, results_dir, posterior, iteration=None)

    for iteration in range(max_iterations):
        round_dir = results_dir / f"iter_{iteration}"
        for idx in range(candidate_count):
            candidate_dir = round_dir / f"candidate_{idx}"
            _write_candidate_context(
                candidate_dir,
                responses_path,
                models_dir,
                iteration,
                idx,
                candidate_count,
                posterior,
            )
            if not _spawn_candidate_agent(candidate_dir, agent_timeout_sec, backend):
                continue
            _admit_candidate(
                candidate_dir / "candidate.py",
                models_dir,
                model_name=f"iter{iteration}_candidate{idx}",
                responses_path=responses_path,
            )
        posterior = _score(
            responses_path, models_dir, complexity_prior_const, cache_dir, fit_kwargs
        )
        _record_history_step(history, results_dir, posterior, iteration=iteration)

    comparison = _compare(responses_path, models_dir, cache_dir, fit_kwargs)
    result = _export(results_dir, models_dir, posterior, comparison)
    result["history"] = history
    result["history_path"] = str(results_dir / "history.json")
    return result


def _best_model(posterior: Dict[str, Any]) -> str:
    return max(posterior["posteriors"], key=lambda m: posterior["posteriors"][m])


def _record_history_step(
    history: List[Dict[str, Any]],
    results_dir: Path,
    posterior: Dict[str, Any],
    iteration: Optional[int],
) -> None:
    """Append one scoring step to the history and persist it immediately.

    The file is rewritten after every step so a crashed run still leaves the
    trajectory up to its last completed scoring.
    """
    history.append(
        {
            "step": len(history),
            "iteration": iteration,
            "best_model": _best_model(posterior),
            "posteriors": dict(posterior["posteriors"]),
            "elpd_loo": dict(posterior["elpd_loo"]),
        }
    )
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
    payload = {**posterior, "comparison": comparison}
    (results_dir / "model_posterior.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    best_model = _best_model(posterior)
    shutil.copyfile(models_dir / f"{best_model}.py", results_dir / "best_model.py")

    hypotheses = {
        e["name"]: (e.get("rationale") or "").strip()
        for e in _manifest_entries(models_dir)
        if e.get("name")
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
    lines += [
        f"| {name} | {p:.4f} | {posterior['elpd_loo'][name]:.2f} |"
        for name, p in ranked
    ]
    lines += ["", "## Hypotheses", ""]
    for name, _ in ranked:
        lines.append(f"- **{name}**: {hypotheses.get(name) or '(no stated hypothesis)'}")

    if comparison:
        # Ordered by az.compare rank (0 = best). elpd_diff/dse are relative to
        # the top model; a model is distinguishable from the best only when
        # elpd_diff is large vs dse (rule of thumb: elpd_diff > 2*dse).
        by_rank = sorted(comparison.items(), key=lambda kv: kv[1]["rank"])
        lines += [
            "",
            "## Distinguishability (arviz.compare, PSIS-LOO)",
            "",
            "`elpd_diff` and `dse` are relative to the best model. A model is "
            "only clearly worse than the best when `elpd_diff > 2 * dse`; "
            "models within ~2·dse of the top are statistically indistinguishable.",
            "",
            "| model | elpd_diff | dse | distinguishable from best | weight |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, row in by_rank:
            if row["rank"] == 0:
                verdict = "— (best)"
            elif row["dse"] > 0 and row["elpd_diff"] > 2 * row["dse"]:
                verdict = "yes"
            else:
                verdict = "no (within ~2·dse)"
            lines.append(
                f"| {name} | {row['elpd_diff']:.2f} | {row['dse']:.2f} | "
                f"{verdict} | {row['weight']:.3f} |"
            )

    (results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "best_model": best_model,
        "posteriors": posterior["posteriors"],
        "elpd_loo": posterior["elpd_loo"],
        "comparison": comparison,
        "model_posterior_path": str(results_dir / "model_posterior.json"),
        "best_model_path": str(results_dir / "best_model.py"),
        "report_path": str(results_dir / "report.md"),
    }
