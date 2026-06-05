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
        best_model.py           # copy of the argmax-posterior model
        report.md
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.models.pymc_inference import load_pymc_model
from src.model_comparison.posterior import compare_table, model_posterior

_PKG_DIR = Path(__file__).resolve().parent
_THEORY_PROMPT = _PKG_DIR / "prompts" / "pymc_theory.md"


# ─────────────────────────────────────────────
# Model-set ("zoo") helpers
# ─────────────────────────────────────────────


def _manifest_names(models_dir: Path) -> List[str]:
    manifest_path = models_dir / "models_manifest.yaml"
    if not manifest_path.exists():
        return []
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    out = []
    for entry in manifest.get("models") or []:
        name = entry.get("name") if isinstance(entry, dict) else entry
        if name:
            out.append(name)
    return out


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


def _admit_candidate(
    candidate_file: Path, models_dir: Path, model_name: str, rationale: str
) -> bool:
    """Validate a candidate PyMC model and, if valid, admit it to the model set.

    Validation = it loads as a module-level ``model: pm.Model`` (via
    ``load_pymc_model``). Invalid candidates are skipped (returns False) so one
    bad agent output does not abort the round.
    """
    if not candidate_file.exists():
        return False
    staged = models_dir / f"{model_name}.py"
    shutil.copyfile(candidate_file, staged)
    try:
        load_pymc_model(model_name, models_dir)
    except Exception:
        staged.unlink(missing_ok=True)
        return False

    entries = []
    seen = set()
    for name in _manifest_names(models_dir):
        if name not in seen:
            entries.append({"name": name})
            seen.add(name)
    if model_name not in seen:
        entries.append({"name": model_name, "rationale": rationale})
    else:
        for e in entries:
            if e["name"] == model_name:
                e["rationale"] = rationale
    _write_manifest(models_dir, entries)
    return True


# ─────────────────────────────────────────────
# Agent spawning
# ─────────────────────────────────────────────


def _write_candidate_context(
    candidate_dir: Path,
    responses_path: Path,
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
        "Write `candidate.py` (a module-level PyMC model) per the instructions.",
    ]
    (candidate_dir / "CONTEXT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if current_posterior is not None:
        (candidate_dir / "model_posterior.json").write_text(
            json.dumps(current_posterior, indent=2), encoding="utf-8"
        )
    hints = [
        "Refine the current best model in a small, evidence-backed way.",
        "Try a different cognitive mechanism if the current models miss structure.",
        "Try a simpler or a higher-variance alternative if progress has stalled.",
    ]
    (candidate_dir / "CANDIDATE_BRIEF.md").write_text(
        f"# Candidate Brief\n\n{hints[candidate_idx % len(hints)]}\n", encoding="utf-8"
    )


def _spawn_candidate_agent(
    candidate_dir: Path, agent_timeout_sec: int, backend: Optional[str]
) -> bool:
    from src.runtime.coding_agent import run_coding_agent

    prompt = (
        f"{_THEORY_PROMPT.read_text(encoding='utf-8')}\n\n"
        f"---\n\nRead `CONTEXT.md` in this directory, then write `candidate.py`.\n"
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
    complexity_prior_const: float = 0.0,
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
    fit_kwargs = fit_kwargs or {}

    posterior = _score(
        responses_path, models_dir, complexity_prior_const, cache_dir, fit_kwargs
    )

    for iteration in range(max_iterations):
        round_dir = results_dir / f"iter_{iteration}"
        for idx in range(candidate_count):
            candidate_dir = round_dir / f"candidate_{idx}"
            _write_candidate_context(
                candidate_dir,
                responses_path,
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
                rationale=f"Inner-loop candidate from round {iteration}.",
            )
        posterior = _score(
            responses_path, models_dir, complexity_prior_const, cache_dir, fit_kwargs
        )

    comparison = _compare(responses_path, models_dir, cache_dir, fit_kwargs)
    return _export(results_dir, models_dir, posterior, comparison)


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

    best_model = max(posterior["posteriors"], key=lambda m: posterior["posteriors"][m])
    shutil.copyfile(models_dir / f"{best_model}.py", results_dir / "best_model.py")

    ranked = sorted(posterior["posteriors"].items(), key=lambda kv: kv[1], reverse=True)
    lines = [
        "# Inner Model Loop Report",
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
