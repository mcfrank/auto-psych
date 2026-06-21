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

from src.models.pymc_inference import fit_model, load_pymc_model, model_logp_is_finite
from src.model_comparison.posterior import compare_table, model_posterior
from src.runtime.config import REPO_ROOT

_PKG_DIR = Path(__file__).resolve().parent
_THEORY_PROMPT = _PKG_DIR / "prompts" / "pymc_theory.md"
_CRITIQUE_PROMPT = _PKG_DIR / "prompts" / "critique.md"

# CriticAL critique defaults. Before each candidate round the inner loop runs a
# posterior-predictive critique of the incumbent (best) model: the critique agent
# proposes test statistics, the PPC harness scores each as a two-sided empirical
# p-value over ``CRITIQUE_PPC_REPLICATES`` posterior-predictive datasets, and
# flags the significant discrepancies (raw p ≤ alpha, no multiple-comparisons
# correction) the next round of candidates should address (see ``src/critique/ppc.py``).
CRITIQUE_N_PROPOSALS = 8
# A critique statistic is a significant discrepancy when its raw two-sided
# p-value is ≤ this (no multiple-comparisons correction). Override per-run with
# --critique-alpha.
CRITIQUE_SIGNIFICANCE_ALPHA = 0.05
CRITIQUE_PPC_REPLICATES = 200

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
    candidate_file: Path,
    models_dir: Path,
    model_name: str,
    responses_path: Path,
    *,
    cache_dir: Optional[Path] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> bool:
    """Validate a candidate and, if valid, admit it to the model set.

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
        print(
            f"  [reject] {model_name}: MCMC sampling failed "
            f"({type(e).__name__}: {e}); dropping it so it cannot abort scoring.",
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
    critique_path: Optional[Path] = None,
) -> None:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    with responses_path.open(encoding="utf-8") as f:
        header = f.readline().strip()
    columns = [c for c in header.split(",") if c]
    raw_sequence_cols = [c for c in ("sequence_a", "sequence_b") if c in columns]
    lines = [
        f"# Inner Loop — round {iteration}, candidate {candidate_idx} of {candidate_count}",
        "",
        f"Responses CSV: `{responses_path}`",
        f"Columns in the responses CSV: `{header}`",
        "",
        "Read the columns you need as `pm.Data` containers, matching each "
        "container name to a column. **Only numeric columns can back a `pm.Data`** "
        "— the precomputed integer/float feature columns and `chose_left`.",
    ]
    if raw_sequence_cols:
        lines += [
            "",
            f"The raw H/T sequence strings `{'` and `'.join(raw_sequence_cols)}` are "
            "**not numeric** and cannot be a `pm.Data` directly. To make your "
            "hypothesis depend on an aspect of the sequence the precomputed features "
            "discard — order, position, recency, or specific sub-sequences — define a "
            "module-level `compute_features(sequence_a, sequence_b) -> dict[str, "
            "float]` in `candidate.py`. The pipeline runs it on the raw sequences for "
            "every trial and exposes each returned key as a column you read with a "
            "matching `pm.Data`. This extends the feature space beyond the "
            "precomputed columns above.",
        ]
    lines += [
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
    if critique_path is not None:
        lines += [
            "",
            f"`critiques.md` ({critique_path}) is a posterior-predictive critique of",
            "the current **best** model: the test statistics on which it significantly",
            "fails to reproduce the data, each with the direction of the discrepancy",
            "and a raw p plus an FDR-adjusted q. These are *exploratory* screens, not",
            "confirmatory tests — several are checked per round, so prefer a",
            "discrepancy that survives the FDR (`q ≤ alpha`). Use the strongest such",
            "discrepancy to motivate a single mechanism that would close that gap.",
        ]
    (candidate_dir / "CONTEXT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_existing_hypotheses(candidate_dir, models_dir, current_posterior)
    if critique_path is not None and critique_path.exists():
        shutil.copyfile(critique_path, candidate_dir / "critiques.md")

    hints = [
        "Refine one existing hypothesis within its single mechanism — e.g. a "
        "different functional form, prior, or normalization. Do NOT graft cues "
        "from other models onto it.",
        "Propose a new single-mechanism hypothesis that the current models cannot "
        "express.",
        "Propose a simpler or higher-variance single-mechanism hypothesis if "
        "progress has stalled.",
    ]
    critique_note = (
        "\nIf `critiques.md` is present, prioritise a hypothesis that addresses one of "
        "the significant discrepancies it reports.\n"
        if critique_path is not None
        else ""
    )
    brief = (
        "# Candidate Brief\n\n"
        f"{hints[candidate_idx % len(hints)]}\n\n"
        "Your candidate must express **exactly one** cognitive hypothesis. Do not "
        "average, weight, or mix cues or mechanisms from several hypotheses into a "
        "single model — a blended mega-model is not a hypothesis.\n"
        f"{critique_note}"
    )
    (candidate_dir / "CANDIDATE_BRIEF.md").write_text(brief, encoding="utf-8")


def _spawn_candidate_agent(
    candidate_dir: Path,
    *,
    models_dir: Path,
    responses_path: Path,
    agent_timeout_sec: int,
    backend: Optional[str],
) -> bool:
    from src.runtime.coding_agent import run_coding_agent

    # Run from REPO_ROOT, NOT candidate_dir. opencode discovers its
    # external_directory grants by walking up from cwd to the worktree's
    # opencode.json; candidate_dir lives on $SCRATCH outside the worktree, so a
    # cwd there loads no grants and opencode auto-rejects every external path the
    # CONTEXT.md points at (critiques.md, responses CSV, model set) — no
    # candidate.py ever gets written. The critique agent runs from REPO_ROOT for
    # the same reason. The candidate_dir is named explicitly below since it is no
    # longer the cwd.
    prompt = (
        f"{_THEORY_PROMPT.read_text(encoding='utf-8')}\n\n"
        f"---\n\nYour working directory for this candidate is `{candidate_dir}`.\n"
        f"Read `CONTEXT.md`, `CANDIDATE_BRIEF.md`, `existing_hypotheses.md`, and "
        f"`critiques.md` (if present) in that directory. Then write `hypothesis.md` "
        f"(your single hypothesis in plain English) and `candidate.py` (a PyMC model "
        f"implementing only it) into that same directory.\n"
    )
    log_path = candidate_dir / "agent.jsonl"
    # For the Claude backend (which honours allowed_dirs via --add-dir) grant the
    # candidate's workspace plus the responses CSV and model set it references.
    success, _ = run_coding_agent(
        prompt,
        cwd=REPO_ROOT,
        log_path=log_path,
        allowed_dirs=[candidate_dir, models_dir, responses_path.parent],
        timeout_secs=agent_timeout_sec,
        backend=backend,
    )
    return success


# ─────────────────────────────────────────────
# Critique round (CriticAL posterior-predictive model criticism)
# ─────────────────────────────────────────────


def _incumbent_hypothesis(models_dir: Path, incumbent: str) -> str:
    """The incumbent's stated hypothesis (its manifest rationale), or a fallback."""
    for entry in _manifest_entries(models_dir):
        if entry.get("name") == incumbent:
            return (entry.get("rationale") or "").strip() or "(no stated hypothesis)"
    return "(no stated hypothesis)"


def _seed_critique_fit_cache(
    incumbent: str,
    models_dir: Path,
    responses_path: Path,
    fit_cache_dir: Path,
    fit_kwargs: Dict[str, Any],
) -> None:
    """Persist the incumbent's fit so the critique agent's PPC harness reuses it.

    The inner loop just fit the incumbent while scoring, so this is an in-process
    cache hit; we only write its InferenceData to ``fit_cache_dir`` (under the
    content-addressed name the harness expects) so the agent's separate harness
    process loads it instead of refitting with MCMC.
    """
    from src.models.pymc_inference import fit_models_cached

    fit_cache_dir.mkdir(parents=True, exist_ok=True)
    fitted = fit_models_cached(
        [incumbent],
        models_dir=models_dir,
        responses_path=responses_path,
        cache_dir=fit_cache_dir,
        **fit_kwargs,
    )[incumbent]
    nc_path = fit_cache_dir / f"{incumbent}.{fitted.fingerprint}.nc"
    if not nc_path.exists():
        fitted.idata.to_netcdf(str(nc_path))


def _write_critique_context(
    critique_dir: Path,
    incumbent: str,
    models_dir: Path,
    responses_path: Path,
    fit_cache_dir: Path,
    *,
    n_proposals: int,
    significance_alpha: float,
    n_replicates: int,
) -> None:
    """Write CRITIQUE_CONTEXT.md: the incumbent, the data schema, and the PPC command."""
    critique_dir.mkdir(parents=True, exist_ok=True)
    with responses_path.open(encoding="utf-8") as f:
        header = f.readline().strip()
    incumbent_file = models_dir / f"{incumbent}.py"
    test_stats_dir = critique_dir / "test_stats"
    results_path = critique_dir / "ppc_results.json"
    ppc_command = (
        "python3 -m src.critique.ppc \\\n"
        f"    --responses {responses_path} \\\n"
        f"    --model {incumbent} \\\n"
        f"    --models-dir {models_dir} \\\n"
        f"    --test-stats-dir {test_stats_dir} \\\n"
        f"    --out {results_path} \\\n"
        f"    --cache-dir {fit_cache_dir} \\\n"
        f"    --n-replicates {n_replicates} \\\n"
        f"    --significance-alpha {significance_alpha}"
    )
    lines = [
        "# Critique context",
        "",
        f"**Incumbent (best) model:** `{incumbent}`",
        f"**Incumbent model code:** `{incumbent_file}`",
        f"**Incumbent hypothesis:** {_incumbent_hypothesis(models_dir, incumbent)}",
        "",
        f"**Responses CSV:** `{responses_path}`",
        f"**Columns (DataFrame your test statistics receive):** `{header}`",
        f"**Model set directory:** `{models_dir}`",
        "",
        f"Propose **{n_proposals}** test statistics. Write each to "
        f"`{test_stats_dir}/<name>.py` as a function `test_statistic(df)` returning a "
        "scalar, with `# name:` and `# description:` header comments.",
        "",
        "You do **not** need to run anything. After you write the statistics, the "
        "pipeline runs the posterior-predictive harness automatically over "
        f"`{test_stats_dir}` and records the results:",
        "",
        "```bash",
        ppc_command,
        "```",
        "",
        f"That writes `{results_path}` with a two-sided empirical p-value per "
        f"statistic ({n_replicates} posterior-predictive replicates). A statistic "
        f"is a **significant discrepancy** when its `p_value` ≤ {significance_alpha} "
        "(raw, no multiple-comparisons correction).",
    ]
    (critique_dir / "CRITIQUE_CONTEXT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _spawn_critique_agent(
    critique_dir: Path,
    incumbent: str,
    *,
    models_dir: Path,
    responses_path: Path,
    cache_dir: Optional[Path],
    fit_kwargs: Dict[str, Any],
    n_proposals: int,
    significance_alpha: float,
    n_replicates: int,
    agent_timeout_sec: int,
    backend: Optional[str],
) -> bool:
    """Critique the incumbent: seed its fit, write context, spawn the critique agent.

    The agent proposes test statistics, runs the PPC harness (which loads the
    seeded fit — no refit), and writes `critiques.md` describing the significant
    discrepancies. Returns the agent's success flag.
    """
    from src.runtime.coding_agent import run_coding_agent

    critique_dir.mkdir(parents=True, exist_ok=True)
    # Share the inner loop's on-disk cache when it has one; otherwise keep a small
    # per-round cache so the harness process reuses the just-computed fit.
    fit_cache_dir = Path(cache_dir) if cache_dir is not None else critique_dir / ".fit_cache"
    _seed_critique_fit_cache(
        incumbent, models_dir, responses_path, fit_cache_dir, fit_kwargs
    )
    _write_critique_context(
        critique_dir,
        incumbent,
        models_dir,
        responses_path,
        fit_cache_dir,
        n_proposals=n_proposals,
        significance_alpha=significance_alpha,
        n_replicates=n_replicates,
    )

    # Name critique_dir explicitly: the agent runs from REPO_ROOT (so opencode
    # loads the worktree's external_directory grants), NOT from critique_dir, so a
    # bare "in this directory" leaves it guessing where CRITIQUE_CONTEXT.md is —
    # which it sometimes gets wrong, then writes no statistics. Same fix as the
    # candidate agent.
    prompt = (
        f"{_CRITIQUE_PROMPT.read_text(encoding='utf-8')}\n\n"
        f"---\n\nYour working directory for this critique is `{critique_dir}`.\n"
        f"Read `CRITIQUE_CONTEXT.md` there, then write your test statistics into "
        f"`{critique_dir}/test_stats/` (one `test_statistic(df)` per file). You do "
        f"NOT need to run the harness or write `critiques.md` — the pipeline runs the "
        f"posterior-predictive check over your statistics and records the results.\n"
    )
    log_path = critique_dir / "agent.jsonl"
    # The agent only needs to write into test_stats/; give it the model set and
    # responses for reference.
    success, _ = run_coding_agent(
        prompt,
        cwd=REPO_ROOT,
        log_path=log_path,
        allowed_dirs=[critique_dir, models_dir, responses_path.parent],
        timeout_secs=agent_timeout_sec,
        backend=backend,
    )
    # Run the PPC harness ourselves so the results are always persisted, rather
    # than relying on the agent to have run it.
    _persist_critique_results(
        critique_dir,
        incumbent,
        models_dir=models_dir,
        responses_path=responses_path,
        fit_cache_dir=fit_cache_dir,
        fit_kwargs=fit_kwargs,
        n_replicates=n_replicates,
        significance_alpha=significance_alpha,
    )
    return success


def _format_critiques_md(result: Dict[str, Any]) -> str:
    """Render a human/agent-readable critique summary from a PPC result dict."""
    sig = [r for r in result.get("results", []) if r.get("significant")]
    lines = [
        f"# Critique of `{result.get('model')}`",
        "",
        f"{result.get('n_significant', 0)} of {result.get('n_test_statistics', 0)} test "
        f"statistics show a significant discrepancy (p ≤ "
        f"{result.get('significance_alpha')}), over {result.get('n_replicates')} "
        "posterior-predictive replicates.",
        "",
    ]
    if sig:
        lines.append("## Significant discrepancies (a better model should address these)")
        lines.append("")
        lines.append(
            "Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across "
            "this round's statistics. Prioritise discrepancies that survive the FDR "
            "(`q ≤ alpha`); a raw-only hit may be one of several screened at once."
        )
        for r in sig:
            q = r.get("p_value_fdr")
            q_str = f"{q:.3g}" if isinstance(q, (int, float)) and q == q else "n/a"
            fdr_mark = " [survives FDR]" if r.get("significant_fdr") else ""
            lines.append(
                f"- **{r['name']}** — {r.get('description', '')} "
                f"(observed {r['t_observed']:.3g} vs null mean {r['null_mean']:.3g}, "
                f"z={r['z_score']:.2f}, p={r['p_value']:.3g}, q={q_str}){fdr_mark}"
            )
    else:
        lines.append("No statistic showed a significant discrepancy — the incumbent fits these checks.")
    return "\n".join(lines) + "\n"


def _incumbent_response_col(incumbent: str, models_dir: Path) -> str:
    """Name of the incumbent model's observed-response ``pm.Data`` column."""
    from src.models.pymc_inference import load_pymc_model, observed_response_data

    return observed_response_data(load_pymc_model(incumbent, models_dir))


def _write_default_test_statistics(
    test_stats_dir: Path, responses_path: Path, response_col: str
) -> int:
    """Write a deterministic fallback battery of PPC test statistics.

    Used when the critique agent proposes none, so the posterior-predictive
    critique still runs instead of silently producing nothing. The statistics are
    generic discrepancy probes built from the data's *actual* columns — the
    marginal response rate, and the response's linear association with each
    varying numeric feature — so they work for any project's responses. Returns
    the number of statistic files written.
    """
    import pandas as pd

    test_stats_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(responses_path)
    if response_col not in df.columns:
        raise ValueError(
            f"response column {response_col!r} not in {responses_path}; "
            f"columns: {list(df.columns)}"
        )
    numeric = df.select_dtypes(include="number")
    # Probe each varying numeric feature, but not the response itself, an id
    # column, or the response's mechanical complement.
    id_like = {response_col, "chose_right", "participant_id", "trial_index"}
    feature_cols = [
        c
        for c in numeric.columns
        if c not in id_like and float(numeric[c].std(skipna=True) or 0.0) > 0.0
    ]

    stats: Dict[str, tuple[str, str]] = {
        "fallback_mean_response": (
            "Marginal mean of the response column (the overall choice rate).",
            f'    return float(df["{response_col}"].astype(float).mean())',
        )
    }
    for col in feature_cols:
        stats[f"fallback_corr_{col}"] = (
            f"Pearson correlation between the response and the `{col}` feature.",
            f'    x = df["{response_col}"].astype(float)\n'
            f'    y = df["{col}"].astype(float)\n'
            "    if x.std() == 0 or y.std() == 0:\n"
            "        return 0.0\n"
            "    return float(np.corrcoef(x, y)[0, 1])",
        )
    for name, (desc, body) in stats.items():
        code = (
            f"# name: {name}\n"
            f"# description: {desc}\n"
            "def test_statistic(df):\n"
            f"{body}\n"
        )
        (test_stats_dir / f"{name}.py").write_text(code, encoding="utf-8")
    return len(stats)


def _persist_critique_results(
    critique_dir: Path,
    incumbent: str,
    *,
    models_dir: Path,
    responses_path: Path,
    fit_cache_dir: Path,
    fit_kwargs: Dict[str, Any],
    n_replicates: int,
    significance_alpha: float,
) -> None:
    """Run the PPC harness over the agent's ``test_stats/`` and persist the results.

    Writes ``ppc_results.json`` (the per-statistic p-values) and a derived
    ``critiques.md``. This runs deterministically in-process so the critique
    results are always recorded — it does not depend on the agent having run the
    harness. The fit is reused from ``fit_cache_dir`` (no resampling).

    If the agent proposed no usable statistics, a deterministic default battery is
    written first (loudly) so the critique never silently produces nothing.
    """
    from src.critique.ppc import run_ppc_for_model

    test_stats_dir = critique_dir / "test_stats"
    if not test_stats_dir.is_dir() or not any(test_stats_dir.glob("*.py")):
        print(
            "  [critique] agent wrote no test statistics — using default battery",
            flush=True,
        )
        response_col = _incumbent_response_col(incumbent, models_dir)
        n_default = _write_default_test_statistics(
            test_stats_dir, responses_path, response_col
        )
        print(f"  [critique] wrote {n_default} default test statistics", flush=True)

    result = run_ppc_for_model(
        incumbent,
        models_dir,
        responses_path,
        test_stats_dir,
        cache_dir=fit_cache_dir,
        n_replicates=n_replicates,
        significance_alpha=significance_alpha,
        fit_kwargs=fit_kwargs,
    )
    (critique_dir / "ppc_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (critique_dir / "critiques.md").write_text(
        _format_critiques_md(result), encoding="utf-8"
    )
    print(
        f"  [critique] PPC: {result['n_significant']}/{result['n_test_statistics']} "
        "statistics show a significant discrepancy",
        flush=True,
    )


def _run_critique_round(
    round_dir: Path,
    *,
    responses_path: Path,
    models_dir: Path,
    posterior: Dict[str, Any],
    cache_dir: Optional[Path],
    fit_kwargs: Dict[str, Any],
    n_proposals: int,
    significance_alpha: float,
    n_replicates: int,
    agent_timeout_sec: int,
    backend: Optional[str],
) -> Optional[Path]:
    """Critique the current incumbent before a candidate round; return critiques.md.

    Returns the path to the round's ``critiques.md`` when the critique agent
    produced one, else ``None`` (with a loud warning) so a failed critique skips
    forward rather than aborting the whole inner loop.
    """
    incumbent = _best_model(posterior)
    critique_dir = round_dir / "critique"
    print(f"  [critique] critiquing incumbent {incumbent!r}", flush=True)
    try:
        _spawn_critique_agent(
            critique_dir,
            incumbent,
            models_dir=models_dir,
            responses_path=responses_path,
            cache_dir=cache_dir,
            fit_kwargs=fit_kwargs,
            n_proposals=n_proposals,
            significance_alpha=significance_alpha,
            n_replicates=n_replicates,
            agent_timeout_sec=agent_timeout_sec,
            backend=backend,
        )
    except Exception as e:  # a critique failure must not kill a long inner-loop run
        print(f"  [critique] skipped — {type(e).__name__}: {e}", flush=True)
        return None

    critiques_md = critique_dir / "critiques.md"
    if not critiques_md.exists():
        print(
            "  [critique] agent produced no critiques.md — candidates run without it",
            flush=True,
        )
        return None
    return critiques_md


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
    enable_critique: bool = True,
    n_critique_proposals: int = CRITIQUE_N_PROPOSALS,
    critique_significance_alpha: float = CRITIQUE_SIGNIFICANCE_ALPHA,
    n_critique_replicates: int = CRITIQUE_PPC_REPLICATES,
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
    enable_critique
        When True, run a CriticAL posterior-predictive critique of the incumbent
        (best) model before each candidate round and feed the resulting
        ``critiques.md`` to the candidate agents (see ``src/critique/ppc.py``).
    n_critique_proposals, critique_significance_alpha, n_critique_replicates
        Test statistics the critique agent proposes per round, the raw p-value
        threshold for a significant discrepancy, and the posterior-predictive replicates
        forming each statistic's null distribution.

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
        critique_path: Optional[Path] = None
        if enable_critique:
            critique_path = _run_critique_round(
                round_dir,
                responses_path=responses_path,
                models_dir=models_dir,
                posterior=posterior,
                cache_dir=cache_dir,
                fit_kwargs=fit_kwargs,
                n_proposals=n_critique_proposals,
                significance_alpha=critique_significance_alpha,
                n_replicates=n_critique_replicates,
                agent_timeout_sec=agent_timeout_sec,
                backend=backend,
            )
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
                critique_path=critique_path,
            )
            if not _spawn_candidate_agent(
                candidate_dir,
                models_dir=models_dir,
                responses_path=responses_path,
                agent_timeout_sec=agent_timeout_sec,
                backend=backend,
            ):
                continue
            _admit_candidate(
                candidate_dir / "candidate.py",
                models_dir,
                model_name=f"iter{iteration}_candidate{idx}",
                responses_path=responses_path,
                cache_dir=cache_dir,
                fit_kwargs=fit_kwargs,
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
        # Ordered by az.compare rank (0 = best by RAW ELPD-LOO — no complexity
        # prior). elpd_diff/dse are relative to that top model; a model is
        # distinguishable only when elpd_diff is large vs dse (~elpd_diff > 2*dse).
        by_rank = sorted(comparison.items(), key=lambda kv: kv[1]["rank"])
        loo_top = by_rank[0][0] if by_rank else None
        # The "Best model" above is the posterior argmax, which INCLUDES the
        # complexity prior, so it can differ from this table's raw-ELPD rank-0.
        # Call that out explicitly rather than letting the report contradict itself.
        reconcile = ""
        if loo_top is not None and loo_top != best_model:
            reconcile = (
                f" NOTE: this table ranks by **raw** ELPD-LOO, so its top model "
                f"(`{loo_top}`) differs from the **Best model** above "
                f"(`{best_model}`), which is the posterior argmax *including the "
                "complexity prior*. The exported `best_model.py` is the latter."
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
        "model_posterior_path": str(results_dir / "model_posterior.json"),
        "best_model_path": str(results_dir / "best_model.py"),
        "report_path": str(results_dir / "report.md"),
    }
