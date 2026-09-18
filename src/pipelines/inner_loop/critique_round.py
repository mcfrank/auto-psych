"""CriticAL critique round: posterior-predictive model criticism.

Before each candidate-generation round the inner loop critiques the current
incumbent (best) model via a posterior-predictive check.  The critique agent
proposes test statistics; the PPC harness scores each as a two-sided empirical
p-value; significant discrepancies steer the next round of candidates.

Extracted from ``pymc_orchestrator`` — only ``_run_critique_round`` is called
from the main loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipelines.inner_loop.import_gate import check_forbidden_imports
from src.pipelines.inner_loop.model_zoo import _manifest_entries
from src.runtime.config import REPO_ROOT

_PKG_DIR = Path(__file__).resolve().parent
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
    agent_model: Optional[str] = None,
    agent_root: Optional[Path] = None,
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

    # Name critique_dir explicitly: the agent runs from agent_root (so opencode
    # loads the worktree's external_directory grants), NOT from critique_dir, so a
    # bare "in this directory" leaves it guessing where CRITIQUE_CONTEXT.md is —
    # which it sometimes gets wrong, then writes no statistics. Same fix as the
    # candidate agent.
    cwd = agent_root if agent_root is not None else REPO_ROOT
    prompt = (
        f"{_CRITIQUE_PROMPT.read_text(encoding='utf-8')}\n\n"
        f"---\n\nYour working directory for this critique is `{critique_dir}`.\n"
        f"Read `CRITIQUE_CONTEXT.md` there, then write your test statistics into "
        f"`{critique_dir}/test_stats/` (one `test_statistic(df)` per file). You do "
        f"NOT need to run the harness or write `critiques.md` — the pipeline runs the "
        f"posterior-predictive check over your statistics and records the results.\n\n"
        f"Use the **bash** tool with a heredoc to create each file. Example:\n"
        f"```bash\n"
        f"mkdir -p {critique_dir}/test_stats\n"
        f"cat << 'EOF' > {critique_dir}/test_stats/my_statistic.py\n"
        f"def test_statistic(df): ...\n"
        f"EOF\n"
        f"```\n"
        f"Do NOT use the `write` tool — use `bash` with `cat << 'EOF' > path`.\n"
    )
    log_path = critique_dir / "agent.jsonl"
    success, _ = run_coding_agent(
        prompt,
        cwd=cwd,
        log_path=log_path,
        allowed_dirs=[critique_dir, models_dir, responses_path.parent],
        timeout_secs=agent_timeout_sec,
        backend=backend,
        model=agent_model,
        usage_label="inner:critique",
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

    for stat_file in sorted(test_stats_dir.glob("*.py")):
        forbidden = check_forbidden_imports(stat_file.read_text(encoding="utf-8"))
        if forbidden:
            print(
                f"  [critique] removing {stat_file.name}: forbidden import "
                f"{', '.join(forbidden)}",
                flush=True,
            )
            stat_file.unlink()

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
    comparison: Dict[str, Dict[str, Any]],
    cache_dir: Optional[Path],
    fit_kwargs: Dict[str, Any],
    n_proposals: int,
    significance_alpha: float,
    n_replicates: int,
    agent_timeout_sec: int,
    backend: Optional[str],
    agent_model: Optional[str] = None,
    agent_root: Optional[Path] = None,
) -> Optional[Path]:
    """Critique the current incumbent before a candidate round; return critiques.md.

    The incumbent is the model the loop would export right now
    (``_best_exportable_model``), so the critique targets the model that is
    actually carried, not a posterior argmax that may be unreliable.

    Returns the path to the round's ``critiques.md`` when the critique agent
    produced one, else ``None`` (with a loud warning) so a failed critique skips
    forward rather than aborting the whole inner loop.
    """
    # Lazy import to avoid circular dependency (pymc_orchestrator imports us).
    from src.pipelines.inner_loop.pymc_orchestrator import _best_exportable_model

    incumbent = _best_exportable_model(posterior, comparison)
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
            agent_model=agent_model,
            agent_root=agent_root,
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
