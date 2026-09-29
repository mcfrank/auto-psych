"""CriticAL critique round: posterior-predictive model criticism.

Before each candidate-generation round the inner loop critiques the current
incumbent (best) model via a posterior-predictive check.  The critique agent
proposes test statistics; the PPC harness scores each as a two-sided empirical
p-value; significant discrepancies steer the next round of candidates.

The critique either works or is visibly absent. The agent's context is inlined
into its prompt; an agent that writes no usable statistic is re-spawned once
(``MAX_CRITIQUE_RETRIES``); if the retry writes none either, the round has
**no** critique: no ``critiques.md`` is written, the candidates run without
one, and the round's ``history.json`` entry records ``"no_critique"``. There is
no pipeline-written fallback battery (see ``_persist_critique_results``).

Extracted from ``pymc_orchestrator`` — only ``_run_critique_round`` (and the
status vocabulary) is used from the main loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipelines.inner_loop.import_gate import check_forbidden_imports
from src.pipelines.inner_loop.model_zoo import _manifest_entries
from src.runtime.coding_agent import AgentPermissionDenied
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
# A critique agent that writes no usable test statistic is re-spawned this many
# times (with a prompt that says the previous attempt wrote nothing). After
# that the round proceeds with no critique, recorded as such.
MAX_CRITIQUE_RETRIES = 1

# Per-round critique status, recorded in every round's history.json entry
# under "critique" (``scoring._record_history_step``).
CRITIQUE_STATUS_CRITIQUED = "critiqued"  # statistics proposed and scored
CRITIQUE_STATUS_NONE = "no_critique"  # the round ran without a critique
CRITIQUE_STATUS_DISABLED = "disabled"  # enable_critique=False


def critique_disabled_status() -> Dict[str, Any]:
    """The status recorded for a round run with the critique switched off."""
    return {"status": CRITIQUE_STATUS_DISABLED}


@dataclass(frozen=True)
class CritiqueRoundOutcome:
    """What a critique round produced.

    ``critiques_md`` is the round's ``critiques.md`` to hand to the candidate
    agents, or ``None`` when there was no critique; ``status`` is the record
    written into the round's ``history.json`` entry (see the
    ``CRITIQUE_STATUS_*`` vocabulary).
    """

    critiques_md: Optional[Path]
    status: Dict[str, Any]


def _critique_log_name(attempt: int) -> str:
    """The agent log for the given attempt: ``agent.jsonl``, then ``agent.retry_<n>.jsonl``."""
    return "agent.jsonl" if attempt == 0 else f"agent.retry_{attempt}.jsonl"


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
) -> str:
    """Write CRITIQUE_CONTEXT.md (the incumbent, the data schema, the PPC command).

    Returns the text: it is inlined into the critique agent's prompt by
    ``_build_critique_prompt``, so the agent never has to open the file. The
    file stays on disk for audit.
    """
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
    text = "\n".join(lines) + "\n"
    (critique_dir / "CRITIQUE_CONTEXT.md").write_text(text, encoding="utf-8")
    return text


def _build_critique_prompt(critique_dir: Path, context_text: str, *, attempt: int = 0) -> str:
    """The critique agent's full prompt: the critique brief + the inlined context.

    The context is a delimited section of the prompt, exactly as the candidate
    agent's documents are (``_build_candidate_prompt``). It used to be a file
    the agent had to read first; when that read was denied the agent wrote no
    statistics and exited, and nobody noticed because a fallback battery
    filled in. ``attempt`` > 0 marks a retry after an attempt that wrote no
    usable statistic, and says so.
    """
    if not context_text:
        raise ValueError("the critique context is empty; nothing to inline into the prompt")
    test_stats_dir = critique_dir / "test_stats"
    # Name critique_dir explicitly: the agent runs from agent_root (opencode's
    # session directory, whose opencode.json is the permission config in force
    # — the launcher pins PWD to it), NOT from critique_dir, so a bare "in this
    # directory" leaves it guessing where to write.
    sections = [
        _CRITIQUE_PROMPT.read_text(encoding="utf-8"),
        "---",
        f"Your working directory for this critique is `{critique_dir}`.\n"
        f"Write your test statistics into `{test_stats_dir}/` (one "
        f"`test_statistic(df)` per file). You do NOT need to run the harness or "
        f"write `critiques.md` — the pipeline runs the posterior-predictive check "
        f"over your statistics and records the results.\n\n"
        f"Use the **bash** tool with a heredoc to create each file. Example:\n"
        f"```bash\n"
        f"mkdir -p {test_stats_dir}\n"
        f"cat << 'EOF' > {test_stats_dir}/my_statistic.py\n"
        f"def test_statistic(df): ...\n"
        f"EOF\n"
        f"```\n"
        f"Do NOT use the `write` tool — use `bash` with `cat << 'EOF' > path`.\n\n"
        f"The critique context below is also on disk at "
        f"`{critique_dir}/CRITIQUE_CONTEXT.md` for reference.",
        f"## CRITIQUE_CONTEXT.md\n\n{context_text}",
    ]
    if attempt > 0:
        sections.insert(
            2,
            f"NOTE: this is the second attempt at this critique round. The first "
            f"attempt ended without a single usable test statistic in "
            f"`{test_stats_dir}/`. If that happens again the round proceeds with "
            f"no critique at all, so write the statistic files first, before "
            f"anything else.",
        )
    return "\n\n".join(sections) + "\n"


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
) -> Dict[str, Any]:
    """Critique the incumbent: seed its fit, write context, spawn the critique agent.

    The agent proposes test statistics into ``critique_dir/test_stats``; the
    pipeline then runs the PPC harness over them (loading the seeded fit — no
    refit) and writes ``ppc_results.json`` + ``critiques.md``. An agent that
    leaves no usable statistic is re-spawned up to ``MAX_CRITIQUE_RETRIES``
    times, each attempt with its own log; when every attempt leaves none the
    harness is not run and the round has no critique.

    Returns the round's status record: ``{"status": "critiqued", "incumbent",
    "attempts", "n_statistics", "n_significant", "n_significant_fdr"}`` or
    ``{"status": "no_critique", "incumbent", "attempts", "reason"}``.
    """
    from src.runtime.coding_agent import run_coding_agent

    critique_dir.mkdir(parents=True, exist_ok=True)
    # Share the inner loop's on-disk cache when it has one; otherwise keep a small
    # per-round cache so the harness process reuses the just-computed fit.
    fit_cache_dir = Path(cache_dir) if cache_dir is not None else critique_dir / ".fit_cache"
    _seed_critique_fit_cache(
        incumbent, models_dir, responses_path, fit_cache_dir, fit_kwargs
    )
    context_text = _write_critique_context(
        critique_dir,
        incumbent,
        models_dir,
        responses_path,
        fit_cache_dir,
        n_proposals=n_proposals,
        significance_alpha=significance_alpha,
        n_replicates=n_replicates,
    )

    # The agent runs from agent_root (opencode's session directory, whose
    # opencode.json is the permission config in force — the launcher pins PWD
    # to it); the prompt names critique_dir explicitly.
    cwd = agent_root if agent_root is not None else REPO_ROOT
    test_stats_dir = critique_dir / "test_stats"
    n_attempts = 1 + MAX_CRITIQUE_RETRIES
    usable: List[Path] = []
    attempts_made = 0
    for attempt in range(n_attempts):
        attempts_made = attempt + 1
        success, _ = run_coding_agent(
            _build_critique_prompt(critique_dir, context_text, attempt=attempt),
            cwd=cwd,
            log_path=critique_dir / _critique_log_name(attempt),
            allowed_dirs=[critique_dir, models_dir, responses_path.parent],
            timeout_secs=agent_timeout_sec,
            backend=backend,
            model=agent_model,
            usage_label="inner:critique",
        )
        usable = _usable_test_statistics(test_stats_dir)
        if usable:
            break
        print(
            f"  [critique] attempt {attempts_made}/{n_attempts}: the critique agent "
            f"wrote no usable test statistic in {test_stats_dir} (agent "
            f"{'succeeded' if success else 'failed'})"
            + (" — retrying" if attempts_made < n_attempts else ""),
            flush=True,
        )
    if not usable:
        reason = (
            f"the critique agent wrote no usable test statistic in {n_attempts} attempts"
        )
        print(f"  [critique] NO CRITIQUE this round: {reason}", flush=True)
        return {
            "status": CRITIQUE_STATUS_NONE,
            "incumbent": incumbent,
            "attempts": attempts_made,
            "reason": reason,
        }
    # Run the PPC harness ourselves so the results are always persisted, rather
    # than relying on the agent to have run it.
    result = _persist_critique_results(
        critique_dir,
        incumbent,
        models_dir=models_dir,
        responses_path=responses_path,
        fit_cache_dir=fit_cache_dir,
        fit_kwargs=fit_kwargs,
        n_replicates=n_replicates,
        significance_alpha=significance_alpha,
    )
    return {
        "status": CRITIQUE_STATUS_CRITIQUED,
        "incumbent": incumbent,
        "attempts": attempts_made,
        "n_statistics": int(result["n_test_statistics"]),
        "n_significant": int(result["n_significant"]),
        "n_significant_fdr": int(result["n_significant_fdr"]),
    }


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


def _usable_test_statistics(test_stats_dir: Path) -> List[Path]:
    """The agent's statistic files that pass the import gate, sorted.

    A file importing outside the candidate allowlist is deleted (loudly): it
    could reach the project's feature code, which the critique must not see.
    An absent directory simply has no statistics.
    """
    if not test_stats_dir.is_dir():
        return []
    usable: List[Path] = []
    for stat_file in sorted(test_stats_dir.glob("*.py")):
        forbidden = check_forbidden_imports(stat_file.read_text(encoding="utf-8"))
        if forbidden:
            print(
                f"  [critique] removing {stat_file.name}: forbidden import "
                f"{', '.join(forbidden)}",
                flush=True,
            )
            stat_file.unlink()
            continue
        usable.append(stat_file)
    return usable


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
) -> Dict[str, Any]:
    """Run the PPC harness over the agent's ``test_stats/`` and persist the results.

    Writes ``ppc_results.json`` (the per-statistic p-values) and a derived
    ``critiques.md``, and returns the harness's result dict. This runs
    deterministically in-process so the critique results are always recorded
    — it does not depend on the agent having run the harness. The fit is
    reused from ``fit_cache_dir`` (no resampling).

    Requires at least one usable statistic and raises otherwise: there is no
    pipeline-written fallback battery. (There used to be one; under the
    raw-only schema it reduced to the marginal choice rate, which any fitted
    Bernoulli likelihood matches by construction, so a critique agent that had
    written nothing looked like one that had found nothing — through a whole
    sweep.) The caller retries the agent and then records "no critique".
    """
    from src.critique.ppc import run_ppc_for_model

    test_stats_dir = critique_dir / "test_stats"
    if not _usable_test_statistics(test_stats_dir):
        raise ValueError(
            f"no usable test statistic in {test_stats_dir}; the critique agent "
            "must write at least one before the PPC harness can run"
        )

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
    return result


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
) -> CritiqueRoundOutcome:
    """Critique the current incumbent before a candidate round.

    The incumbent is the model the loop would export right now
    (``_best_exportable_model``), so the critique targets the model that is
    actually carried, not a posterior argmax that may be unreliable.

    Returns the round's ``critiques.md`` (to feed the candidate agents) with a
    ``"critiqued"`` status when the agent's statistics were scored; otherwise
    no path and a ``"no_critique"`` status carrying the reason — the agent
    wrote no usable statistic in any attempt, or the critique crashed (a
    critique failure must not abort a long inner-loop run, but it is recorded,
    never swallowed). A permission denial is re-raised: that is a misconfigured
    launch every later agent would hit too.
    """
    # Lazy import to avoid circular dependency (pymc_orchestrator imports us).
    from src.pipelines.inner_loop.scoring import _best_exportable_model

    incumbent = _best_exportable_model(posterior, comparison)
    critique_dir = round_dir / "critique"
    print(f"  [critique] critiquing incumbent {incumbent!r}", flush=True)
    try:
        status = _spawn_critique_agent(
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
    except AgentPermissionDenied:
        # A misconfigured launch, not a critique failure: every later agent
        # would be denied the same way. Let it kill the run.
        raise
    except Exception as e:  # a critique failure must not kill a long inner-loop run
        reason = f"{type(e).__name__}: {e}"
        print(f"  [critique] NO CRITIQUE this round — {reason}", flush=True)
        return CritiqueRoundOutcome(
            None,
            {"status": CRITIQUE_STATUS_NONE, "incumbent": incumbent, "reason": reason},
        )

    if status["status"] != CRITIQUE_STATUS_CRITIQUED:
        print(
            "  [critique] candidates run without a critique this round", flush=True
        )
        return CritiqueRoundOutcome(None, status)
    critiques_md = critique_dir / "critiques.md"
    if not critiques_md.exists():
        raise RuntimeError(
            f"critique status is {status['status']!r} but {critiques_md} was not "
            "written; _persist_critique_results must write it whenever it scores"
        )
    return CritiqueRoundOutcome(critiques_md, status)
