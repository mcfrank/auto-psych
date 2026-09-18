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
from src.pipelines.inner_loop.import_gate import (
    CANDIDATE_IMPORT_ALLOWLIST,
    check_forbidden_imports,
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
from src.pipelines.outer_loop.columns import RAW_RESPONSE_COLUMNS
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

# Exploration lenses, one per candidate per round (candidate_idx % len). Each
# is a distinct way to search the hypothesis space; together they push rounds
# toward genuine novelty rather than conservative revision of the incumbent
# (the old 3-hint rotation pushed novelty in only one candidate of three).
# Every lens still demands exactly ONE mechanism per model. Override per run
# with the `candidate_hints` parameter / `--hints-file` knob.
DEFAULT_CANDIDATE_HINTS = [
    "Refine one existing hypothesis within its single mechanism — e.g. a "
    "different functional form, prior, or normalization. Do NOT graft cues "
    "from other models onto it.",
    "Propose a mechanism from a genuinely different psychological process "
    "family than anything in the current set — one the current models cannot "
    "express, not a variant of them.",
    "Propose a single mechanism whose predictions would disagree most sharply "
    "with the current best model somewhere in the stimulus space — and say in "
    "your hypothesis where that disagreement lives.",
    "Build a mechanism around information the current models ignore. Derive "
    "the exact statistic your hypothesis needs from the raw sequences with "
    "`compute_features` (order, position, recency, specific sub-sequences).",
    "Propose a process-level account — a memory limit, attention window, "
    "encoding cost, or sequential-updating process — rather than another "
    "statistical summary of the stimulus.",
    "Take a normative account (e.g. Bayesian inference over candidate "
    "generators) and add exactly one principled distortion: a bias, a "
    "resource limit, or a mis-weighting.",
    "Propose something simpler or higher-variance than anything in the set — "
    "if it is wrong, the model comparison will say so loudly, and that is "
    "informative.",
]


# ─────────────────────────────────────────────
# Agent spawning
# ─────────────────────────────────────────────


def _describe_standing(row: Dict[str, Any]) -> str:
    """One clause on a model's standing from its ``az.compare`` row."""
    rank = int(row["rank"])
    elpd = float(row["elpd_loo"])
    diff = float(row["elpd_diff"])
    dse = float(row["dse"])
    if rank == 0:
        text = f"rank 0, the best model on this data, ELPD-LOO {elpd:.1f}"
    else:
        if dse > 0:
            verdict = (
                "statistically tied with the best"
                if diff <= DEFAULT_PRUNE_DSE_MULTIPLIER * dse
                else "distinguishable from the best — it has lost on this data"
            )
            margin = f"{diff / dse:.1f}× dse: {verdict}"
        else:
            margin = "dse 0"
        text = (
            f"rank {rank}, {diff:.1f} ± {dse:.1f} nats behind the best "
            f"({margin}), ELPD-LOO {elpd:.1f}"
        )
    if row.get("loo_unreliable"):
        frac = row.get("frac_bad_k")
        detail = (
            f" ({100 * float(frac):.0f}% of trials with a high Pareto k)"
            if frac is not None
            else ""
        )
        text += f"; PSIS-LOO unreliable{detail} — its ELPD is untrustworthy"
    return text


def _write_existing_hypotheses(
    candidate_dir: Path,
    models_dir: Path,
    current_posterior: Optional[Dict[str, Any]],
    comparison: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Write the hypotheses already in the model set + how each stands.

    Each model's hypothesis is its manifest rationale; its standing is its
    ``az.compare`` row (``rank``, ``elpd_diff ± dse`` against the best,
    PSIS-LOO reliability), best first.  Without a comparison table (no scoring
    yet) only the ELPD-LOO, if any, is shown, in manifest order.  The
    candidate agent reads this to pick a *distinct* or *refined* hypothesis —
    never to merge the top models into a blend.
    Returns the written text (it is also injected into the agent's prompt).
    """
    elpd = (current_posterior or {}).get("elpd_loo", {})
    entries = _manifest_entries(models_dir)
    if comparison:
        ranked = [e for e in entries if e["name"] in comparison]
        unranked = [e for e in entries if e["name"] not in comparison]
        ranked.sort(key=lambda e: int(comparison[e["name"]]["rank"]))
        entries = ranked + unranked
    blocks: List[str] = []
    for entry in entries:
        name = entry["name"]
        hypothesis = (entry.get("rationale") or "").strip() or "(no stated hypothesis)"
        header = f"## {name}"
        if comparison and name in comparison:
            header += f"  — {_describe_standing(comparison[name])}"
        elif comparison and name not in comparison:
            header += "  — no comparison row"
        elif name in elpd:
            header += f"  — ELPD-LOO {elpd[name]:.2f}"
        blocks.append(f"{header}\n\n{hypothesis}\n")
    body = "\n".join(blocks) if blocks else "(no models yet)\n"
    text = (
        "# Existing hypotheses\n\n"
        "Each model below is ONE cognitive hypothesis, with how it stands on the "
        "current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's "
        "deficit against the best and the standard error of that difference: "
        f"within about {DEFAULT_PRUNE_DSE_MULTIPLIER:g}·dse the two are "
        "statistically tied on this data; beyond it the model has lost. "
        "\"PSIS-LOO unreliable\" means the estimate itself is untrustworthy (too "
        "many high-Pareto-k trials), not that the model is bad. Propose a "
        "hypothesis that is genuinely different from these, or a refinement of a "
        "single one of them — never a combination of several.\n\n" + body
    )
    (candidate_dir / "existing_hypotheses.md").write_text(text, encoding="utf-8")
    return text


def _write_candidate_context(
    candidate_dir: Path,
    responses_path: Path,
    models_dir: Path,
    iteration: int,
    candidate_idx: int,
    candidate_count: int,
    current_posterior: Optional[Dict[str, Any]],
    critique_path: Optional[Path] = None,
    hints: Optional[List[str]] = None,
    ledger: Optional[HypothesisLedger] = None,
    comparison: Optional[Dict[str, Dict[str, Any]]] = None,
    lens_index: Optional[int] = None,
) -> Dict[str, Optional[str]]:
    """Write the candidate's context documents and return their text.

    The files (CONTEXT.md, CANDIDATE_BRIEF.md, existing_hypotheses.md,
    attempted_hypotheses.md, critiques.md) stay on disk for audit/
    reproducibility, but the returned strings are what actually reach the
    agent — they are injected verbatim into its prompt (see
    ``_build_candidate_prompt``), so steering content is never optional
    reading. ``lens_index`` selects the exploration lens for this candidate's
    brief (see ``_lens_index``); when ``None`` falls back to
    ``candidate_idx % len(hints)``. With a ``ledger``,
    ``attempted_hypotheses.md`` lists every hypothesis tried earlier (this
    experiment or a previous one) that is no longer in the model set, with
    what happened to it.
    """
    candidate_dir.mkdir(parents=True, exist_ok=True)
    with responses_path.open(encoding="utf-8") as f:
        header = f.readline().strip()
    columns = [c for c in header.split(",") if c]
    raw_set = set(RAW_RESPONSE_COLUMNS)
    feature_cols = [c for c in columns if c not in raw_set]
    raw_sequence_cols = [c for c in ("sequence_a", "sequence_b") if c in columns]
    lines = [
        f"# Inner Loop — round {iteration}, candidate {candidate_idx} of {candidate_count}",
        "",
        f"Responses CSV: `{responses_path}`",
        f"Columns in the responses CSV: `{header}`",
        "",
    ]
    if feature_cols:
        lines += [
            "Read the columns you need as `pm.Data` containers, matching each "
            "container name to a column. **Only numeric columns can back a `pm.Data`** "
            "— the feature columns and `chose_left`.",
        ]
        if raw_sequence_cols:
            lines += [
                "",
                f"The raw H/T sequence strings `{'` and `'.join(raw_sequence_cols)}` are "
                "**not numeric** and cannot be a `pm.Data` directly. To make your "
                "hypothesis depend on an aspect of the sequence the existing feature "
                "columns discard — order, position, recency, or specific sub-sequences "
                "— define a module-level `compute_features(sequence_a, sequence_b) -> "
                "dict[str, float]` in `candidate.py`. The pipeline runs it on the raw "
                "sequences for every trial and exposes each returned key as a column "
                "you read with a matching `pm.Data`. This extends the feature space "
                "beyond the columns above.",
            ]
    else:
        lines += [
            "There are **no feature columns** in this CSV — only the raw H/T "
            "sequence strings and the response (`chose_left`). The only numeric "
            "column you can read directly as a `pm.Data` is `chose_left`.",
            "",
            "Your model **must** compute its own features from the raw sequences. "
            "Define a module-level hook in `candidate.py` — either:",
            "",
            "- `compute_features(sequence_a: str, sequence_b: str) -> dict[str, "
            "float]`: returns named numeric features for one stimulus pair; the "
            "pipeline calls it per trial and exposes each key as a `pm.Data` column.",
            "- `prepare_observed(rows: list[dict]) -> dict[str, np.ndarray]`: "
            "builds all observed arrays at once from the full row list.",
            "",
            "One of these hooks is **required** — without it the model cannot bind "
            "any stimulus input.",
        ]
    allowlist_str = ", ".join(f"`{m}`" for m in sorted(CANDIDATE_IMPORT_ALLOWLIST))
    lines += [
        "",
        "**Allowed imports:** your `candidate.py` may only import from this "
        f"allowlist: {allowlist_str}. Any other import (including the project's "
        "feature library, pandas, or any `src.*` module) causes immediate "
        "rejection at admission. Every helper your model needs must be written "
        "in the file itself — self-contained code only.",
        "",
        "Work in three steps:",
        "1. Write `hypothesis.md` — one cognitive hypothesis, in plain English.",
        "2. Write `model_name.txt` — a short snake_case name for the model (it",
        "   becomes the model's identifier everywhere downstream).",
        "3. Write `candidate.py` — a module-level PyMC model implementing only that",
        "   hypothesis.",
        "",
        "`existing_hypotheses.md` lists the hypotheses already in the model set and",
        "how well each fits. Read it so you propose a *distinct* or *refined*",
        "hypothesis — never a blend of several — under a name not already taken.",
    ]
    if ledger is not None:
        lines += [
            "",
            "`attempted_hypotheses.md` lists the hypotheses tried earlier — in this",
            "experiment or a previous one — that are no longer in the model set,",
            "with what happened to each (pruned after losing by a stated margin, or",
            "rejected at admission, most often as a near-duplicate of a model still",
            "in the set). Do not re-propose any of them under a new name.",
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
    context_text = "\n".join(lines) + "\n"
    (candidate_dir / "CONTEXT.md").write_text(context_text, encoding="utf-8")

    hypotheses_text = _write_existing_hypotheses(
        candidate_dir, models_dir, current_posterior, comparison=comparison
    )
    attempted_text: Optional[str] = None
    if ledger is not None:
        attempted_text = ledger.render_markdown(live_names=_manifest_names(models_dir))
        (candidate_dir / "attempted_hypotheses.md").write_text(
            attempted_text, encoding="utf-8"
        )
    critiques_text: Optional[str] = None
    if critique_path is not None and critique_path.exists():
        critiques_text = critique_path.read_text(encoding="utf-8")
        (candidate_dir / "critiques.md").write_text(critiques_text, encoding="utf-8")

    hints = list(hints) if hints is not None else list(DEFAULT_CANDIDATE_HINTS)
    if lens_index is None:
        lens_index = candidate_idx % len(hints)
    critique_note = (
        "\nIf `critiques.md` is present, prioritise a hypothesis that addresses one of "
        "the significant discrepancies it reports.\n"
        if critique_path is not None
        else ""
    )
    brief = (
        "# Candidate Brief\n\n"
        f"{hints[lens_index]}\n\n"
        "Your candidate must express **exactly one** cognitive hypothesis. Do not "
        "average, weight, or mix cues or mechanisms from several hypotheses into a "
        "single model — a blended mega-model is not a hypothesis.\n"
        f"{critique_note}"
    )
    (candidate_dir / "CANDIDATE_BRIEF.md").write_text(brief, encoding="utf-8")

    return {
        "context": context_text,
        "brief": brief,
        "existing_hypotheses": hypotheses_text,
        "attempted": attempted_text,
        "critiques": critiques_text,
    }


def _build_candidate_prompt(
    candidate_dir: Path, docs: Dict[str, Optional[str]]
) -> str:
    """The candidate agent's full prompt: task instructions + injected context.

    Every context document is inlined as a delimited section so the agent
    cannot skip the round brief, the current hypotheses, or the critique. The
    same documents exist as files in the working directory for reference.
    """
    sections = [
        f"{_THEORY_PROMPT.read_text(encoding='utf-8')}",
        "---",
        f"Your working directory for this candidate is `{candidate_dir}`.\n"
        f"Write `hypothesis.md` (your single hypothesis in plain English), "
        f"`model_name.txt` (a short snake_case name for the model), and "
        f"`candidate.py` (a PyMC model implementing only it) into that "
        f"directory.\n\n"
        f"IMPORTANT: your shell runs from the repository checkout, NOT the "
        f"candidate directory. A relative write like `> candidate.py` lands in "
        f"the wrong place and your candidate is rejected. Always write to the "
        f"absolute paths:\n"
        f"  {candidate_dir}/hypothesis.md\n"
        f"  {candidate_dir}/model_name.txt\n"
        f"  {candidate_dir}/candidate.py\n\n"
        f"Use the **bash** tool with a heredoc to create each file. Example:\n"
        f"```bash\n"
        f"cat << 'EOF' > {candidate_dir}/hypothesis.md\n"
        f"Your hypothesis text here.\n"
        f"EOF\n"
        f"```\n"
        f"Do NOT use the `write` or `edit` tool for creating new files — use "
        f"`bash` with `cat << 'EOF' > path` as shown above.\n\n"
        f"The context documents below are also on disk there for reference.",
        f"## CONTEXT.md\n\n{docs['context']}",
        f"## CANDIDATE_BRIEF.md\n\n{docs['brief']}",
        f"## existing_hypotheses.md\n\n{docs['existing_hypotheses']}",
    ]
    if docs.get("attempted"):
        sections.append(f"## attempted_hypotheses.md\n\n{docs['attempted']}")
    if docs.get("critiques"):
        sections.append(f"## critiques.md\n\n{docs['critiques']}")
    return "\n\n".join(sections) + "\n"


def _spawn_candidate_agent(
    candidate_dir: Path,
    docs: Dict[str, Optional[str]],
    *,
    models_dir: Path,
    responses_path: Path,
    agent_timeout_sec: int,
    backend: Optional[str],
    agent_model: Optional[str] = None,
    agent_root: Optional[Path] = None,
) -> bool:
    from src.runtime.coding_agent import run_coding_agent

    # Run from agent_root (the scrubbed agent tree), not from the harness
    # checkout. opencode discovers its external_directory grants by walking up
    # from cwd to the worktree's opencode.json, so cwd must be a tree that has
    # .here and opencode.json. The agent tree is scrubbed of feature code,
    # research library modules and GT-recipe files, so the agent cannot read
    # them. The candidate_dir is named explicitly since it is not the cwd.
    cwd = agent_root if agent_root is not None else REPO_ROOT
    prompt = _build_candidate_prompt(candidate_dir, docs)
    log_path = candidate_dir / "agent.jsonl"
    success, _ = run_coding_agent(
        prompt,
        cwd=cwd,
        log_path=log_path,
        allowed_dirs=[candidate_dir, models_dir, responses_path.parent],
        timeout_secs=agent_timeout_sec,
        backend=backend,
        model=agent_model,
        usage_label="inner:candidate",
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
