"""Candidate-agent spawning and prompt assembly for the PyMC inner loop.

This module contains everything that prepares a candidate agent's context
documents, assembles its prompt and launches the coding-agent subprocess.
Extracted from ``pymc_orchestrator`` to keep that module focused on the
orchestration loop itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger
from src.pipelines.inner_loop.import_gate import CANDIDATE_IMPORT_ALLOWLIST
from src.pipelines.inner_loop.model_zoo import (
    DEFAULT_PRUNE_DSE_MULTIPLIER,
    _manifest_entries,
    _manifest_names,
)
from src.pipelines.outer_loop.columns import RAW_RESPONSE_COLUMNS
from src.runtime.config import REPO_ROOT

_PKG_DIR = Path(__file__).resolve().parent
_THEORY_PROMPT = _PKG_DIR / "prompts" / "pymc_theory.md"

# Exploration lenses, one per candidate per round. Each is a distinct way to
# search the hypothesis space; together they push rounds toward genuine novelty
# rather than conservative revision of the incumbent. Every lens still demands
# exactly ONE mechanism per model. Override per run with the `candidate_hints`
# parameter / `--hints-file` knob. See the decision record for the rationale.
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
