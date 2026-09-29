"""Candidate-agent spawning and prompt assembly for the PyMC inner loop.

This module contains everything that prepares a candidate agent's context
documents, assembles its prompt and launches the coding-agent subprocess.
Extracted from ``pymc_orchestrator`` to keep that module focused on the
orchestration loop itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.pipelines.inner_loop.task_description import read_task_description
from src.models.mcmc_defaults import (
    CANDIDATE_FIT_TIME_LIMIT_SEC,
    ESCALATED_TARGET_ACCEPT,
    MAX_R_HAT,
    MIN_BULK_ESS,
    CANDIDATE_CHECK_CHAINS,
    CANDIDATE_CHECK_DRAWS,
    CANDIDATE_CHECK_TUNE,
)
from src.pipelines.inner_loop.check_candidate import check_candidate_command
from src.pipelines.inner_loop.hypothesis_ledger import (
    HypothesisLedger,
    LedgerEntry,
    collapse_whitespace,
)
from src.pipelines.inner_loop.import_gate import CANDIDATE_IMPORT_ALLOWLIST
from src.pipelines.inner_loop.model_zoo import (
    DEFAULT_PRUNE_DSE_MULTIPLIER,
    SLOT_EXPLORE,
    SLOT_REFINE_CHOSEN,
    SLOT_REFINE_INCUMBENT,
    SLOT_ROLES,
    _manifest_entries,
    _manifest_names,
    parse_prune_margin,
)
from src.pipelines.outer_loop.columns import RAW_RESPONSE_COLUMNS
from src.runtime.config import REPO_ROOT

_PKG_DIR = Path(__file__).resolve().parent
_THEORY_PROMPT = _PKG_DIR / "prompts" / "pymc_theory.md"

# Exploration lenses, one per exploratory slot per round. Each is a distinct
# way to search the hypothesis space; together they push rounds toward genuine
# novelty rather than conservative revision of the incumbent. Every lens still
# demands exactly ONE mechanism per model. Twelve lenses so a six-candidate
# round (three exploratory slots) walks four rounds without repeating one.
# Override per run with the `candidate_hints` parameter / `--hints-file` knob.
# See the decision record for the rationale.
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
    "Propose a mechanism in which the two sequences are not judged one at a "
    "time: the comparison itself does the work — a contrast effect, anchoring "
    "on whichever sequence is read first, or a shared reference point both are "
    "judged against — so the same sequence would be judged differently beside "
    "a different partner.",
    "Propose a mechanism driven by the single most salient local feature — the "
    "longest run, the most lopsided window, or the presence of one specific "
    "motif — a maximum over sub-sequences rather than a sum or average over the "
    "whole sequence, so one striking stretch decides the judgment.",
    "Propose a mechanism at the level of the decision rule rather than the "
    "evidence: a lapse rate, a bias toward one response side, an indifference "
    "band within which the choice is a coin flip, or probability matching — "
    "with the evidence itself kept as simple as the simplest current model.",
    "Propose a mechanism in which people track a running tally as they read "
    "the sequence — the cumulative lead of heads over tails, or how far and how "
    "often it drifts from balance before correcting — so the path to the final "
    "count matters, not the count.",
    "Propose an exemplar account: people carry a few remembered examples of "
    "what random and designed sequences look like and judge a new sequence by "
    "its similarity to the nearest ones — define the prototypes and the "
    "similarity explicitly, as the single mechanism.",
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


def _pruned_source(models_dir: Path, entry: LedgerEntry) -> Optional[Path]:
    """The file of a model the ledger records as pruned, or ``None``.

    Pruning runs once, at the end of an experiment, and moves the file to
    that experiment's ``model_loop/models/pruned/``; each experiment starts a
    fresh zoo. So during the rounds this menu is written in, the current
    zoo's ``pruned/`` is empty, and the file of every model on the menu's
    pruned list is in an earlier experiment's zoo. The prune's ledger context
    starts with that experiment's directory name (``experiment2 end of
    experiment``), a sibling of the current experiment's directory
    (``models_dir`` is ``<run>/<experiment>/model_loop/models``). The menu
    used to look only in the current zoo, and so never showed a pruned
    model's source although the brief promised it.
    """
    in_this_zoo = models_dir / "pruned" / f"{entry.name}.py"
    if in_this_zoo.exists():
        return in_this_zoo
    experiment = entry.context.split(" ", 1)[0] if entry.context else ""
    if not experiment:
        return None
    run_root = models_dir.parent.parent.parent
    source = run_root / experiment / "model_loop" / "models" / "pruned" / f"{entry.name}.py"
    return source if source.exists() else None


def _write_refinement_menu(
    candidate_dir: Path,
    models_dir: Path,
    comparison: Optional[Dict[str, Dict[str, Any]]],
    ledger: HypothesisLedger,
    *,
    incumbent: str,
) -> str:
    """Write ``refinement_menu.md``: the models a refinement slot may refine.

    One ranked list of every model in the project other than the incumbent:
    the live models, best first by their ``az.compare`` standing, then the
    models the ledger records as pruned (in this experiment or an earlier
    one), narrowest margin first — each with its hypothesis in full, its
    standing or prune margin, and its source (``models/<name>.py``, or the
    ``pruned/<name>.py`` of the experiment that pruned it:
    ``_pruned_source``; a pruned model whose file cannot be found is listed
    by hypothesis alone, and says so). Framed as a menu, not a blacklist: the retired
    hypotheses an exploratory slot is told not to re-propose are exactly
    what a refinement slot exists to draw on. Returns the text (it is also
    injected into the agent's prompt).
    """
    entries = _manifest_entries(models_dir)
    names = [e["name"] for e in entries]
    if incumbent not in names:
        raise ValueError(
            f"The incumbent {incumbent!r} is not in the model set {names}; the "
            "refinement menu lists the models other than it."
        )
    live = [e for e in entries if e["name"] != incumbent]
    if comparison:
        ranked = [e for e in live if e["name"] in comparison]
        unranked = [e for e in live if e["name"] not in comparison]
        ranked.sort(key=lambda e: int(comparison[e["name"]]["rank"]))
        live = ranked + unranked
    pruned = sorted(
        ledger.pruned(live_names=names), key=lambda e: parse_prune_margin(e.detail)
    )

    lines = [
        "# Refinement menu",
        "",
        f"The models you may refine, other than the incumbent `{incumbent}`. Each "
        "entry is one mechanism: its hypothesis in full as its author stated "
        "it, how it stands, and where its source is. This is a menu, not a "
        "list of what is ruled out: a pruned model lost to the best model on "
        "the data it was scored on, by the stated margin, but its mechanism may "
        "be partly right — and a model that lost narrowly is the most promising "
        "target here. Read the source of the model you pick before you write "
        "anything.",
        "",
        "## Live models other than the incumbent (in the set; best first)",
        "",
    ]
    if not live:
        lines += ["No live model other than the incumbent.", ""]
    for entry in live:
        name = entry["name"]
        if comparison and name in comparison:
            standing = _describe_standing(comparison[name])
        else:
            standing = "no comparison row"
        hypothesis = (entry.get("rationale") or "").strip() or "(no stated hypothesis)"
        lines += [
            f"### {name} — {standing}",
            "",
            f"**Hypothesis:** {collapse_whitespace(hypothesis)}",
            "",
            f"**Source:** `{models_dir / f'{name}.py'}`",
            "",
        ]
    lines += ["## Pruned models (out of the set; narrowest margin first)", ""]
    if not pruned:
        lines += ["No model has been pruned yet in this project.", ""]
    for entry in pruned:
        source = _pruned_source(models_dir, entry)
        if source is not None:
            source_line = f"**Source:** `{source}`"
        else:
            source_line = (
                "**Source:** none on disk — only its hypothesis above is carried."
            )
        outcome = f"pruned ({entry.context})" if entry.context else "pruned"
        lines += [
            f"### {entry.name} — {outcome}",
            "",
            f"**Margin:** {entry.detail}",
            "",
            f"**Hypothesis:** {entry.hypothesis or '*(none recorded)*'}",
            "",
            source_line,
            "",
        ]
    text = "\n".join(lines)
    (candidate_dir / "refinement_menu.md").write_text(text, encoding="utf-8")
    return text


# What the round brief says the slot's job is, in CONTEXT.md's heading.
_ROLE_LABELS = {
    SLOT_EXPLORE: "exploratory slot",
    SLOT_REFINE_INCUMBENT: "refinement slot: the incumbent",
    SLOT_REFINE_CHOSEN: "refinement slot: a model of your choosing",
}

# The two rules a refinement slot lifts (the exploratory brief keeps both;
# lens 0 of the battery carries the anti-grafting clause).
_ONE_HYPOTHESIS_RULE = (
    "Your candidate must express **exactly one** cognitive hypothesis. Do not "
    "average, weight, or mix cues or mechanisms from several hypotheses into a "
    "single model — a blended mega-model is not a hypothesis.\n"
)
_REFINEMENT_RULES = (
    "For this slot the rule against grafting cues from other models onto a "
    "hypothesis is lifted, and so is the rule against composing mechanisms: "
    "you may add a component from another model. Two things still hold: make "
    "**one** deliberate, stated change (a grab-bag of cues added to fit better "
    "is not a refinement), and change something that matters — the novelty "
    "gate rejects a candidate whose predictions match a live model's across "
    "the stimulus space"
)


def _incumbent_brief(
    models_dir: Path,
    comparison: Optional[Dict[str, Dict[str, Any]]],
    incumbent: str,
    critique_note: str,
) -> str:
    """The brief of an incumbent-refinement slot: the incumbent, named."""
    entries = {e["name"]: e for e in _manifest_entries(models_dir)}
    if incumbent not in entries:
        raise ValueError(
            f"The incumbent {incumbent!r} is not in the model set "
            f"{sorted(entries)}; a refinement slot cannot name it."
        )
    hypothesis = (entries[incumbent].get("rationale") or "").strip()
    hypothesis = collapse_whitespace(hypothesis) or "(no stated hypothesis)"
    if comparison and incumbent in comparison:
        standing = _describe_standing(comparison[incumbent])
    else:
        standing = "the best model of the latest scoring step"
    return (
        "# Candidate Brief\n\n"
        f"Refine the incumbent: `{incumbent}` — {standing}. Its hypothesis, as "
        "its author stated it:\n\n"
        f"> {hypothesis}\n\n"
        f"Its source is `{models_dir / f'{incumbent}.py'}`. Read it before you "
        "write anything.\n\n"
        "Produce the version of this model you believe would beat it on the "
        "current data: keep the mechanism that makes it win and change what it "
        "gets wrong — a different functional form, prior or normalisation of "
        "its mechanism, a cue it ignores, or a component taken from another "
        f"model in `refinement_menu.md`. {_REFINEMENT_RULES}, the incumbent's "
        "included.\n\n"
        "Say in `hypothesis.md` which model you refined and what you changed: "
        "that is part of the claim.\n"
        f"{critique_note}"
    )


def _chosen_brief(incumbent: str, critique_note: str) -> str:
    """The brief of an agent-chosen refinement slot: pick from the menu."""
    return (
        "# Candidate Brief\n\n"
        "Refine a model of your choosing — any model in `refinement_menu.md`, "
        "which lists every model in this project other than the incumbent "
        f"`{incumbent}` (the incumbent has its own refinement slots this "
        "round). The menu gives the other live models, with their standing "
        "against the best, and the models pruned earlier, with the margin by "
        "which each lost and its source (in the `pruned/` directory of the "
        "experiment that pruned it). A pruned model lost on the data it was scored "
        "on, but its mechanism may be partly right, and this slot exists to "
        "find out. Choose the model whose mechanism you judge most promising "
        "and most improvable — a narrow loser over a distant one, unless you "
        "see exactly what the distant one got wrong — read its source, and "
        "produce the version of it that could overtake the incumbent.\n\n"
        f"{_REFINEMENT_RULES}, and a pruned model re-implemented as it was "
        "would pass the gate only to lose again by the same margin.\n\n"
        "Say in `hypothesis.md` which model you chose and what you changed: "
        "that is part of the claim.\n"
        f"{critique_note}"
    )


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
    attempt_note: Optional[str] = None,
    omit_from_attempted: Iterable[str] = (),
    role: str = SLOT_EXPLORE,
    incumbent: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Write the candidate's context documents and return their text.

    The files (CONTEXT.md, CANDIDATE_BRIEF.md, existing_hypotheses.md,
    attempted_hypotheses.md or refinement_menu.md, critiques.md,
    ATTEMPT_NOTE.md) stay on disk for audit/reproducibility, but the returned
    strings are what actually reach the agent — they are injected verbatim
    into its prompt (see ``_build_candidate_prompt``), so steering content is
    never optional reading.

    ``role`` is the slot's role (``model_zoo.slot_roles``). An exploratory
    slot's brief carries the lens ``lens_index`` selects (``None`` ⇒
    ``candidate_idx % len(hints)``) and the one-hypothesis-no-blend rule, and
    — with a ``ledger`` — ``attempted_hypotheses.md``: every hypothesis tried
    earlier (this experiment or a previous one) that is no longer in the
    model set, framed as "do not re-propose". A refinement slot
    (``SLOT_REFINE_INCUMBENT``: refine ``incumbent``, named in the brief;
    ``SLOT_REFINE_CHOSEN``: refine a non-incumbent model of the agent's
    choosing) carries no lens, lifts the anti-grafting and anti-composition
    rules, and gets ``refinement_menu.md`` — the live and pruned models as
    targets — in place of the retired list; it requires ``incumbent`` and
    ``ledger``. The role is the slot's assignment: which model the agent
    refined is stated in its ``hypothesis.md`` and never parsed.

    ``attempt_note`` is the note a retry or repair attempt of this slot opens
    with (see ``_retry_note`` / ``_repair_note``); ``None`` for a slot's first
    attempt. ``omit_from_attempted`` names whose ledger history is this slot's
    *own* earlier attempt: a repair must not be told not to re-propose the very
    model it is repairing, so those names are left out of
    ``attempted_hypotheses.md``.
    """
    if role not in SLOT_ROLES:
        raise ValueError(f"Unknown slot role {role!r}; expected one of {SLOT_ROLES}.")
    refining = role != SLOT_EXPLORE
    if refining and incumbent is None:
        raise ValueError(f"A {role!r} slot needs the incumbent's name.")
    if refining and ledger is None:
        raise ValueError(f"A {role!r} slot's refinement menu needs the ledger.")
    candidate_dir.mkdir(parents=True, exist_ok=True)
    with responses_path.open(encoding="utf-8") as f:
        header = f.readline().strip()
    columns = [c for c in header.split(",") if c]
    raw_set = set(RAW_RESPONSE_COLUMNS)
    feature_cols = [c for c in columns if c not in raw_set]
    raw_sequence_cols = [c for c in ("sequence_a", "sequence_b") if c in columns]
    lines = [
        f"# Inner Loop — round {iteration}, candidate {candidate_idx} of "
        f"{candidate_count} ({_ROLE_LABELS[role]})",
        "",
        read_task_description(responses_path),
        "",
        "## Your job",
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
        "in the file itself — self-contained code only. It may not read files "
        "or reach the interpreter either (`open`, `np.load`, `eval`, "
        "`__import__` and the like are rejected).",
        "",
        "Work in three steps:",
        "1. Write `hypothesis.md` — one cognitive hypothesis, in plain English.",
        "2. Write `model_name.txt` — a short snake_case name for the model (it",
        "   becomes the model's identifier everywhere downstream).",
        "3. Write `candidate.py` — a module-level PyMC model implementing only that",
        "   hypothesis.",
        "",
        "**Check your model before you finish.** Run this from the repository "
        "checkout (your shell's working directory). It runs the admission gates "
        "you can act on — the import allowlist, a loadable module-level "
        "`model: pm.Model`, a finite log-probability on the real responses, a "
        f"short MCMC fit ({CANDIDATE_CHECK_DRAWS} draws, {CANDIDATE_CHECK_TUNE} "
        f"tune, {CANDIDATE_CHECK_CHAINS} chain: a smoke test, not a full "
        "production fit) and a finite ELPD-LOO — and prints `OK` or the exact "
        "reason admission would reject the file. Fix anything it reports. It "
        "does not check novelty against the other models, nor convergence, nor "
        "speed: admission's full fit must have almost no divergent transitions, "
        f"R-hat <= {MAX_R_HAT} and bulk ESS >= {MIN_BULK_ESS}, and each of its "
        f"sampling runs must finish within {CANDIDATE_FIT_TIME_LIMIT_SEC / 60:g} "
        "minutes (a model still sampling then is stopped and rejected as too slow). "
        "A fit that narrowly fails is already refit once with smaller NUTS steps "
        f"(target_accept {ESCALATED_TARGET_ACCEPT:g}), and one far from converging "
        "is not refit at all, so asking for smaller steps is not a fix: prefer "
        "smooth, well-identified parameterisations (non-centred hierarchical or "
        "scale parameters, priors that constrain every parameter, no parameters "
        "that trade off against each other, no hard thresholds in the likelihood) "
        "and a likelihood vectorised over trials.",
        "",
        "```bash",
        check_candidate_command(candidate_dir, responses_path),
        "```",
        "",
    ]
    if refining:
        lines += [
            "`existing_hypotheses.md` lists the hypotheses already in the model set and",
            "how well each fits. This is a refinement slot — `CANDIDATE_BRIEF.md` says",
            "which model you improve — so your model is a better version of one already",
            "proposed, under a name not already taken.",
            "",
            "`refinement_menu.md` lists the models you may draw on — the live models",
            "other than the incumbent and the models pruned earlier — with their full",
            "hypotheses, their standing or the margin by which they lost, and their",
            "source files. Read the source of the model you refine.",
        ]
    else:
        lines += [
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
                "in the set). Do not re-propose any of them unchanged or as a",
                "near-duplicate; a pruned mechanism may come back only with a",
                "substantive change.",
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
    menu_text: Optional[str] = None
    if refining:
        menu_text = _write_refinement_menu(
            candidate_dir, models_dir, comparison, ledger, incumbent=incumbent
        )
    elif ledger is not None:
        # A name passed as "live" is simply left out of the retired list.
        attempted_text = ledger.render_markdown(
            live_names=[*_manifest_names(models_dir), *omit_from_attempted]
        )
        (candidate_dir / "attempted_hypotheses.md").write_text(
            attempted_text, encoding="utf-8"
        )
    if attempt_note is not None:
        (candidate_dir / "ATTEMPT_NOTE.md").write_text(attempt_note, encoding="utf-8")
    critiques_text: Optional[str] = None
    if critique_path is not None and critique_path.exists():
        critiques_text = critique_path.read_text(encoding="utf-8")
        (candidate_dir / "critiques.md").write_text(critiques_text, encoding="utf-8")

    critique_note = (
        "\nIf `critiques.md` is present, prioritise a hypothesis that addresses one of "
        "the significant discrepancies it reports.\n"
        if critique_path is not None
        else ""
    )
    if role == SLOT_REFINE_INCUMBENT:
        brief = _incumbent_brief(models_dir, comparison, incumbent, critique_note)
    elif role == SLOT_REFINE_CHOSEN:
        brief = _chosen_brief(incumbent, critique_note)
    else:
        hints = list(hints) if hints is not None else list(DEFAULT_CANDIDATE_HINTS)
        if lens_index is None:
            lens_index = candidate_idx % len(hints)
        brief = (
            "# Candidate Brief\n\n"
            f"{hints[lens_index]}\n\n"
            f"{_ONE_HYPOTHESIS_RULE}"
            f"{critique_note}"
        )
    (candidate_dir / "CANDIDATE_BRIEF.md").write_text(brief, encoding="utf-8")

    return {
        "context": context_text,
        "brief": brief,
        "existing_hypotheses": hypotheses_text,
        "attempted": attempted_text,
        "menu": menu_text,
        "critiques": critiques_text,
        "attempt_note": attempt_note,
    }


def _retry_note(previous_dir: Path) -> str:
    """The note a slot's retry opens with after an attempt that wrote no ``candidate.py``."""
    return (
        "NOTE: this is the second attempt at this candidate slot. The first "
        f"attempt, in `{previous_dir}`, ended without writing `candidate.py`. If "
        "that happens again the slot is lost for this round, so write "
        "`hypothesis.md`, `model_name.txt` and `candidate.py` to the paths above "
        "before anything else.\n"
    )


def _repair_note(previous_dir: Path, reason: str) -> str:
    """The note a slot's repair opens with: the rejection reason, verbatim."""
    if not reason.strip():
        raise ValueError("a repair attempt needs the rejection reason; got an empty one")
    return (
        "NOTE: this is a repair attempt. Your previous attempt at this candidate "
        f"slot, in `{previous_dir}`, was rejected at admission for this reason:\n\n"
        f"    {reason}\n\n"
        "Its `hypothesis.md`, `model_name.txt` and `candidate.py` have been copied "
        "into your working directory as a starting point. Fix what the reason "
        "describes — a different mechanism if the model predicts like an existing "
        "one, corrected code if it does not load, fit or score, a cheaper model if "
        "it was too slow to fit — and write all "
        "three files again to the paths above. A second rejection is final: there "
        "is no further attempt at this slot.\n"
    )


def _build_candidate_prompt(
    candidate_dir: Path, docs: Dict[str, Optional[str]]
) -> str:
    """The candidate agent's full prompt: task instructions + injected context.

    Every context document is inlined as a delimited section so the agent
    cannot skip the round brief, the current hypotheses, or the critique. The
    same documents exist as files in the working directory for reference. A
    retry or repair attempt's note (``docs["attempt_note"]``) goes right after
    the output instructions, before any other document. An exploratory slot
    carries ``attempted_hypotheses.md``; a refinement slot carries
    ``refinement_menu.md`` instead.
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
    if docs.get("attempt_note"):
        sections.insert(3, docs["attempt_note"])
    if docs.get("attempted"):
        sections.append(f"## attempted_hypotheses.md\n\n{docs['attempted']}")
    if docs.get("menu"):
        sections.append(f"## refinement_menu.md\n\n{docs['menu']}")
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
    notes_dir: Optional[Path] = None,
) -> bool:
    from src.runtime.coding_agent import run_coding_agent

    # Run from agent_root (the scrubbed agent tree), not from the harness
    # checkout. opencode roots its session at the cwd (the launcher pins PWD
    # to it — see src/runtime/coding_agent.py): paths under the session
    # directory are internal, and its opencode.json is the permission config
    # in force, so cwd must be a tree that has .here and opencode.json. The
    # agent tree is scrubbed of feature code, research library modules and
    # GT-recipe files, so the agent cannot read them. The candidate_dir is
    # named explicitly since it is not the cwd; the launcher grants any
    # allowed_dirs entry that falls outside the cwd.
    cwd = agent_root if agent_root is not None else REPO_ROOT
    prompt = _build_candidate_prompt(candidate_dir, docs)
    log_path = candidate_dir / "agent.jsonl"
    success, _ = run_coding_agent(
        prompt,
        cwd=cwd,
        log_path=log_path,
        allowed_dirs=[candidate_dir, models_dir, responses_path.parent],
        writable_dirs=[candidate_dir],  # the zoo and the data are read-only
        timeout_secs=agent_timeout_sec,
        backend=backend,
        model=agent_model,
        usage_label="inner:candidate",
        stock=True,  # a subject of the experiment: none of the user's Claude setup
        memory_dir=notes_dir,  # notes shared with later agents of this run only
        sandbox=True,  # sees only its own tree, scratch (/tmp) and a private home
    )
    return success
