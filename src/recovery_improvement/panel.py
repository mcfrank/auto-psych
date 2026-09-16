"""A review panel: several coding agents review both model-discovery loops and
discuss, in written rounds, how to improve them.

The CLIs (Claude Code, Codex, opencode) cannot talk to each other live, so
the panel is a file-based discussion under ``<panel root>/``:

    round1/<member>.md        independent reviews (no one has seen the others)
    round2..N/<member>.md     each member reads the whole thread and responds
    thread.md                 every note so far, rebuilt after each round
    synthesis/plan.md         the moderator's consensus plan (+ optional
                              next_run.env, the sweep that would test item 1)

Each round is one Slurm job (``panel_round.sbatch``) that runs the members
one after another and skips any whose note already exists, so an
interrupted round (a subscription session limit, a dead job) resumes
where it stopped. Stages are chained ``afterany``; a stage whose inputs are
not all present requeues itself for later instead of running on a partial
thread.

Members are ``name:backend:model:lens``. A lens is the perspective the
member is asked to review from (:data:`LENSES`); the moderator is one of the
members and writes the synthesis with the same backend/model.
"""

from __future__ import annotations

import csv
import re
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from src.recovery_improvement.campaign import Campaign, parse_env_file
from src.recovery_improvement.digest import build_digest
from src.recovery_improvement.iteration import (
    PRESCRIPTION_NAME,
    RunAgent,
    git,
)
from src.recovery_improvement.next_run import describe_allowed_keys, parse_next_run
from src.recovery_improvement.session_limit import SessionLimitHit, detect_session_limit

SYNTHESIS = "synthesis"
PLAN_NAME = "plan.md"
# Backends whose sessions cannot run commands or read files on this cluster
# (Codex: its bubblewrap sandbox needs net/uts namespaces that Sherlock caps at
# 0, and the workspace policy forbids bypassing the sandbox). Such members get
# an inlined evidence pack and return their note as their final message.
DEFAULT_PROMPT_ONLY_BACKENDS = ("codex",)
DEFAULT_EVIDENCE_MAX_BYTES = 300_000
NEXT_RUN_NAME = "next_run.env"
THREAD_NAME = "thread.md"
REPO_DIRNAME = "repo"
BACKENDS = ("claude", "codex", "opencode")

LENSES: dict[str, tuple[str, str]] = {
    "methods": (
        "Statistical methodology",
        "Judge the loop as a statistician: model comparison (ELPD-LOO, PSIS reliability, "
        "stacking weights vs. softmax posteriors), the EIG design and its model prior, "
        "identifiability of the model families, sampler settings and diagnostics, how "
        "recovery is measured (the held-out pool, Pearson r vs. RMSE, test-retest), and "
        "whether the numbers the loop acts on are trustworthy. Prefer changes with a "
        "principled justification and say what would falsify them.",
    ),
    "search": (
        "Agentic search",
        "Judge the loop as a search process over hypotheses: what the candidate agents are "
        "told (prompts, context, critiques, lenses), how candidates are admitted (novelty "
        "gate), pruned and carried forward, how the critique steers the next round, how the "
        "budget (rounds x candidates x experiments) is spent, and where the search stalls "
        "or wastes slots. Read the actual candidate models and transcripts, not just the "
        "summaries.",
    ),
    "crossloop": (
        "Cross-loop comparison",
        "Compare the two loops mechanism by mechanism: the auto-psych loop (PyMC models of "
        "binary choices, ELPD-LOO comparison, EIG-designed experiments, CriticAL posterior-"
        "predictive critique) and the verbal-protocol loop (search-policy models of Game-of-24 "
        "traces, exact sequential likelihood, canonical zoo with duplicate rejection, a critic "
        "that proposes and evaluates statistics, an explicit failure policy). For each "
        "difference say which side has the better design and what the other should borrow, "
        "with evidence from both repos' results. Review the verbal-protocol loop's own "
        "results for weaknesses in their own right, too.",
    ),
    "systems": (
        "Systems and robustness",
        "Judge the loop as a long-running system: failure modes seen in the logs (timeouts, "
        "crashes, unreliable fits, wasted agent calls), cost per recovered model, "
        "reproducibility, what is silently lost between experiments, and what "
        "instrumentation is missing to diagnose the next problem.",
    ),
}


@dataclass(frozen=True)
class Member:
    name: str
    backend: str
    model: str
    lens: str

    @property
    def lens_title(self) -> str:
        return LENSES[self.lens][0]

    @property
    def lens_text(self) -> str:
        return LENSES[self.lens][1]


def parse_members(spec: str) -> list[Member]:
    """``"name:backend:model:lens ..."`` -> members; every field validated."""
    members: list[Member] = []
    for token in spec.split():
        parts = token.split(":")
        if len(parts) != 4:
            raise ValueError(f"member {token!r}: expected name:backend:model:lens")
        name, backend, model, lens = parts
        if not name.isidentifier():
            raise ValueError(f"member name {name!r} must be an identifier")
        if backend not in BACKENDS:
            raise ValueError(f"member {name!r}: backend {backend!r} not in {BACKENDS}")
        if lens not in LENSES:
            raise ValueError(f"member {name!r}: lens {lens!r} not in {sorted(LENSES)}")
        if any(m.name == name for m in members):
            raise ValueError(f"duplicate member name {name!r}")
        members.append(Member(name, backend, model, lens))
    if not members:
        raise ValueError("MEMBERS names no panel members")
    return members


@dataclass(frozen=True)
class Panel:
    root: Path
    name: str
    source_repo: Path
    verbal_repo: Optional[Path]
    campaign_root: Optional[Path]
    baseline_roots: tuple[Path, ...]
    members: tuple[Member, ...]
    n_discussion_rounds: int
    moderator: Member
    member_max_turns: int
    member_timeout_sec: int
    member_max_budget_usd: float
    slurm: dict[str, str]
    driver_sbatch: Path
    prompt_only_backends: tuple[str, ...] = DEFAULT_PROMPT_ONLY_BACKENDS
    evidence_files: tuple[Path, ...] = ()
    """Files inlined into prompt-only members' briefs (they cannot open files)."""
    evidence_max_bytes: int = DEFAULT_EVIDENCE_MAX_BYTES

    def is_prompt_only(self, member: Member) -> bool:
        return member.backend in self.prompt_only_backends

    @classmethod
    def load(cls, root: Path | str) -> "Panel":
        root = Path(root)
        env_path = root / "panel.env"
        if not env_path.is_file():
            raise FileNotFoundError(f"No panel.env under {root}")
        v = parse_env_file(env_path)
        required = ("PANEL_NAME", "SOURCE_REPO", "MEMBERS", "N_DISCUSSION_ROUNDS", "MODERATOR",
                    "MEMBER_MAX_TURNS", "MEMBER_TIMEOUT_SEC", "MEMBER_MAX_BUDGET_USD")
        missing = [k for k in required if not v.get(k)]
        if missing:
            raise ValueError(f"{env_path} is missing required keys: {missing}")
        members = tuple(parse_members(v["MEMBERS"]))
        moderators = [m for m in members if m.name == v["MODERATOR"]]
        if not moderators:
            raise ValueError(f"MODERATOR {v['MODERATOR']!r} is not a member")
        source_repo = Path(v["SOURCE_REPO"])
        return cls(
            root=root,
            name=v["PANEL_NAME"],
            source_repo=source_repo,
            verbal_repo=Path(v["VERBAL_REPO"]) if v.get("VERBAL_REPO") else None,
            campaign_root=Path(v["CAMPAIGN_ROOT"]) if v.get("CAMPAIGN_ROOT") else None,
            baseline_roots=tuple(Path(p) for p in v.get("BASELINE_ROOTS", "").split()),
            members=members,
            n_discussion_rounds=int(v["N_DISCUSSION_ROUNDS"]),
            moderator=moderators[0],
            member_max_turns=int(v["MEMBER_MAX_TURNS"]),
            member_timeout_sec=int(v["MEMBER_TIMEOUT_SEC"]),
            member_max_budget_usd=float(v["MEMBER_MAX_BUDGET_USD"]),
            slurm={
                "PANEL_PARTITION": v.get("PANEL_PARTITION") or "normal",
                "PANEL_TIME": v.get("PANEL_TIME") or "10:00:00",
                "PANEL_CPUS": v.get("PANEL_CPUS") or "4",
                "PANEL_MEM": v.get("PANEL_MEM") or "16GB",
            },
            driver_sbatch=Path(
                v.get("DRIVER_SBATCH")
                or source_repo / "scripts" / "recovery_improvement" / "panel_round.sbatch"
            ),
            prompt_only_backends=tuple(
                v["PROMPT_ONLY_BACKENDS"].split()
                if "PROMPT_ONLY_BACKENDS" in v else DEFAULT_PROMPT_ONLY_BACKENDS
            ),
            evidence_files=tuple(
                (source_repo / f if not Path(f).is_absolute() else Path(f))
                for f in v.get("EVIDENCE_FILES", "").split()
            ),
            evidence_max_bytes=int(v.get("EVIDENCE_MAX_BYTES") or DEFAULT_EVIDENCE_MAX_BYTES),
        )

    @property
    def total_rounds(self) -> int:
        return 1 + self.n_discussion_rounds

    def round_dir(self, k: int) -> Path:
        return self.root / f"round{k}"

    def note_path(self, k: int, member: Member) -> Path:
        return self.round_dir(k) / f"{member.name}.md"

    @property
    def synthesis_dir(self) -> Path:
        return self.root / SYNTHESIS

    @property
    def plan_path(self) -> Path:
        return self.synthesis_dir / PLAN_NAME

    @property
    def repo(self) -> Path:
        return self.root / REPO_DIRNAME

    def scratch_dir(self, member: Member) -> Path:
        return self.root / "scratch" / member.name

    @property
    def journal_path(self) -> Path:
        return self.root / "journal.md"

    @property
    def stop_path(self) -> Path:
        return self.root / "STOP"

    def stages(self) -> list[str]:
        return [str(k) for k in range(1, self.total_rounds + 1)] + [SYNTHESIS]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def append_journal(panel: Panel, text: str) -> None:
    with panel.journal_path.open("a", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n\n")


# --- inputs --------------------------------------------------------------------


def prepare_repo(panel: Panel) -> Path:
    """A read-only clone of the auto-psych checkout at HEAD for the members."""
    if (panel.repo / ".git").exists():
        return panel.repo
    panel.root.mkdir(parents=True, exist_ok=True)
    git(panel.root, "clone", "--quiet", "--", str(panel.source_repo), str(panel.repo))
    return panel.repo


def verbal_digest(verbal_repo: Optional[Path]) -> str:
    """Per study of the verbal-protocol loop: seed vs. final-round KL to the
    ground truth (mean over replicates), from ``summary/recovery.csv``."""
    if verbal_repo is None:
        return "(no verbal-protocol repo configured)"
    if not verbal_repo.is_dir():
        return f"(verbal-protocol repo not found at {verbal_repo})"
    studies_root = verbal_repo / "data" / "cog-models" / "agentic-model-recovery"
    lines = [f"Repo: `{verbal_repo}`", ""]
    docs = sorted((verbal_repo / ".claude" / "docs").glob("*.md"))
    if docs:
        lines.append("Prior analyses (read these): " + ", ".join(f"`{d.relative_to(verbal_repo)}`" for d in docs))
    spec = verbal_repo / "src" / "cog_models" / "disco_loop" / "FUNCTIONAL_SPEC.md"
    if spec.is_file():
        lines.append(f"Functional spec: `{spec.relative_to(verbal_repo)}`")
    lines.append("")
    if not studies_root.is_dir():
        lines.append(f"(no agentic-recovery studies under {studies_root})")
        return "\n".join(lines)
    lines.append("Agentic-recovery studies (`summary/recovery.csv`; KL to the ground truth in "
                 "nats/action, mean over replicates; seed = round -1 where the study recorded it, "
                 "final = last round). Newer studies report `kl_symmetric_nats`, older ones "
                 "`kl_per_action_nats` — the metric column is shown per row, so compare within a "
                 "study, not across metrics:")
    lines.append("")
    lines.append("| study | ground truth | n reps | metric | seed KL | first-round KL | final KL | final round |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for study in sorted(p for p in studies_root.iterdir() if p.is_dir()):
        summary = study / "summary" / "recovery.csv"
        if not summary.is_file():
            lines.append(f"| {study.name} | (no summary/recovery.csv — unsummarised) | | | | | | |")
            continue
        with summary.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            columns = reader.fieldnames or []
        metric = next((c for c in ("kl_symmetric_nats", "kl_per_action_nats") if c in columns), None)
        if metric is None or not {"model", "replicate", "round"} <= set(columns):
            lines.append(f"| {study.name} | (recovery.csv has no KL column I know; columns: {', '.join(columns)}) | | | | | | |")
            continue
        by_model: dict[str, dict[str, dict[int, float]]] = {}
        for row in rows:
            try:
                kl = float(row[metric])
            except ValueError:
                continue
            by_model.setdefault(row["model"], {}).setdefault(row["replicate"], {})[int(row["round"])] = kl

        def _mean(values: list[float]) -> str:
            return f"{sum(values) / len(values):.2f}" if values else "n/a"

        for model, reps in sorted(by_model.items()):
            seeds = [r[-1] for r in reps.values() if -1 in r]
            firsts = [r[0] for r in reps.values() if 0 in r]
            finals = [r[max(r)] for r in reps.values() if r]
            last_round = max(max(r) for r in reps.values())
            lines.append(
                f"| {study.name} | {model} | {len(reps)} | {metric} | {_mean(seeds)} | "
                f"{_mean(firsts)} | {_mean(finals)} | {last_round} |"
            )
    lines.append("")
    lines.append("Per-run trees: `<study>/<run>/` with `iter_<k>/` (critic + theorist + candidates), "
                 "`model_zoo/`, `final_comparison.md`, `best_model.py`, `token_usage.jsonl`.")
    return "\n".join(lines)


def campaign_prescriptions(campaign_root: Optional[Path]) -> str:
    if campaign_root is None or not campaign_root.is_dir():
        return "(no campaign configured)"
    parts = []
    for iter_dir in sorted(campaign_root.glob("iter[0-9]*"), key=lambda p: int(p.name[4:])):
        path = iter_dir / PRESCRIPTION_NAME
        if path.is_file():
            parts.append(f"### {iter_dir.name} ({path})\n\n{path.read_text(encoding='utf-8').strip()}")
    journal = campaign_root / "journal.md"
    if journal.is_file():
        parts.append(f"### campaign journal ({journal})\n\n{journal.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(parts) if parts else f"(campaign {campaign_root} has no prescriptions yet)"


def sweep_defaults(campaign_root: Optional[Path]) -> dict[str, str]:
    if campaign_root is None or not (campaign_root / "campaign.env").is_file():
        return {}
    return Campaign.load(campaign_root).sweep_defaults


def build_digests(panel: Panel) -> dict[str, str]:
    """The two digests every brief carries (written once, reused per stage)."""
    auto_path = panel.root / "digest_autopsych.md"
    verbal_path = panel.root / "digest_verbal.md"
    if not auto_path.is_file():
        roots = [(r.name, r) for r in panel.baseline_roots]
        if panel.campaign_root is not None:
            for iter_dir in sorted(panel.campaign_root.glob("iter[0-9]*"), key=lambda p: int(p.name[4:])):
                if (iter_dir / "sweep").is_dir():
                    roots.append((f"{panel.campaign_root.name}/{iter_dir.name}", iter_dir / "sweep"))
        auto_path.write_text(
            build_digest(roots, primary_label=roots[0][0]) if roots else "(no sweep roots configured)",
            encoding="utf-8",
        )
    if not verbal_path.is_file():
        verbal_path.write_text(verbal_digest(panel.verbal_repo), encoding="utf-8")
    return {
        "digest_autopsych": auto_path.read_text(encoding="utf-8"),
        "digest_verbal": verbal_path.read_text(encoding="utf-8"),
    }


def evidence_pack(panel: Panel) -> str:
    """The configured evidence files, inlined with headers, within the byte cap.
    A missing file is reported in the pack rather than skipped silently."""
    if not panel.evidence_files:
        return "(no evidence files configured)"
    parts: list[str] = []
    used = 0
    for path in panel.evidence_files:
        if not path.is_file():
            parts.append(f"### {path}\n\n(missing on disk)")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        room = panel.evidence_max_bytes - used
        if room <= 0:
            parts.append(f"### {path}\n\n(omitted: evidence pack is at its {panel.evidence_max_bytes}-byte cap)")
            continue
        if len(text) > room:
            text = text[:room] + f"\n\n[... truncated at the {panel.evidence_max_bytes}-byte cap ...]"
        used += len(text)
        parts.append(f"### {path}\n\n```\n{text}\n```")
    return "\n\n".join(parts)


def capabilities_text(panel: Panel, member: Member) -> str:
    if panel.is_prompt_only(member):
        return (
            "**You cannot run commands or open files in this session** (the sandbox this CLI "
            "needs is unavailable on this cluster). Everything you can use is in this brief: "
            "the digests, the campaign's prescriptions, the evidence pack of source files and "
            "reports at the end, and the thread. Reason carefully from those, quote the exact "
            "lines you rely on, say explicitly which claims you could not verify, and name the "
            "file and check you want a file-capable member to run. **Your final message is your "
            "note**: write it in full, in the deliverable format, as the last thing you say."
        )
    return (
        "You can run commands and read files. Write your note to the path given below; the "
        "wrapper validates that it exists."
    )


# --- the thread ------------------------------------------------------------------


def build_thread(panel: Panel, upto_round: int) -> str:
    """Every note of rounds 1..upto_round, in order, with headers."""
    parts = []
    for k in range(1, upto_round + 1):
        for member in panel.members:
            path = panel.note_path(k, member)
            if path.is_file():
                body = path.read_text(encoding="utf-8").strip()
                parts.append(f"### Round {k} — {member.name} ({member.lens_title}, {member.backend}/{member.model})\n\n{body}")
    return "\n\n---\n\n".join(parts) if parts else "(no notes yet)"


def write_thread(panel: Panel, upto_round: int) -> Path:
    path = panel.root / THREAD_NAME
    path.write_text(
        f"# Panel `{panel.name}` — discussion thread (rounds 1..{upto_round})\n\n"
        + build_thread(panel, upto_round),
        encoding="utf-8",
    )
    return path


def missing_inputs(panel: Panel, stage: str) -> list[str]:
    """Notes that must exist before ``stage`` (a round number or 'synthesis')."""
    upto = panel.total_rounds if stage == SYNTHESIS else int(stage) - 1
    return [
        str(panel.note_path(k, m))
        for k in range(1, upto + 1)
        for m in panel.members
        if not panel.note_path(k, m).is_file()
    ]


# --- briefs ----------------------------------------------------------------------


def _round_instructions(panel: Panel, k: int) -> str:
    if k == 1:
        return (
            "This is the **independent round**: nobody has written anything yet. Form your own "
            "view from the evidence. Do not guess what the others will say."
        )
    return (
        f"This is **discussion round {k} of {panel.total_rounds}**. The thread below holds every note "
        "so far. Respond to the other members by name: where you agree, say what evidence "
        "convinced you; where you disagree, say why and cite the file that shows it; where a "
        "proposal is under-specified, specify it. Revise your own ranking in light of the "
        "discussion — changing your mind for a good reason is the point. Keep new material to "
        "what the discussion needs; do not repeat your round-1 note."
    )


def _substitute(template: str, mapping: dict[str, str]) -> str:
    try:
        return string.Template(template).substitute(mapping)
    except KeyError as exc:
        raise ValueError(f"brief template uses an unknown placeholder: {exc}") from exc


def compose_member_brief(
    template: str,
    panel: Panel,
    member: Member,
    round_k: int,
    *,
    digests: dict[str, str],
    venv_py: str = "python",
    repair_feedback: str = "",
) -> str:
    others = ", ".join(
        f"{m.name} ({m.lens_title}, {m.backend})" for m in panel.members if m.name != member.name
    ) or "(none)"
    mapping = {
        "panel_name": panel.name,
        "round": str(round_k),
        "total_rounds": str(panel.total_rounds),
        "member": member.name,
        "backend": member.backend,
        "model": member.model,
        "lens_title": member.lens_title,
        "lens_text": member.lens_text,
        "others": others,
        "repo": str(panel.repo),
        "verbal_repo": str(panel.verbal_repo) if panel.verbal_repo else "(none)",
        "panel_root": str(panel.root),
        "scratch_dir": str(panel.scratch_dir(member)),
        "note_path": str(panel.note_path(round_k, member)),
        "venv_py": venv_py,
        "round_instructions": _round_instructions(panel, round_k),
        "thread": build_thread(panel, round_k - 1) if round_k > 1 else "(none — independent round)",
        "prescriptions": campaign_prescriptions(panel.campaign_root),
        "repair_feedback": repair_feedback,
        "capabilities": capabilities_text(panel, member),
        "evidence": evidence_pack(panel) if panel.is_prompt_only(member) else
                    "(not inlined: you can open the files yourself)",
        **digests,
    }
    return _substitute(template, mapping)


def compose_synthesis_brief(
    template: str, panel: Panel, *, venv_py: str = "python", repair_feedback: str = ""
) -> str:
    defaults = sweep_defaults(panel.campaign_root)
    mapping = {
        "panel_name": panel.name,
        "total_rounds": str(panel.total_rounds),
        "moderator": panel.moderator.name,
        "members_list": "\n".join(
            f"- {m.name}: {m.lens_title} ({m.backend}/{m.model})" for m in panel.members
        ),
        "thread": build_thread(panel, panel.total_rounds),
        "plan_path": str(panel.plan_path),
        "next_run_path": str(panel.synthesis_dir / NEXT_RUN_NAME),
        "allowed_keys": describe_allowed_keys(),
        "sweep_defaults": "\n".join(f"{k}={v}" for k, v in defaults.items()) or "(no campaign: no sweep defaults)",
        "repo": str(panel.repo),
        "verbal_repo": str(panel.verbal_repo) if panel.verbal_repo else "(none)",
        "panel_root": str(panel.root),
        "venv_py": venv_py,
        "repair_feedback": repair_feedback,
        "capabilities": capabilities_text(panel, panel.moderator),
    }
    return _substitute(template, mapping)


# --- running a stage -------------------------------------------------------------

AgentFactory = Callable[[Member, str], RunAgent]
"""``agent_for(member, label) -> run_agent`` (label names the stage for logs)."""


def partial_note_path(note_path: Path) -> Path:
    return note_path.with_name(note_path.stem + ".partial.md")


def _raise_if_limit(panel: Panel, member: Member, result_text: str, note_path: Path) -> None:
    """On a session limit, set aside whatever the member had written of its
    note (so the resumed session finishes it rather than being skipped as
    done) and pause the stage."""
    limit = detect_session_limit(result_text)
    if limit is None:
        return
    if note_path.is_file():
        note_path.rename(partial_note_path(note_path))
    append_journal(panel, f"- {member.name}: session limit hit ({limit.message}); stage paused")
    raise SessionLimitHit(limit)


def _resume_feedback(note_path: Path) -> str:
    partial = partial_note_path(note_path)
    if not partial.is_file():
        return ""
    return (
        "## RESUMING AFTER A SESSION LIMIT\n\n"
        f"A previous session of you was cut off by the subscription's session limit. What it had "
        f"written of its note is at `{partial}` — read it, finish it, and write the complete note "
        f"to `{note_path}`.\n"
    )


_NEXT_RUN_BLOCK = re.compile(r"```next_run\.env\s*\n(.*?)```", re.DOTALL)


def _deliver_from_message(panel: Panel, member: Member, text: str, note_path: Path) -> None:
    """A prompt-only member's deliverable is its final message: write it to the
    note path. For the synthesis, a fenced ```next_run.env block becomes the
    sweep spec file next to the plan."""
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(text.strip() + "\n", encoding="utf-8")
    if note_path == panel.plan_path:
        match = _NEXT_RUN_BLOCK.search(text)
        if match:
            (panel.synthesis_dir / NEXT_RUN_NAME).write_text(match.group(1).strip() + "\n", encoding="utf-8")


def _run_with_repair(
    panel: Panel,
    member: Member,
    label: str,
    compose: Callable[[str], str],
    deliverable_problems: Callable[[], list[str]],
    run_agent: RunAgent,
    *,
    cwd: Path,
    log_dir: Path,
    note_path: Path,
) -> None:
    """Run one session; if its deliverable is missing or invalid, run one repair
    session with the problems injected; then fail loudly. A prompt-only member
    cannot write files, so its final message is taken as the deliverable."""
    brief = compose("")
    (log_dir / f"brief_{label}.md").write_text(brief, encoding="utf-8")
    _, text = run_agent(brief, cwd=cwd, log_path=log_dir / f"stream_{label}.jsonl")
    _raise_if_limit(panel, member, text, note_path)
    if panel.is_prompt_only(member):
        _deliver_from_message(panel, member, text, note_path)
    problems = deliverable_problems()
    if problems:
        feedback = (
            "## REPAIR REQUIRED\n\nYour previous session ended without a valid deliverable:\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\n\nFix exactly this and finish.\n"
        )
        brief = compose(feedback)
        (log_dir / f"brief_{label}.repair.md").write_text(brief, encoding="utf-8")
        _, text = run_agent(brief, cwd=cwd, log_path=log_dir / f"stream_{label}.repair.jsonl")
        _raise_if_limit(panel, member, text, note_path)
        if panel.is_prompt_only(member):
            _deliver_from_message(panel, member, text, note_path)
        problems = deliverable_problems()
    if problems:
        raise RuntimeError(
            f"{label}: deliverable still invalid after a repair round:\n"
            + "\n".join(f"- {p}" for p in problems)
        )


def _note_problems(path: Path) -> list[str]:
    if not path.is_file() or len(path.read_text(encoding="utf-8").strip()) < 200:
        return [f"{path} is missing or too short (write your full note there)"]
    return []


def run_round(
    panel: Panel,
    k: int,
    *,
    agent_for: AgentFactory,
    member_template: str,
    venv_py: str = "python",
) -> list[str]:
    """Run every member of round ``k`` that has no note yet; rebuild the thread."""
    if panel.stop_path.exists():
        raise RuntimeError(f"panel STOP file present at {panel.stop_path}")
    missing = missing_inputs(panel, str(k))
    if missing:
        raise RuntimeError(f"round {k} cannot start; missing notes: {missing}")
    repo = prepare_repo(panel)
    digests = build_digests(panel)
    round_dir = panel.round_dir(k)
    round_dir.mkdir(parents=True, exist_ok=True)
    append_journal(panel, f"## Round {k} — {_timestamp()}")
    ran: list[str] = []
    for member in panel.members:
        note = panel.note_path(k, member)
        if note.is_file():
            append_journal(panel, f"- {member.name}: note already present, skipped")
            continue
        panel.scratch_dir(member).mkdir(parents=True, exist_ok=True)
        label = f"{member.name}_round{k}"
        resume_note = _resume_feedback(note)
        _run_with_repair(
            panel, member, label,
            compose=lambda fb, m=member, rn=resume_note: compose_member_brief(
                member_template, panel, m, k, digests=digests, venv_py=venv_py,
                repair_feedback=(rn + "\n" + fb).strip(),
            ),
            deliverable_problems=lambda n=note: _note_problems(n),
            run_agent=agent_for(member, label),
            cwd=repo, log_dir=round_dir, note_path=note,
        )
        append_journal(panel, f"- {member.name}: wrote `{note}`")
        ran.append(member.name)
    write_thread(panel, k)
    return ran


def _plan_problems(panel: Panel) -> list[str]:
    problems = _note_problems(panel.plan_path)
    next_run = panel.synthesis_dir / NEXT_RUN_NAME
    if next_run.is_file():
        try:
            parse_next_run(next_run, repo=panel.repo, defaults=sweep_defaults(panel.campaign_root))
        except ValueError as exc:
            problems.append(str(exc))
    return problems


def run_synthesis(
    panel: Panel, *, agent_for: AgentFactory, synthesis_template: str, venv_py: str = "python"
) -> Path:
    if panel.stop_path.exists():
        raise RuntimeError(f"panel STOP file present at {panel.stop_path}")
    missing = missing_inputs(panel, SYNTHESIS)
    if missing:
        raise RuntimeError(f"synthesis cannot start; missing notes: {missing}")
    if panel.plan_path.is_file():
        raise RuntimeError(f"{panel.plan_path} already exists; the panel has concluded")
    repo = prepare_repo(panel)
    panel.synthesis_dir.mkdir(parents=True, exist_ok=True)
    write_thread(panel, panel.total_rounds)
    append_journal(panel, f"## Synthesis — {_timestamp()} (moderator {panel.moderator.name})")
    resume_note = _resume_feedback(panel.plan_path)
    _run_with_repair(
        panel, panel.moderator, "synthesis",
        compose=lambda fb: compose_synthesis_brief(
            synthesis_template, panel, venv_py=venv_py, repair_feedback=(resume_note + "\n" + fb).strip()
        ),
        deliverable_problems=lambda: _plan_problems(panel),
        run_agent=agent_for(panel.moderator, "synthesis"),
        cwd=repo, log_dir=panel.synthesis_dir, note_path=panel.plan_path,
    )
    has_next = (panel.synthesis_dir / NEXT_RUN_NAME).is_file()
    append_journal(
        panel,
        f"- plan written: `{panel.plan_path}`"
        + (f"; proposed sweep: `{panel.synthesis_dir / NEXT_RUN_NAME}`" if has_next else "; no sweep proposed"),
    )
    return panel.plan_path
