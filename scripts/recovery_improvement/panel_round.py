"""CLI: run one stage of a review panel (a discussion round or the synthesis).

Called by ``panel_round.sbatch``. A stage whose inputs (the previous rounds'
notes) are not all present requeues itself for 20 minutes later rather than
running on a partial thread; a member that hits its subscription's session
limit pauses the stage, which requeues itself for just after the reset and
resumes with the members that still have no note.

Usage (inside the sbatch job):
    $VENV_PY scripts/recovery_improvement/panel_round.py --panel-root <root> --stage 1
    $VENV_PY scripts/recovery_improvement/panel_round.py --panel-root <root> --stage synthesis
    # briefs only, nothing run or submitted:
    ... --stage 1 --dry-run
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.recovery_improvement.iteration import RunAgent  # noqa: E402
from src.recovery_improvement.panel import (  # noqa: E402
    SYNTHESIS,
    Member,
    Panel,
    append_journal,
    build_digests,
    compose_member_brief,
    compose_synthesis_brief,
    missing_inputs,
    prepare_repo,
    run_round,
    run_synthesis,
)
from src.recovery_improvement.session_limit import SessionLimitHit, retry_begin_time  # noqa: E402
from src.recovery_improvement.slurm import REVIEW_DISALLOWED_TOOLS, scrubbed_environment  # noqa: E402
from src.runtime import token_usage  # noqa: E402
from src.runtime.coding_agent import run_coding_agent  # noqa: E402

PROMPT_DIR = here() / "scripts" / "recovery_improvement"
# Waiting for an earlier stage: 20-minute checks for the first two hours, then
# hourly, for up to a week — a weekly usage limit on one member must not make
# the stage behind it give up.
MAX_INPUT_WAITS = 6 + 24 * 7
MAX_LIMIT_WAITS = 12


def input_wait(waits_so_far: int) -> str:
    return "now+20minutes" if waits_so_far < 6 else "now+60minutes"


@dataclass
class Args:
    """Run one stage of a review panel."""

    panel_root: Path
    """Panel directory holding panel.env (written by panel.sh)."""
    stage: str
    """A round number (1..N) or 'synthesis'."""
    member_template: Path = PROMPT_DIR / "panel_member_prompt.md"
    synthesis_template: Path = PROMPT_DIR / "panel_synthesis_prompt.md"
    dry_run: bool = False
    """Write the briefs this stage would send to <root>/dryrun_<stage>/ and exit."""


def agent_factory(panel: Panel):
    """Real sessions: Claude with the campaign-style caps, Codex/opencode with
    the timeout only (they have no turn/budget flags)."""
    allowed = [panel.root, *panel.baseline_roots]
    if panel.verbal_repo:
        allowed.append(panel.verbal_repo)
    if panel.campaign_root:
        allowed.append(panel.campaign_root)

    def agent_for(member: Member, label: str) -> RunAgent:
        extra: list[str] = []
        if member.backend == "claude":
            extra = [
                "--max-turns", str(panel.member_max_turns),
                "--max-budget-usd", str(panel.member_max_budget_usd),
                "--disallowedTools", *REVIEW_DISALLOWED_TOOLS,
            ]

        def run(prompt: str, *, cwd: Path, log_path: Path) -> tuple[bool, str]:
            token_usage.start_usage_log(panel.root / "token_usage.jsonl")
            return run_coding_agent(
                prompt, cwd=cwd, log_path=log_path, allowed_dirs=allowed, model=member.model,
                timeout_secs=panel.member_timeout_sec, backend=member.backend,
                usage_label=f"panel:{panel.name}:{label}", extra_args=extra,
            )

        return run

    return agent_for


def requeue(panel: Panel, stage: str, *, begin, counter_name: str, cap: int, why: str) -> None:
    """``begin`` is an sbatch --begin value, or a callable of the wait count."""
    counter = panel.root / f"{counter_name}_{stage}"
    waits = int(counter.read_text()) + 1 if counter.is_file() else 1
    if waits > cap:
        raise SystemExit(f"stage {stage}: {why}; requeued {waits - 1} times already, giving up")
    counter.write_text(str(waits))
    if callable(begin):
        begin = begin(waits - 1)
    slurm = panel.slurm
    cmd = [
        "sbatch", "--parsable", f"--begin={begin}",
        f"--job-name=panel_{panel.name}", f"--partition={slurm['PANEL_PARTITION']}",
        f"--time={slurm['PANEL_TIME']}", f"--cpus-per-task={slurm['PANEL_CPUS']}",
        f"--mem={slurm['PANEL_MEM']}",
        f"--output={panel.root}/slurm_logs/stage{stage}_%j.out",
        f"--error={panel.root}/slurm_logs/stage{stage}_%j.out",
        f"--export=ALL,PANEL_ROOT={panel.root},STAGE={stage}",
        str(panel.driver_sbatch),
    ]
    proc = subprocess.run(cmd, env=scrubbed_environment(os.environ), text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"requeue failed: {' '.join(cmd)}\n{proc.stdout}")
    job_id = proc.stdout.strip().split(";")[0]
    append_journal(panel, f"- stage {stage}: {why}; requeued as job {job_id} to begin {begin} ({waits}/{cap})")
    print(f"stage {stage}: {why}; requeued as job {job_id} to begin {begin}")


def main(args: Args) -> None:
    panel = Panel.load(args.panel_root)
    venv_py = os.environ.get("VENV_PY") or sys.executable
    member_template = args.member_template.read_text(encoding="utf-8")
    synthesis_template = args.synthesis_template.read_text(encoding="utf-8")
    stage = args.stage
    if stage not in panel.stages():
        raise SystemExit(f"stage {stage!r} is not one of {panel.stages()}")

    if args.dry_run:
        out = panel.root / f"dryrun_{stage}"
        out.mkdir(parents=True, exist_ok=True)
        prepare_repo(panel)
        digests = build_digests(panel)
        if stage == SYNTHESIS:
            (out / "brief_synthesis.md").write_text(
                compose_synthesis_brief(synthesis_template, panel, venv_py=venv_py), encoding="utf-8")
        else:
            for m in panel.members:
                (out / f"brief_{m.name}.md").write_text(
                    compose_member_brief(member_template, panel, m, int(stage), digests=digests, venv_py=venv_py),
                    encoding="utf-8")
        print(f"dry run: briefs in {out}")
        return

    if panel.stop_path.exists():
        print(f"{panel.stop_path} present; exiting")
        return
    missing = missing_inputs(panel, stage)
    if missing:
        requeue(panel, stage, begin=input_wait, counter_name="input_waits", cap=MAX_INPUT_WAITS,
                why=f"waiting for {len(missing)} earlier note(s) (first: {missing[0]})")
        return

    try:
        if stage == SYNTHESIS:
            out = run_synthesis(panel, agent_for=agent_factory(panel),
                                synthesis_template=synthesis_template, venv_py=venv_py)
            print(f"synthesis done: {out}")
        else:
            ran = run_round(panel, int(stage), agent_for=agent_factory(panel),
                            member_template=member_template, venv_py=venv_py)
            print(f"round {stage} done; ran {ran or 'nobody (all notes present)'}")
    except SessionLimitHit as hit:
        requeue(panel, stage, begin=retry_begin_time(hit.limit), counter_name="limit_waits",
                cap=MAX_LIMIT_WAITS, why=f"session limit ({hit.limit.message})")


if __name__ == "__main__":
    main(tyro.cli(Args))
