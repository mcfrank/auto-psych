"""CLI: run one iteration of a recovery-improvement campaign.

Called by ``review_iteration.sbatch`` on a compute node. In ``review`` mode it
digests the sweeps to review, runs the Claude review agent, validates its
deliverables, launches the declared sweep and chains the next review job. In
``finalize`` mode (after the last iteration's sweep) it only writes the final
digest and closes the journal.

Usage (inside the sbatch job):
    $VENV_PY scripts/recovery_improvement/run_iteration.py \\
        --campaign-root $SCRATCH/auto-psych/recovery_improvement/<name> --iteration 1

    # Inspect what iteration N *would* see (digest + prompt), without spawning
    # Claude or submitting anything:
    ... --iteration N --dry-run
"""

from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.recovery_improvement.campaign import Campaign  # noqa: E402
from src.recovery_improvement.digest import build_digest  # noqa: E402
from src.recovery_improvement.iteration import (  # noqa: E402
    DIGEST_NAME,
    MODE_FINALIZE,
    MODE_REVIEW,
    PROMPT_NAME,
    REPO_DIRNAME,
    compose_prompt,
    finalize_campaign,
    primary_label,
    roots_to_review,
    run_iteration,
)
from src.recovery_improvement.slurm import (  # noqa: E402
    make_run_agent,
    submit_review,
    submit_sweep,
)

DEFAULT_PROMPT_TEMPLATE = here() / "scripts" / "recovery_improvement" / "review_prompt.md"


@dataclass
class Args:
    """Run one review (or the finalize step) of a recovery-improvement campaign."""

    campaign_root: Path
    """Campaign directory holding campaign.env (written by start_campaign.sh)."""
    iteration: int
    """Which iteration this is (1-based)."""
    mode: Literal["review", "finalize"] = MODE_REVIEW
    """review: agent + sweep + chain; finalize: final digest only."""
    prompt_template: Path = DEFAULT_PROMPT_TEMPLATE
    """The review agent's brief ($placeholder template)."""
    dry_run: bool = False
    """Write the digest and prompt to <campaign>/dryrun_iter<N>/ and exit —
    no repo clone, no agent, no Slurm submission."""


def main(args: Args) -> None:
    campaign = Campaign.load(args.campaign_root)
    venv_py = os.environ.get("VENV_PY") or sys.executable

    if args.mode == MODE_FINALIZE:
        out = finalize_campaign(campaign, args.iteration)
        print(f"Campaign {campaign.name} finalized; digest at {out}")
        return

    template = args.prompt_template.read_text(encoding="utf-8")
    if args.dry_run:
        out_dir = campaign.root / f"dryrun_iter{args.iteration}"
        out_dir.mkdir(parents=True, exist_ok=True)
        roots = roots_to_review(campaign, args.iteration)
        digest = build_digest(roots, primary_label=primary_label(campaign, args.iteration))
        (out_dir / DIGEST_NAME).write_text(digest, encoding="utf-8")
        prompt = compose_prompt(
            template, campaign=campaign, iteration=args.iteration,
            repo=campaign.iteration_dir(args.iteration) / REPO_DIRNAME,
            digest=digest, venv_py=venv_py,
        )
        (out_dir / PROMPT_NAME).write_text(prompt, encoding="utf-8")
        print(f"dry run: wrote {out_dir / DIGEST_NAME} and {out_dir / PROMPT_NAME}")
        return

    result = run_iteration(
        campaign,
        args.iteration,
        run_agent=make_run_agent(campaign, args.iteration),
        submit_sweep=submit_sweep,
        submit_review=functools.partial(submit_review, campaign),
        prompt_template=template,
        venv_py=venv_py,
    )
    print(f"\nIteration {result.iteration} of campaign {campaign.name}: decision={result.decision}")
    if result.sweep_jobs:
        print(
            f"  sweep jobs: setup {result.sweep_jobs.setup_id}, array {result.sweep_jobs.array_id}, "
            f"analysis {result.sweep_jobs.analysis_id}"
        )
        print(f"  next job: {result.next_review_job}")
    else:
        print(f"  stopped: {result.stop_reason}")


if __name__ == "__main__":
    main(tyro.cli(Args))
