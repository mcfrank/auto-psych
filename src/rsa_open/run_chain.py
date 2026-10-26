"""Run one chain of the open campaign, on simulated participants (the rehearsals) or live
(``--collection live``: `src.rsa_open.live`).

    <agent python> -m src.rsa_open.run_chain --chain 0 --run-dir <agent tree>/_runs/chain_0 \
        --private-dir <private dir with ground_truth.json> --agent-root <agent tree>

The designer (Gemini 4) and the theorists (Gemini 3.8 Flash) are sandboxed opencode agents with
no network (`src.rsa.loop.no_network`); the pilot's participants and its legibility checkers are
the harness's own Gemini calls. The chain starts from ``seeds/chain_<k>`` and designs against
``seeds/bar`` (`src.rsa_open.seeds`); its simulated people are ``--private-dir``'s ground truth
(`src.rsa_open.ground_truth`), which no agent may read. Re-running resumes: each stage whose
output exists is skipped (`src.rsa_open.chain`).

Live, ``--private-dir`` is the chain's private directory (raw responses with Prolific ids) and
``--campaign-dir`` the campaign's (the participant-id map, the recruiting slots). The exit code
says how a chain stopped short of its end: ``EXIT_AWAITING_REVIEW`` (an open chain's first
paradigm needs the PI's approval) or ``EXIT_DRAFT_ONLY`` (a test launch: the page and a Prolific
draft are up); anything else that stops it raises.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import tyro

from src.rsa.fit import FitSettings
from src.rsa_open.chain import Chain, ChainConfig, OutOfJobTime
from src.rsa_open.corpus import literature_block
from src.rsa_open.designer import DESIGNER_MODEL, designer_spawner
from src.rsa_open.ground_truth import load_ground_truth
from src.rsa_open.pilot import PILOT_PARTICIPANTS
from src.rsa_open.seeds import OUT as SEEDS

THEORIST_MODEL = DESIGNER_MODEL  # Gemini 4 for every agent, theorists and critic too (PI 2026-10-26)
EXIT_AWAITING_REVIEW = 3
EXIT_DRAFT_ONLY = 4
EXIT_OUT_OF_JOB_TIME = 5



@dataclass
class Args:
    chain: int
    run_dir: Path
    """The chain's run directory, inside the agents' tree."""
    private_dir: Path
    """Simulated: the ground truth's directory; live: the chain's private directory. Outside the
    agents' tree either way."""
    agent_root: Path
    """The agents' working tree (their cwd)."""
    literature: Path
    """E: every literature trial (promote's all_trials.csv on Sherlock), as the ground truth's fit."""
    arm: Literal["open", "fixed"] = "open"
    """"fixed": the closed forced-choice paradigm every experiment, no designer (`src.rsa_open.fixed`)."""
    fixed_spec: Optional[Path] = None
    """The fixed arm's spec, if not the closed paradigm's (the two-measure rehearsal's)."""
    seeds: Path = SEEDS
    n_experiments: int = 4
    max_iterations: int = 5
    candidate_count: int = 6
    cv_folds: int = 5
    """Grouped CV folds per block (fewer for a dry run)."""
    pilot_participants: int = PILOT_PARTICIPANTS
    designer_model: str = DESIGNER_MODEL
    theorist_model: str = THEORIST_MODEL
    theorist_timeout_sec: int = 2400
    seed: int = 0
    other_chains: Tuple[Path, ...] = ()
    """The campaign's other chains' run directories: their finished live experiments' timing goes
    into every design's brief (none in a simulated rehearsal)."""
    no_sandbox: bool = False
    code_commit: Optional[str] = None
    collection: Literal["simulated", "live"] = "simulated"
    prolific_mode: Literal["test", "live"] = "test"
    """Live: "test" deploys experiment 1's page and a Prolific draft, then stops; "live" publishes."""
    confirm_live_recruitment: bool = False
    live_config: Optional[Path] = None
    """The campaign's live settings (default scripts/rsa/live/open_live.yaml)."""
    run_label: Optional[str] = None
    """This chain's label on its studies and Hosting paths (live)."""
    repo_root: Optional[Path] = None
    """The checkout that deploys the page (live; the job's deploy copy)."""
    campaign_dir: Optional[Path] = None
    """Shared by every chain of the campaign: participant ids, recruiting slots (live)."""
    participants: Optional[int] = None
    """A smoke test: recruit this many, then stop (needs --stop-after-collect; never modelled)."""
    stop_after_collect: bool = False
    job_deadline: Optional[float] = None
    """When this job ends (epoch seconds; the job script reads it from Slurm): no stage is started
    without the time it needs (`src.rsa_open.chain.STAGE_MIN_HOURS`)."""


def _commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(args: Args) -> Chain:
    from src.rsa.loop.orchestrator import coding_agent_spawner

    run_dir = Path(args.run_dir)
    gt, live = None, None
    if args.collection == "live":
        from src.rsa_open.live import OPEN_LIVE_CONFIG, load_live_config, read_campaign_config

        missing = [k for k in ("run_label", "repo_root", "campaign_dir") if getattr(args, k) is None]
        if missing:
            raise ValueError(f"live collection needs --{', --'.join(m.replace('_', '-') for m in missing)}")
        path = args.live_config or OPEN_LIVE_CONFIG
        campaign = read_campaign_config(path)
        live = load_live_config(path, prolific_mode=args.prolific_mode,
                                confirm_live_recruitment=args.confirm_live_recruitment,
                                firebase_project=campaign["firebase_project"], run_label=args.run_label,
                                repo_root=Path(args.repo_root), private_dir=Path(args.private_dir),
                                campaign_dir=Path(args.campaign_dir), participants=args.participants)
    else:
        gt = load_ground_truth(args.private_dir)
    cfg = ChainConfig(chain=f"chain_{args.chain}", arm=args.arm, fixed_spec=args.fixed_spec, seeds_dir=Path(args.seeds) / f"chain_{args.chain}",
                      bar_dir=Path(args.seeds) / "bar", literature=literature_block(args.literature), out_dir=run_dir,
                      ground_truth=gt, live=live, stop_after_collect=args.stop_after_collect,
                      deadline=args.job_deadline,
                      n_experiments=args.n_experiments, settings=FitSettings(dense_mass=True),
                      pilot_participants=args.pilot_participants, seed=args.seed, other_chains=args.other_chains,
                      code_commit=args.code_commit or _commit(),
                      loop=dict(max_iterations=args.max_iterations, candidate_count=args.candidate_count,
                                cv_folds=args.cv_folds))
    designer = None if args.arm == "fixed" else designer_spawner(
        agent_root=args.agent_root, model=args.designer_model, sandbox=not args.no_sandbox,
        shell_dir=run_dir / ".agent_shell")

    def theorists(loop_dir: Path):
        return coding_agent_spawner(models_dir=loop_dir / "models", responses_path=loop_dir / "data" / "corpus.json",
                                    timeout_sec=args.theorist_timeout_sec, backend="opencode",
                                    model=args.theorist_model, agent_root=args.agent_root,
                                    sandbox=not args.no_sandbox, network=False, shell_dir=loop_dir / ".agent_shell")

    return Chain(cfg, designer_spawn=designer, theorist_spawn=theorists)


def main(args: Args) -> list:
    from src.runtime.token_usage import read_usage_log, start_usage_log, write_usage_report

    # Every agent's and the pilot's spend, in the run directory (run A's were kept only in memory);
    # written in a finally so an aborted chain still accounts for it, and over the whole log, so a
    # resumed chain's summary includes its earlier attempts.
    from src.rsa.live.collect import DraftOnly
    from src.rsa_open.live import AwaitingReview

    usage_log = Path(args.run_dir) / "token_usage.jsonl"
    marker = start_usage_log(usage_log)
    try:
        exports = build(args).run()
    except AwaitingReview as stop:
        print(f"[chain] STOPPED, awaiting the PI's review: {stop}", flush=True)
        raise SystemExit(EXIT_AWAITING_REVIEW)
    except DraftOnly as stop:
        print(f"[chain] STOPPED, {stop}", flush=True)
        raise SystemExit(EXIT_DRAFT_ONLY)
    except OutOfJobTime as stop:
        print(f"[chain] PAUSED: {stop}", flush=True)
        raise SystemExit(EXIT_OUT_OF_JOB_TIME)
    finally:
        write_usage_report(Path(args.run_dir), marker, heading="RSA open chain",
                           records=read_usage_log(usage_log) if usage_log.exists() else [])
    print(json.dumps([{k: e.get(k) for k in ("best_model", "live")} for e in exports], indent=1))
    return exports


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()
    main(tyro.cli(Args))
