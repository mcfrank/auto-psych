"""Run one chain of the open campaign on simulated participants (Phase 3, the rehearsal).

    <agent python> -m src.rsa_open.run_chain --chain 0 --run-dir <agent tree>/_runs/chain_0 \
        --private-dir <private dir with ground_truth.json> --agent-root <agent tree>

The designer (Gemini 4) and the theorists (Gemini 3.8 Flash) are sandboxed opencode agents with
no network (`src.rsa.loop.no_network`); the pilot's participants and its legibility checkers are
the harness's own Gemini calls. The chain starts from ``seeds/chain_<k>`` and designs against
``seeds/bar`` (`src.rsa_open.seeds`); its simulated people are ``--private-dir``'s ground truth
(`src.rsa_open.ground_truth`), which no agent may read. Re-running resumes: each stage whose
output exists is skipped (`src.rsa_open.chain`).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import tyro

from src.rsa.fit import FitSettings
from src.rsa.loop.run import DEFAULT_AGENT_MODEL
from src.rsa_open.chain import Chain, ChainConfig
from src.rsa_open.corpus import literature_block
from src.rsa_open.designer import DESIGNER_MODEL, designer_spawner
from src.rsa_open.ground_truth import load_ground_truth
from src.rsa_open.pilot import PILOT_PARTICIPANTS
from src.rsa_open.seeds import OUT as SEEDS

THEORIST_MODEL = DEFAULT_AGENT_MODEL  # the closed campaign's theorist (PLAN.md, "Theorist")



@dataclass
class Args:
    chain: int
    run_dir: Path
    """The chain's run directory, inside the agents' tree."""
    private_dir: Path
    """The ground truth's directory, outside the agents' tree."""
    agent_root: Path
    """The agents' working tree (their cwd)."""
    literature: Path
    """E: every literature trial (promote's all_trials.csv on Sherlock), as the ground truth's fit."""
    seeds: Path = SEEDS
    n_experiments: int = 3
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


def _commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(args: Args) -> Chain:
    from src.rsa.loop.orchestrator import coding_agent_spawner

    run_dir = Path(args.run_dir)
    gt = load_ground_truth(args.private_dir)
    cfg = ChainConfig(chain=f"chain_{args.chain}", seeds_dir=Path(args.seeds) / f"chain_{args.chain}",
                      bar_dir=Path(args.seeds) / "bar", literature=literature_block(args.literature), out_dir=run_dir,
                      ground_truth=gt, n_experiments=args.n_experiments, settings=FitSettings(dense_mass=True),
                      pilot_participants=args.pilot_participants, seed=args.seed, other_chains=args.other_chains,
                      code_commit=args.code_commit or _commit(),
                      loop=dict(max_iterations=args.max_iterations, candidate_count=args.candidate_count,
                                cv_folds=args.cv_folds))
    designer = designer_spawner(agent_root=args.agent_root, model=args.designer_model, sandbox=not args.no_sandbox,
                                shell_dir=run_dir / ".agent_shell")

    def theorists(loop_dir: Path):
        return coding_agent_spawner(models_dir=loop_dir / "models", responses_path=loop_dir / "data" / "corpus.json",
                                    timeout_sec=args.theorist_timeout_sec, backend="opencode",
                                    model=args.theorist_model, agent_root=args.agent_root,
                                    sandbox=not args.no_sandbox, network=False, shell_dir=loop_dir / ".agent_shell")

    return Chain(cfg, designer_spawn=designer, theorist_spawn=theorists)


def main(args: Args) -> list:
    exports = build(args).run()
    print(json.dumps([{k: e.get(k) for k in ("best_model", "live")} for e in exports], indent=1))
    return exports


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()
    main(tyro.cli(Args))
