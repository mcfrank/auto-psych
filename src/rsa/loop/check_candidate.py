"""Self-check for an agent's memo candidate: the admission gates with a short fit.

    uv run python -m src.rsa.loop.check_candidate <candidate_dir> --responses <responses.csv>

Runs the code gate, loading, the contract on the training displays and on
the novelty pool's shapes, a short NUTS fit (zero-probability choices, finite
ELPD) and the pool predictions. It does not run the novelty comparison
(that needs the loop's model set) and reports non-convergence of the short
fit as a warning, since the loop's fit is longer. Exit code 0 when nothing
fails.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tyro

from src.rsa.dataset import DEFAULT_TRIALS_CSV, load_forced_choice
from src.rsa.fit import FitSettings
from src.rsa.loop.gates import GateConfig, admit
from src.rsa.loop.novelty import novelty_pool

CHECK_SETTINGS = FitSettings(num_warmup=300, num_samples=300, num_chains=2)
CHECK_TIME_LIMIT_SEC = 10 * 60


@dataclass
class Args:
    candidate_dir: Path
    responses: Path = DEFAULT_TRIALS_CSV


def check(candidate_dir: Path, responses: Path) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as cache:
        cfg = GateConfig(
            responses_path=responses, cache_dir=Path(cache), settings=CHECK_SETTINGS,
            novelty_threshold=0.0, time_limit_sec=CHECK_TIME_LIMIT_SEC,
        )
        out = admit(
            candidate_dir, "candidate", cfg=cfg,
            training=load_forced_choice(responses).contexts, pool=novelty_pool(), admitted_preds={},
        )
    if out.admitted:
        return True, (
            f"PASS: the candidate meets every gate the self-check runs "
            f"(short-fit ELPD-LOO {out.elpd_loo:.1f}). The loop also checks novelty "
            f"against the models in the set."
        )
    if out.reason.startswith("the fit did not converge"):
        return True, (
            "WARNING (not a failure of the self-check): the short fit did not converge "
            f"({out.reason}). The loop fits longer; if your model has parameters the "
            "data cannot pin down, reparameterise."
        )
    return False, f"FAIL: {out.reason}"


def main(args: Args) -> int:
    ok, message = check(args.candidate_dir, args.responses)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
