"""Self-check for a theorist's theory: the open loop's admission gates with a short fit.

    <the harness python> -m src.rsa_open.loop.check_candidate <candidate_dir> --corpus <corpus.json>

Runs the code gate, the theory checks, the listener and speaker contracts on the corpus and
the novelty pool, and a short fit to every paradigm at once. It does not run the novelty
comparison (that needs the loop's set) and reports non-convergence of the short fit as a
warning, since the loop's fit is longer. Exit code 0 when nothing fails.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tyro

from src.rsa.fit import FitSettings
from src.rsa.loop.novelty import plain_pool
from src.rsa_open.loop.fitting import read_corpus
from src.rsa_open.loop.gates import GateConfig, admit

CHECK_SETTINGS = FitSettings(num_warmup=300, num_samples=300, num_chains=2, dense_mass=True)
CHECK_TIME_LIMIT_SEC = 15 * 60


@dataclass
class Args:
    candidate_dir: tyro.conf.Positional[Path]
    corpus: Path


def check(candidate_dir: Path, corpus: Path) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as cache:
        cfg = GateConfig(corpus, Path(cache), CHECK_SETTINGS, novelty_threshold=0.0,
                         time_limit_sec=CHECK_TIME_LIMIT_SEC)
        out = admit(candidate_dir, "candidate", cfg=cfg, blocks=read_corpus(corpus), pool=plain_pool(),
                    admitted_preds={})
    if out.admitted:
        return True, ("PASS: the theory meets every gate the self-check runs. The loop also checks novelty "
                      "against the theories in the set.")
    if out.reason.startswith("the fit did not converge"):
        return True, ("WARNING (not a failure of the self-check): the short fit did not converge "
                      f"({out.reason}). The loop fits longer; if your theory has parameters the data cannot "
                      "pin down, reparameterise.")
    return False, f"FAIL: {out.reason}"


def main(args: Args) -> int:
    ok, message = check(args.candidate_dir, args.corpus)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()
    sys.exit(main(tyro.cli(Args)))
