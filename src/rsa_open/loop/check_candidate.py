"""Self-check for a theorist's theory: the open loop's admission gates with a short fit.

    <the harness python> -m src.rsa_open.loop.check_candidate <candidate_dir> --corpus <corpus.json> \
        --full-corpus <the loop's corpus.json>

Runs the code gate, the theory checks, the listener and speaker contracts on the corpus and
the novelty pool, and a short fit to every paradigm at once. It does not run the novelty
comparison (that needs the loop's set) and reports non-convergence of the short fit as a
warning, since the loop's fit is longer. Then it says how hard the posterior was to sample
(`src.rsa_open.sampler`: steps per draw, step size, correlated parameters) and projects the
loop's admission fit on ``--full-corpus`` from the short fit; a projection over the admission
limit fails, since the loop would reject the theory as too slow. Exit code 0 when nothing fails.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import tyro

from src.rsa.fit import FitSettings
from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC
from src.rsa.loop.novelty import plain_pool
from src.rsa_open.corpus import Block, CorpusFit
from src.rsa_open.fit import TheoryModel
from src.rsa_open.loop.fitting import read_corpus
from src.rsa_open.loop.gates import GateConfig, admit
from src.rsa_open.sampler import projected_fit_seconds, step_seconds

CHECK_SETTINGS = FitSettings(num_warmup=300, num_samples=300, num_chains=2, dense_mass=True)
CHECK_TIME_LIMIT_SEC = 15 * 60


@dataclass
class Args:
    candidate_dir: tyro.conf.Positional[Path]
    corpus: Path
    full_corpus: Path
    """The loop's corpus (all the data), for the projected admission fit time."""


def sampling_lines(fit: CorpusFit, theory, full_blocks: Sequence[Block],
                   full_settings: FitSettings = FitSettings(dense_mass=True)) -> tuple[list, Optional[float]]:
    """The sampler report's lines and the projected admission fit (seconds), from the short fit."""
    report = fit.sampler_report()
    lines = [f"- {line}." for line in report.lines()] if report else []
    if not fit.timing:
        return lines, None
    full_step = step_seconds(theory, full_blocks, full_settings.num_chains)
    seconds = projected_fit_seconds(fit.timing, CHECK_SETTINGS, full_settings, full_step)
    lines.append(f"- Projected admission fit on all the data: about {seconds / 60:.0f} minutes (about "
                 f"{seconds / full_step:,.0f} leapfrog steps at {full_step * 1000:.0f} ms each; rough, ±40%). "
                 f"The loop's limit is {FIT_TIME_LIMIT_SEC // 60} minutes.")
    return lines, seconds


def check(candidate_dir: Path, corpus: Path, full_corpus: Path) -> tuple[bool, str]:
    blocks = read_corpus(corpus)
    with tempfile.TemporaryDirectory() as cache:
        cfg = GateConfig(corpus, Path(cache), CHECK_SETTINGS, novelty_threshold=0.0,
                         time_limit_sec=CHECK_TIME_LIMIT_SEC)
        out = admit(candidate_dir, "candidate", cfg=cfg, blocks=blocks, pool=plain_pool(), admitted_preds={})
    lines, seconds = (sampling_lines(out.fit, TheoryModel(Path(candidate_dir) / "candidate.py", name="candidate"),
                                     read_corpus(full_corpus)) if out.fit is not None else ([], None))
    sampling = ("\n\nSampling:\n" + "\n".join(lines)) if lines else ""
    if seconds is not None and seconds > FIT_TIME_LIMIT_SEC:
        return False, (f"FAIL: the admission fit on all the data would take about {seconds / 60:.0f} minutes, over "
                       f"the loop's {FIT_TIME_LIMIT_SEC // 60}-minute limit: the loop would reject it as too slow. "
                       f"Make the posterior easier to sample (below) or the theory cheaper." + sampling)
    if out.admitted:
        return True, ("PASS: the theory meets every gate the self-check runs. The loop also checks novelty "
                      "against the theories in the set." + sampling)
    if out.reason.startswith("the fit did not converge"):  # its reason already gives the sampler report
        return True, ("WARNING (not a failure of the self-check): the short fit did not converge "
                      f"({out.reason}). The loop fits longer; if your theory has parameters the data cannot "
                      "pin down, reparameterise." + ("\n\n" + lines[-1] if seconds is not None else ""))
    return False, f"FAIL: {out.reason}"


def main(args: Args) -> int:
    ok, message = check(args.candidate_dir, args.corpus, args.full_corpus)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()
    sys.exit(main(tyro.cli(Args)))
