"""How long does one theory's fit take, and where does the time go? (2026-11-08)

Runs the loop's own fit (`src.rsa_open.corpus.fit_corpus`) on a corpus and prints aggregates only:
trials, distinct displays and patterns, the time to build the data, warm-up (compilation
included) and sampling, and leapfrog steps per iteration. No agents, nothing written.

    python scripts/rsa/open_compute_checks/fit_speed.py <theory.py> [--corpus <corpus.json>] \
        [--warmup 1000] [--samples 1000] [--chains 4]

Without ``--corpus`` it uses the committed CC-BY literature sources (27,949 trials). With one
(a loop's ``corpus.json``, trial-level: run it where it lives, on Sherlock) it fits that corpus.
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.rsa.fit import FitSettings  # noqa: E402
from src.rsa_open.corpus import Block, fit_corpus, literature_block  # noqa: E402
from src.rsa_open.fit import TheoryModel  # noqa: E402
from src.rsa_open.loop.fitting import read_corpus  # noqa: E402

LITERATURE = ROOT / "src/pipelines/outer_loop/projects/rsa_reference/data"


def committed_literature() -> list:
    parts = [literature_block(LITERATURE / f) for f in
             ("pragmods_trials.csv", "sikos_2021_trials.csv", "mayn_demberg_2026_trials.csv")]
    first = parts[0]
    return [Block("E", first.link, sum((b.contexts for b in parts), ()), sum((b.responses for b in parts), ()),
                  sum((b.units for b in parts), ()), sum((b.sources for b in parts), ()), first.fixed)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("theory", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    a = ap.parse_args()
    t = time.perf_counter()
    blocks = read_corpus(a.corpus) if a.corpus else committed_literature()
    datas = [b.data for b in blocks]
    print(f"data: {time.perf_counter() - t:.1f}s; " + "; ".join(
        f"{b.name} {b.n} trials, {sum(len(g.indices) for g in d.groups)} displays, {d.n_patterns} patterns"
        for b, d in zip(blocks, datas)), flush=True)
    settings = replace(FitSettings(dense_mass=True), num_warmup=a.warmup, num_samples=a.samples, num_chains=a.chains)
    t = time.perf_counter()
    fitted = fit_corpus(TheoryModel(a.theory), blocks, settings)
    tm = fitted.timing
    iters = a.warmup + a.samples
    print(f"fit: {time.perf_counter() - t:.1f}s in all; warm-up {tm['warmup_sec']:.1f}s (compile included), "
          f"sampling {tm['sampling_sec']:.1f}s; steps per iteration "
          f"{(tm['warmup_steps'] + tm['sampling_steps']) / iters:.1f}; converged {fitted.converged}"
          + ("" if fitted.converged else f" ({'; '.join(fitted.convergence_problems)})"), flush=True)


if __name__ == "__main__":
    main()
