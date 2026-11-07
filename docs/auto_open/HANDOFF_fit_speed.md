# Handoff: fit speed after deduplication, on chain 1's real corpus (2026-11-08)

For the Sherlock session. Written by Claude Opus 5.5; not reviewed by a person.

**What changed.** The open campaign's fit (`src/rsa_open/fit.py`, `OpenData`) evaluated the theory once
per *trial*. It now evaluates it once per distinct display, and the link once per distinct (display,
response) pattern, weighted by its count, as the closed campaign's fit always did. Per-trial results are
exact: `tests/test_rsa_open_fit_patterns.py` holds every link to the old per-trial computation.

Measured in a 4-core cloud container on the committed CC-BY literature sources (27,949 trials, 173
displays, 402 patterns), at the loop's settings (4 chains, dense mass), for the depth-3 starting
theory `isolated_graded_costly_l3`:

| | per trial (old) | per pattern (new) |
|---|---|---|
| one gradient | 25.9 ms | 1.8 ms |
| NUTS 250 + 250 | 854 s | 77 s |
| NUTS 1,000 + 1,000 | about 55 min (extrapolated) | 119 s (about 42 s of it compiling) |

The posterior was the same (e.g. alpha 0.964 against 0.966), and so was the potential to float32
rounding. Separately, building the data took about 20 s per 17k trials, from thousands of small device
arrays; it now builds arrays for the distinct displays only.

**First, the slow tests** of the changed code, which a 4-core container can't run in reasonable time
(the fast ones, 131 across the fitting, corpus, gate, loop and critique modules, pass):
`uv run pytest -q tests/test_rsa_open_fit.py tests/test_rsa_open_fit_patterns.py tests/test_rsa_open_corpus.py
tests/test_rsa_open_loop_fitting.py tests/test_rsa_open_loop_orchestrator.py`.

**What to run** (no agents; fits in this process; aggregates only), with
`scripts/rsa/open_compute_checks/fit_speed.py`, on a clone of `auto-psych-open` at this commit and,
for comparison, one at `rsa-open-prereg-a2` (the code chain 1 runs):

1. `fit_speed.py data/rsa_open/seeds/bar/isolated_graded_costly_l3.py` with no corpus: should
   reproduce the table above on a Sherlock core (and tell us how Sherlock's cores compare).
2. With `--corpus` = chain 1's `experiment3/corpus.json` (copied, never modified):
   - three of experiment 3's slowest admitted theories (by their fit sidecars' `timing`) and the
     depth-3 seed, at the loop's settings (1,000 + 1,000);
   - the same theories at chain 1's recorded times, for comparison (from the sidecars; no refit);
   - trials, displays and patterns per block, as the script prints them.
3. Peak memory (`/usr/bin/time -v` or `sacct MaxRSS`), new code only: chain 1's fits peaked at 12-15 GB,
   and some folds at 28-32 GB.

Run them as single-core jobs on `owners` or `normal` (not the lab node while chain 1 is on it), 24 GB.

**Report** in `DISTRIBUTED_DESIGN_REVIEW_V3.md` §13 (or after the refit test's section): a table of
theory × (chain 1's recorded time, the new code's time, its warm-up and sampling split, steps per
iteration, peak memory), and whether each new fit converged.

**Why it matters.** If real fits drop from tens of minutes to a few, then step mode's arrays would
mostly be jobs under Slurm's sensible minimum (SRCC: 20 minutes), the fits could run inside the
experiment's own job again, and the 3-day scoping (`HANDOFF_refit_test.md`) can be revisited with
full-fidelity settings. The refit test still stands: it measures cheaper settings, not cheaper code.
