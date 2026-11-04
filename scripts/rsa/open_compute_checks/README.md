# Compute checks for the open campaign's stage 2 plan

Written by Claude Opus 5.5; not reviewed by a person.

The scripts behind `docs/auto_open/DISTRIBUTED_DESIGN_REVIEW_V3.md` §§2, 9 and 10, kept **exactly as they ran** on Sherlock on
2026-11-03, against tag `rsa-open-prereg-a2`. They're one-off measurements on copies of chain 1's files, not tools the pipeline
uses. They start no agent, recruit no one and write nothing to the chain. They print and save aggregates only, except that a
seeded `fold` or `baseline` task saves per-trial scores to `opt/pointwise/`, which stay on Sherlock.

## Setup

They expect this layout in `B=$SCRATCH/auto-psych/sched_test`; the sbatch files name these paths directly:

- `$B/clone`: a clean clone at `rsa-open-prereg-a2`.
- `$B/<these scripts>`, copied flat, and `$B/opt/tables/`.
- `$B/opt/input/`: `corpus.json`, `cv/fold_<k>.json`, and for each theory `<theory>/<theory>.py` with its fits and sidecars from
  the chain's `experiment2/loop/.fit_cache` and `models/`.
- `$B/ref_fits/input/`: `corpus.json`, `structural_alignment_pragmatic_tom.py`, and `reference/` with the chain's fit.

The inputs are never committed: `corpus.json` holds trial-level data. Run the scripts on Sherlock for the same reason.
`CPU_GEN:SIE` is the lab node's CPU. `$OPT_TABLE` holds one task's arguments per line.

## What ran

| script | measures | submitted as |
|---|---|---|
| `profile_resume.py`, `.sbatch` | where a restart's time goes | `sbatch profile_resume.sbatch` |
| `ref_fit.py`, `.sbatch` | queue wait, preemption, speed against the lab node | `sbatch -p owners --array=1-24%24 ref_fit.sbatch`; `-p normal --array=1-4` |
| `opt_tests.py fit` via `opt_task.sbatch` | a full fit's time and memory; saves the adaptation | `tables/capture.txt`, `capture2.txt`; `-c 1 --mem=24G` |
| same, parallel and sequential chains | B2 | `tables/parallel.txt` (`-c 4 --mem=32G`), `sequential.txt` (`-c 1 --mem=24G`) |
| `opt_tests.py fold` via `opt_task.sbatch` | B1, 2 theories | `tables/folds.txt`; `-p normal --array=1-15%15 -c 1 --mem=24G -t 04:00:00` |
| `opt_tests.py baseline` | the noise baseline: chain settings, seed 1 | `tables/baseline.txt`; same, `--array=1-25%24` |
| `opt_tests.py fold <theory> <k> 100 <seed>` | B1 on the leaders, seeds 0 and 1 | `tables/reuse.txt`; same, `--array=1-50%24` |
| `opt_tests.py thin` via `opt_thin.sbatch` | B3: fewer fold draws | `sbatch --partition=normal opt_thin.sbatch` |
| `opt_analyze.py <opt dir>` | applies `standing()` to the baseline and reuse scores; writes `opt/results2/leaders.json` | after the arrays |
| `compute_audit.py <run dir>` | core-hours by kind of fit, from the chain's sidecars | by hand |
| `step_waste.py <loop dir>` | leapfrog steps wasted by vectorized chains, from saved fits | by hand |

`opt_task.sbatch` defaults to `owners`; its `-p normal` submissions override that. Run `opt_analyze.py` with the clone as the working
directory and `PYTHONPATH=$B/clone` (`pyprojroot` finds the root from the working directory). A 24 GB task on `normal` is allocated
4 CPUs and uses 1, so it's charged for 4.

## Checks that validate a rerun

- `opt_analyze.py`: "chain seed 0" must reproduce the chain's recorded exp2 scores (`bipolar_boundary_spillover_topic_tom_2`
  -12204.62, `polar_grounding_boundary_spill_tom` -12207.79, `aspect_qud_boundary_tier_shadowing_tom` -12210.19,
  `structural_alignment_pragmatic_tom` -12212.03, `rsa_l2_singleton_feat_color_valence_l0` -12336.35).
- `opt_tests.py thin`: at 500 draws, all recorded scores reproduce exactly.
- `ref_fit.py`: asserts the fingerprint of the lab node's fit before fitting.
- `opt_tests.py fit`: checks the chain's input before fitting.

**Fragile by design:** the paths name chain 1's run tree, and everything assumes the code at tag a2. On later code, A1 and other
changes may break the imports. The fits' caches share `$B/jax_cache`, so a rerun of an identical task hits the cache and looks about
9 minutes faster for a theory that compiles slowly (§10's "cache hit").
