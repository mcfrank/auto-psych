# Number-game recovery sweep, November 2026

Model recovery on the number game with the 2016 paper's seeds: each of
Tenenbaum (2000) and Bigelow & Piantadosi's (2016) Mixture and Independent
models is in turn the simulated ground truth (at its parameters fitted to the
whole dataset), and the loop starts from the other two (the ground truth is
held out of the agents' tree). Three repeats (seeds 1–3) per ground truth,
three experiments per cell, 450 designed (example set, target) items × 5
simulated responses per experiment, Claude Opus 5.5 agents billed to the
subscription, five rounds of six candidates per experiment, selection by
grouped cross-validation over example sets (4 folds), `hide_participant_id`,
`warm_start_nuts` fits (`scripts/number_game/configs/holdout_recovery.yaml`,
submitted with `scripts/number_game/slurm/submit_recovery.sh` on the SC
cluster). Comparable with no earlier sweep: `../number_game_recovery-oct26/`
used the previous seeds.

- `test_retest.{json,csv,png}`: the sweep's summary over all nine cells
  (`scripts/subjective_randomness/holdout_test_retest.py`).
- `run<R>/<ground truth>/`: each cell's `holdout.{json,csv,png}` (recovery
  metrics at every scoring step), `trajectory.json` (the best model at every
  step), `agent_activity.md` and `code_commit`.
  `run1/tenenbaum_rules_similarity/superseded/20261101T111021_finished_cell/`
  holds that cell's records before its redo (below).
- `collected/run<R>/<ground truth>/number_game/experiment<N>/`: the experiment
  trees, as in `../number_game_recovery-oct26/` (simulated responses, design,
  carried models and ledger, every round's proposals and critiques; no fit
  traces), written by `scripts/outer_loop_live/collect_human_results.py`.
  `collected/run<R>/SUMMARY.md` is the collector's generic table.
- `MISSING_CELLS.txt` (`cell_status.py missing`; empty: no cell is missing),
  `OUTSIDE_PATH_WARNINGS.txt` (the sweep roots' own records, concatenated).
- `code_patches/<sweep root>/`: each sweep root's patch log (`README`) and the
  patches it records.
- `readme_tables.py`: prints the tables below from these files.

Every data file here is a byte copy of what the sweep wrote, or the output of
one of the programs named above; the assembly is
`~/auto-psych-runs/nov26b-collect/assemble.sh`.

## Results

Scored on the recovery harness's evaluation pool at the last step (step 17):

| ground truth | repeat | final best model | Pearson r | RMSE | KL regret | best changes (of 18 steps) | first agent-written best |
|---|---|---|---|---|---|---|---|
| `tenenbaum_rules_similarity` | 1 | `digit_reversal_pair_concepts` | 0.993 | 0.038 | 0.0034 | 14 | step 1 |
| `tenenbaum_rules_similarity` | 2 | `new_set_criterion_ratio_ceiling` | 0.995 | 0.028 | 0.0024 | 8 | step 2 |
| `tenenbaum_rules_similarity` | 3 | `representative_coverage_family_doubt` | 0.999 | 0.015 | 0.0006 | 12 | step 1 |
| `bigelow_piantadosi_mixture` | 1 | `range_reestimating_span_hesitant_guess` | 0.999 | 0.013 | 0.0006 | 13 | step 2 |
| `bigelow_piantadosi_mixture` | 2 | `position_decaying_example_evidence` | 1.000 | 0.008 | 0.0004 | 16 | step 1 |
| `bigelow_piantadosi_mixture` | 3 | `approximate_near_miss_rules` | 0.993 | 0.032 | 0.0025 | 14 | step 2 |
| `bigelow_piantadosi_independent` | 1 | `salient_decade_place_value_rules` | 0.938 | 0.101 | 0.0307 | 14 | step 1 |
| `bigelow_piantadosi_independent` | 2 | `final_digit_glance_effortful_search` | 0.993 | 0.033 | 0.0028 | 13 | step 1 |
| `bigelow_piantadosi_independent` | 3 | `square_pair_concepts` | 0.964 | 0.077 | 0.0173 | 15 | step 1 |

| ground truth | Pearson r, mean (sd) over 3 repeats | min | max |
|---|---|---|---|
| `tenenbaum_rules_similarity` | 0.996 (0.003) | 0.993 | 0.999 |
| `bigelow_piantadosi_mixture` | 0.997 (0.003) | 0.993 | 1.000 |
| `bigelow_piantadosi_independent` | 0.965 (0.028) | 0.938 | 0.993 |

ICC(2,1) of Pearson r across repeats: 0.48; mean pairwise correlation between repeats: 0.81

The loop's best model against the fitted starting seeds, mean over the three
repeats, at the end of each experiment:

| ground truth | experiment | loop's best, Pearson r | fitted seeds, Pearson r | loop's best, RMSE | fitted seeds, RMSE |
|---|---|---|---|---|---|
| `tenenbaum_rules_similarity` | 1 | 0.992 | 0.985 | 0.038 | 0.063 |
| `tenenbaum_rules_similarity` | 2 | 0.998 | 0.987 | 0.019 | 0.059 |
| `tenenbaum_rules_similarity` | 3 | 0.996 | 0.988 | 0.027 | 0.055 |
| `bigelow_piantadosi_mixture` | 1 | 0.995 | 0.995 | 0.030 | 0.027 |
| `bigelow_piantadosi_mixture` | 2 | 0.996 | 0.997 | 0.026 | 0.021 |
| `bigelow_piantadosi_mixture` | 3 | 0.997 | 0.998 | 0.017 | 0.018 |
| `bigelow_piantadosi_independent` | 1 | 0.921 | 0.758 | 0.113 | 0.195 |
| `bigelow_piantadosi_independent` | 2 | 0.959 | 0.751 | 0.076 | 0.202 |
| `bigelow_piantadosi_independent` | 3 | 0.965 | 0.767 | 0.070 | 0.194 |

Every cell's best model was an agent-written one from step 1 or 2 on. Against
Tenenbaum and Independent ground truths the loop's best beat the fitted seeds
at every experiment, by a wide margin for Independent. Against the Mixture
ground truth it did not: the fitted seeds (Tenenbaum and Independent) already
predict it almost exactly, and the loop's best matched them without improving
on them. No two repeats of a ground truth ended on the same best model.

## How the sweep ran

The nine cells ran in four work roots on four nodes, on two staged commits:

| work root | node | cells (repeat/ground truth) | staged commit |
|---|---|---|---|
| `nov26b-c1` | cocoflops1 | 1/Tenenbaum, 1/Mixture | `e969f6fe` |
| `nov26b-c2` | cocoflops2 | 1/Independent, 2/Independent | `e969f6fe` |
| `nov26b-jag28` | jagupard28, then tiger5 | 2/Tenenbaum, 2/Mixture, 3/Tenenbaum, 3/Mixture | `81b6e2a4` |
| `nov26b-tiger4` | tiger4 | 3/Independent | `81b6e2a4` |

`e969f6fe` and `81b6e2a4` differ only in the coco cluster's job scripts and
their tests (`81b6e2a4`: bwrap and uv from the shared home, `COCO_PARTITION`). All
four roots then had the same `src/` patches applied mid-sweep, logged with
timestamps in `code_patches/` (a cell's `code_commit` does not change with a
patch):

- 2026-10-30, about 10:45: `81b6e2a4..2543dc56` (recognise Claude's
  "You've hit your 5-hour limit"; no HDF5 file lock on the fit cache).
- 2026-10-30, about 12:30: `2543dc56..d07c467a` (a model loop checkpoints
  every round and resumes from the next). Rounds that the overnight usage
  limit had left with no candidates were then redone from the round they were
  lost in: 1/Mixture, 2/Tenenbaum, 2/Mixture and 3/Independent (each cell's
  `superseded/<timestamp>/before_repair` in its work root).
- 2026-11-01: the export fit-alias fix (committed as `e534c4c6`) in the c1 and
  jag28 roots. 3/Mixture's experiment-1 winner had a fallback name
  (`iter4_candidate5`, the agent's name being over the length limit), and the
  design for experiment 2 could not find its fit under the exported name
  (`FitNotCached`); with the fix it resumed at experiment 2's design.

Two further events:

- 1/Tenenbaum lost experiment 2's round 4 to the usage limit after the
  2026-10-30 redo pass had run, on pre-patch code. It finished with 17 of 18
  steps; its records are kept in `superseded/`. On 2026-11-01 that round was
  redone and experiment 3 rerun in full, so every cell ran the same rounds.
- The jag28 root's run2 cells moved from jagupard28 to tiger5 on 2026-10-30,
  and 3/Mixture on 2026-11-01 (preempted twice by the node's owners), each
  without its fit caches; the cells resumed from their last finished round
  and refitted what they needed.
