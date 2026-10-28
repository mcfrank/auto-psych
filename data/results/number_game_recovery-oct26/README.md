# Number-game recovery sweep, October 2026

Model recovery on the number game: each of the three literature seeds is in
turn the simulated ground truth, and the loop starts from the other two (the
ground truth is held out of the agents' tree). One repeat, three experiments
per cell, 450 designed (example set, target) items × 5 simulated responses
per experiment, Claude Opus 5.5 agents, five rounds of six candidates per
experiment, selection by grouped cross-validation over example sets (4 folds),
`hide_participant_id`, `warm_start_nuts` fits
(`scripts/number_game/configs/holdout_recovery.yaml`, submitted with
`scripts/number_game/slurm/submit_recovery.sh` on cocoflops2).

- `test_retest.{json,csv,png}`: the sweep's summary over its finished cells
  (one repeat, so no reliability statistics).
- `run1/<ground truth>/`: each cell's `holdout.{json,csv,png}` (recovery
  metrics at every scoring step), `trajectory.json` (the best model at every
  step), `agent_activity.md`, `code_commit` and `code_history.txt`.
- `collected/<ground truth>/number_game/experiment<N>/`: the experiment trees
  of the finished cells, as in `../number_game_dataset-oct26/` (responses,
  design, carried models and ledger, every round's proposals and critiques;
  no fit traces). `collected/SUMMARY.md` is the collector's generic table.
- `MISSING_CELLS.txt`, `OUTSIDE_PATH_WARNINGS.txt`: the sweep's own records.

## Results

Scored on the recovery harness's evaluation pool at the last step (step 17):

| ground truth | starting models | final best model | Pearson r | RMSE | KL regret | best changes (of 18 steps) |
|---|---|---|---|---|---|---|
| `bigelow_piantadosi_interval` | independent, Tenenbaum | `uneven_familiarity_conservative_updating` | 0.995 | 0.028 | 0.0020 | 15 |
| `tenenbaum_rules_similarity` | both B&P seeds | `parity_representative_sequence_edges` | 0.998 | 0.023 | 0.0016 | 12 |

In both cells the best model was an agent-written one from the first round
on (17 of 18 steps), and the final one's predictions nearly match the ground
truth's.

## Missing cell

`bigelow_piantadosi_independent` has no result. Its model loop finished all
three experiments (each has `model_loop/export_complete.json`), but the job
was OOM-killed during the harness's final scoring, three times including the
retry rounds (the last at 128 GB). Only its `code_commit` and
`code_history.txt` are here; its tree is still on cocoflops2 at
`/scr/wurgaft/auto-psych/number_game_recovery/run1/bigelow_piantadosi_independent/`.

## Code

The sweep was staged on `25dacd67`. Every cell ran experiment 1 on `01ded870`;
experiments that stalled in a PyTensor thread race were restarted from their
model-loop start on `25dacd67` (per-thread `change_flags`; results unchanged),
as each cell's `code_history.txt` records. `25dacd67` and `e471a07a` reached
`main` with the `cleanup` merge (`45facd5d`).
