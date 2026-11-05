# Number-game dataset runs, November 2026

Three runs (seeds 1–3) of the full pipeline on Tenenbaum's number game ("given
these example numbers, is *t* in the concept too?", yes/no), three experiments
each, with Claude Opus 5.5 agents billed to the subscription. There were no
live participants: each experiment's design picked 280 (example set, target)
items by EIG, and collection drew 9 real responses per item from Bigelow &
Piantadosi's (2016) bank, never reusing a response (`data_source: dataset`,
`scripts/number_game/configs/dataset_run.yaml`: five rounds of six candidates
per experiment, selection by grouped cross-validation over example sets with 4
folds, `hide_participant_id`, `warm_start_nuts` fits). Unlike
`../number_game_dataset-oct26/`, the loop started from the 2016 paper's seeds:
Tenenbaum (2000) and Bigelow & Piantadosi's Mixture and Independent models, so
the two sweeps are not comparable. The runs differ only in their seed, so each
drew its own items and responses.

- `collected/run<R>/number_game/experiment<N>/`: the drawn responses
  (`data/responses.csv`; `model_loop/responses.csv` is the pooled set the loop
  fitted), the design (`design/stimuli.json`), the carried models with the
  ledger of every hypothesis tried (`cognitive_models/`), and every round's
  proposals, critiques and `history.json` (`model_loop/`). Fit traces are not
  included. `collected/SUMMARY.md` is the collector's generic table (written
  for live studies: there are no participants or Prolific IDs here); in it
  "P(best)" is the softmax over grouped-CV ELPD.
- `harness/run<R>/`: each run's own records (`holdout.json`, `trajectory.json`:
  the best model at every scoring step), `code_commit` and `agent_activity.md`
  (what the agents fetched or touched outside their directories: no web calls
  in any run).
- `unseen/run<R>/`: `score_unseen_items.py`'s output (below).
- `readme_tables.py`: prints the tables below from these files.

Every data file here is a byte copy of what the runs wrote, or the output of
one of the programs named above; the assembly is
`~/auto-psych-runs/ngds-nov26-collect/assemble.sh` (from the work roots staged
by `stage.sh` there).

## How the runs went

ELPDs are on each run's own drawn items, so they compare models within a run,
not runs with each other. The "lead" is the winner's grouped-CV ELPD margin over
the second-ranked model, with its standard error; "carried" counts the models
compared at the end of the experiment.

**run1**: the best model changed 15 times in 18 scoring steps; the first agent-written best was at step 1.

| experiment | responses pooled | best at start | best at end | grouped-CV ELPD (per response) | lead over runner-up (dse) | carried |
|---|---|---|---|---|---|---|
| 1 | 2,520 | `bigelow_piantadosi_independent` | `coincidence_sharpened_exemplar` | -1571.2 (-0.623) | 1.6 (5.1) | 8 |
| 2 | 5,040 | `coincidence_sharpened_exemplar` | `round_number_averaging_exemplar` | -2698.6 (-0.535) | 3.3 (4.8) | 8 |
| 3 | 7,560 | `round_number_averaging_exemplar` | `span_fallback_rule_violation` | -4059.0 (-0.537) | 16.3 (9.4) | 3 |

**run2**: the best model changed 17 times in 18 scoring steps; the first agent-written best was at step 1.

| experiment | responses pooled | best at start | best at end | grouped-CV ELPD (per response) | lead over runner-up (dse) | carried |
|---|---|---|---|---|---|---|
| 1 | 2,520 | `bigelow_piantadosi_independent` | `seed_multiple_salience_overlap` | -1475.4 (-0.585) | 14.7 (6.1) | 2 |
| 2 | 5,040 | `final_digit_overshadowing_overlap` | `incremental_band_narrowing` | -3152.6 (-0.626) | 0.0 (6.7) | 8 |
| 3 | 7,560 | `span_interval_similarity` | `neighbourhood_doubt_three_example_memory` | -4721.2 (-0.624) | 0.6 (1.4) | 8 |

**run3**: the best model changed 14 times in 18 scoring steps; the first agent-written best was at step 1.

| experiment | responses pooled | best at start | best at end | grouped-CV ELPD (per response) | lead over runner-up (dse) | carried |
|---|---|---|---|---|---|---|
| 1 | 2,520 | `bigelow_piantadosi_mixture` | `extrapolation_verification_failure` | -1525.5 (-0.605) | 21.4 (8.2) | 2 |
| 2 | 5,040 | `range_bounded_parity_concepts` | `consistent_examples_bounded_rules` | -3088.2 (-0.613) | n/a | 1 |
| 3 | 7,560 | `consistent_examples_bounded_rules` | `exception_weighted_rule_rejection` | -4092.9 (-0.541) | n/a | 1 |


The three runs ended on three different models
(`collected/SUMMARY.md`, "Winning-model agreement"), and of three different
kinds: run 1 an exemplar-similarity model with a rule-violation penalty, run 2
a salience-weighted feature-overlap model with a three-example working-memory
limit, run 3 a tempered Bayesian concept learner over rules and ranges in which
an example that breaks a rule costs it an inferred amount.

## On the items the runs never drew

`score_unseen_items.py` fits each model on the run's 7,560 pooled responses and
scores it on every (set, target) item of the bank that run never drew. Item R²
is the squared correlation, across items, of predicted P(yes) with the observed
yes rate; the ceiling is the R² that sampling noise in the observed rates
allows. As in `../number_game_dataset-oct26/`, two caveats: much of the
log-likelihood gap can be calibration, and the unseen items share the example
sets each run sampled, so their targets are new but many of their example sets
are not.

**run1**: 24,659 unseen items, 263,833 responses (yes rate 0.316); noise ceiling R² 0.76.

| model | log-lik per response | item R² | % of ceiling |
|---|---|---|---|
| constant: training yes rate | -0.631 | — | — |
| constant: unseen yes rate | -0.624 | — | — |
| `tenenbaum_rules_similarity` | -0.710 | 0.30 | 40% |
| `bigelow_piantadosi_mixture` | -0.705 | 0.36 | 47% |
| `bigelow_piantadosi_independent` | -0.654 | 0.49 | 65% |
| `coincidence_sharpened_exemplar` | -0.554 | 0.47 | 63% |
| `round_number_averaging_exemplar` | -0.544 | 0.50 | 67% |
| `span_fallback_rule_violation` | -0.535 | 0.56 | 74% |

**run2**: 24,659 unseen items, 263,663 responses (yes rate 0.314); noise ceiling R² 0.76.

| model | log-lik per response | item R² | % of ceiling |
|---|---|---|---|
| constant: training yes rate | -0.658 | — | — |
| constant: unseen yes rate | -0.622 | — | — |
| `tenenbaum_rules_similarity` | -0.742 | 0.23 | 30% |
| `bigelow_piantadosi_mixture` | -0.701 | 0.25 | 33% |
| `bigelow_piantadosi_independent` | -0.622 | 0.50 | 65% |
| `seed_multiple_salience_overlap` | -0.559 | 0.46 | 61% |
| `incremental_band_narrowing` | -0.524 | 0.63 | 83% |
| `neighbourhood_doubt_three_example_memory` | -0.525 | 0.63 | 82% |

**run3**: 24,659 unseen items, 263,645 responses (yes rate 0.316); noise ceiling R² 0.76.

| model | log-lik per response | item R² | % of ceiling |
|---|---|---|---|
| constant: training yes rate | -0.631 | — | — |
| constant: unseen yes rate | -0.624 | — | — |
| `tenenbaum_rules_similarity` | -0.705 | 0.29 | 38% |
| `bigelow_piantadosi_mixture` | -0.700 | 0.33 | 44% |
| `bigelow_piantadosi_independent` | -0.630 | 0.46 | 61% |
| `extrapolation_verification_failure` | -0.552 | 0.51 | 67% |
| `consistent_examples_bounded_rules` | -0.536 | 0.56 | 75% |
| `exception_weighted_rule_rejection` | -0.535 | 0.57 | 76% |

In every run the final model is ahead of every seed on both measures: 74%, 82% and 76% of the
ceiling against at most 65% for the best seed (Independent), which on log-likelihood does no
better than a constant at the unseen yes rate. In runs 2 and 3 nearly all of the gain was in
place by the end of experiment 2.

Each run's models were scored on that run's own node, from its fit cache where a fit was
already there, with the code and venv the run ran on (`score_run.sh`, a copy of
`~/auto-psych-runs/ngds-nov26-collect/score_run.sh`).

## How the sweep ran

All three runs ran on `b50e1be3` throughout, with no patches, each in its own
work root on an `sc-loprio` node: run 1 on tiger3 (2026-11-01 22:35 to
11-02 07:15), run 2 on tiger1 (to 11-02 14:48), run 3 on tiger4 (preempted
three times by the node's owners; its last start, 2026-11-03 06:35, resumed in
experiment 2's model loop from its last finished round, and it finished at
19:27). Run 3's Slurm log holds only that last start.
