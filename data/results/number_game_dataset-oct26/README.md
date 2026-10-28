# Number-game dataset runs, October 2026

Three runs (seeds 1–3) of the full pipeline on Tenenbaum's number game ("given these example
numbers, is *t* in the concept too?", yes/no), three experiments, with
Claude Opus 5.5 agents. There were no live participants: each experiment's
design picked 280 (example set, target) items by EIG, and collection drew 9
real responses per item from Bigelow & Piantadosi's (2016) bank of 272,700,
never reusing a response (`data_source: dataset`,
`scripts/number_game/configs/dataset_run.yaml`, submitted with
`scripts/number_game/slurm/submit_dataset_run.sh` on the SC cluster). The loop
started from the three literature seeds (Tenenbaum 2000 and two ports of
Bigelow's numpyro models) and selected models by grouped cross-validation over
example sets (4 folds; no PSIS-LOO), with `hide_participant_id`. The runs
differ only in their seed, so each drew its own items and responses.

- `collected/run<R>/number_game/experiment<N>/`: as in the live series
  (`../human_experiment_sr-oct26/README.md`): the drawn responses
  (`data/responses.csv`), the design (`design/stimuli.json`), the carried
  models with the ledger of every hypothesis tried (`cognitive_models/`), and
  every round's proposals, critiques and `history.json` (`model_loop/`). Fit
  traces are not included. `collected/SUMMARY.md` is the collector's generic
  table (written for live studies: there are no participants or Prolific IDs
  here); in it "P(best)" is the softmax over grouped-CV ELPD.
- `harness/run<R>/`: each run's own records (`holdout.json`, `trajectory.json`:
  the best model at every scoring step), the code it ran on (`code_commit`,
  and `code_history.txt` where that changed mid-run) and, for runs 2–3,
  `agent_activity.md` (what the agents fetched or touched outside their
  directories: no web calls).
- `score_unseen_items.py`: how run 1's table B below was computed.

Browse it: `python -m src.viewer.server` (see `src/viewer/`).

## Run 1 (seed 1)

The best model changed 14 times in 18 scoring steps, and from the first round
on it was always one the agents wrote.

| experiment | responses pooled | best at start | best at end | grouped-CV ELPD (per response) | carried |
|---|---|---|---|---|---|
| 1 | 2,520 | `bigelow_piantadosi_independent` | `digit_similarity_with_common_factor` | −1660.3 (−0.659) | 8 |
| 2 | 5,040 | `digit_similarity_with_common_factor` | `mutual_similarity_contrast_standard` | −3153.0 (−0.626) | 7 |
| 3 | 7,560 | `mutual_similarity_contrast_standard` | `even_rule_digit_size_principle_counting` | −4677.2 (−0.619) | 4 |

The winners are logistic models on a few interpretable features instead of
an enumerated hypothesis space: graded similarity of the target's last digit
to the examples' (which decides parity, multiples of 5 and 10 and "ends in
*d*" at once), a boost for multiples of the examples' common factor, a
decade-band magnitude term that an explicit rule overrides, and in the final
model a target judged against how alike the examples are to each other, a
size principle for a shared last digit, support for counting up by
multiples, and "even" (but not "odd") as a privileged concept.

**B. On every item the run never drew** (24,659 items, 263,569 responses;
each model fitted on the run's 7,560 responses; noise ceiling R² 0.76):

| model | log-lik per response | item R² | % of ceiling |
|---|---|---|---|
| base rate (47.5% yes, from the training data) | −0.676 | — | — |
| constant at the true 31% yes rate | −0.621 | — | — |
| Tenenbaum 2000 | −0.718 | 0.26 | 35% |
| Bigelow & Piantadosi, independent | −0.607 | 0.53 | 70% |
| Bigelow & Piantadosi, interval | −0.604 | 0.52 | 68% |
| loop winner after experiment 1 | −0.566 | 0.46 | 60% |
| loop winner after experiment 2 (`rule_gated_decade_familiarity`) | −0.539 | 0.59 | 77% |
| final loop model | **−0.527** | **0.60** | **79%** |

Item R² is the squared correlation, across items, of predicted P(yes) with
the observed yes rate. Two caveats: much of the log-likelihood gap is
calibration (the seeds map belief straight to P(yes) with fixed noise; the
loop's logistic models fit an intercept and gain), and the unseen items share
the 255 example sets the run sampled, so their targets are new but many of
their example sets are not. On the run's own items (chosen by the design to be
where the models disagree), held out by example set, the seeds scored below a
constant base rate.

## Runs 2 and 3 (seeds 2 and 3)

Not yet scored on the unseen items (table B was run 1 only).

**Run 2.** The best model changed 15 times in 18 scoring steps; from the first
round on it was always one the agents wrote.

| experiment | responses pooled | best at start | best at end | grouped-CV ELPD (per response) | carried |
|---|---|---|---|---|---|
| 1 | 2,520 | `bigelow_piantadosi_independent` | `coherence_sharpened_decision_exemplar` | −1541.9 (−0.612) | 8 |
| 2 | 5,040 | `dual_route_shared_factor_rule_exemplar` | `zigzag_breadth_sequence_exemplar` | −3100.8 (−0.615) | 8 |
| 3 | 7,560 | `zigzag_breadth_sequence_exemplar` | `relational_analogy_transfer` | −4665.1 (−0.617) | 7 |

**Run 3.** The best model changed 7 times in 18 scoring steps; from the first
round on it was always one the agents wrote. Experiment 2 ended with one model
carried: everything else was pruned behind `outlier_tolerant_rule_concepts`.

| experiment | responses pooled | best at start | best at end | grouped-CV ELPD (per response) | carried |
|---|---|---|---|---|---|
| 1 | 2,520 | `bigelow_piantadosi_interval` | `session_growing_acquiescence_concepts` | −1587.6 (−0.630) | 7 |
| 2 | 5,040 | `session_growing_acquiescence_concepts` | `outlier_tolerant_rule_concepts` | −2796.5 (−0.555) | 1 |
| 3 | 7,560 | `outlier_tolerant_rule_concepts` | `outlier_tolerant_rule_concepts` | −3867.6 (−0.512) | 7 |

ELPDs are on each run's own drawn items, so they compare models within a run,
not runs with each other. The three runs ended on three different models
(`collected/SUMMARY.md`, "Winning-model agreement").

Code: run 2 ran throughout on `54c48cc7`. Run 3's experiment 1 ran on
`01ded870`; experiment 2's first attempt (`e471a07a`, log-only diagnostics)
stalled in a PyTensor thread race and was discarded, and experiments 2–3 ran
on `25dacd67` (per-thread `change_flags`; results unchanged), per
`harness/run3/code_history.txt`. Those two commits are on the `cleanup` and
`run3-diagnostics` branches, not on `main`. An earlier seed-3 start on
cocoflops2 left no result and is not included.

## What happened in run 1, in order

The run recorded commits under their pre-merge hashes (the branch was later
replayed onto `main`'s scrubbed history); each is given with its hash on `main`.

1. Experiments 1 and 2 ran on `573e2c9a` (now `e86a7f02`), 2026-10-15 to 16.
   The Claude subscription hit its session limit in experiment 1's first round,
   and `subscription_then_api` then kept every later agent on the API key for
   good. Rounds also grew from 38 to 97 minutes: recomputing R-hat over every
   per-trial prediction on each fit load took about an hour of each round end.
2. At experiment 3's start (before any agent ran) the run was stopped and
   resumed on `2395ef69` (now `1f0f7ec7`), where `subscription_then_api` returns
   to the subscription at the limit's stated reset; every experiment-3 agent
   ran on the subscription. The R-hat cost was fixed for later runs in
   `375a82c6` (now `6f3df266`; a replay of experiment 3's round 3 end went from
   81 to under 5 minutes).
3. The run finished on 2026-10-16 at 17:29. The job then crashed printing its
   summary (it expected a ground truth); `holdout.json` was already written.
   Fixed in `c5217946` (now `d82cbab9`).

Agents: 124 calls, about $25 billed to the API (experiments 1–2) and the rest
on the subscription.
