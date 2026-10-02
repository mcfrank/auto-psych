# Full live run sr-oct26

Three independent runs (run1–run3) of the whole pipeline, three experiments
each, 40 Prolific participants per experiment, Claude Opus 5.5 agents
(`full_run_sr-oct26.yaml` at the repository root; runbook:
`docs/onboarding/full_run_checklist.md`). **Status (2026-10-02): run 1 is
complete; runs 2 and 3 have not been launched.**

An experiment is final when its `model_loop/export_complete.json` exists;
`collected/SUMMARY.md` lists each experiment's winning model and marks
unfinished ones `incomplete`. Prolific IDs (participant and study) are
scrubbed everywhere; the raw collected data stay on Sherlock only.

Per run, `collected/run<i>/subjective_randomness/experiment<N>/` holds:
- `data/responses.csv`: the models' data (`sequence_a, sequence_b,
  participant_id, trial_index, chose_left`); `participant_id` is numbered
  within the run (unique across its experiments) and never a Prolific ID
- `design/stimuli.json`: the chosen stimulus pairs and their expected
  information; `design/screened_out.json`: models left out of the design
- `cognitive_models/`: the models carried to the next experiment, with
  `models_manifest.yaml` and the ledger of every hypothesis tried
- `model_loop/`: every round's proposals, critiques, `history.json` (the best
  model after every round), `model_posterior.json` and `report.md`

Browse it: `python -m src.viewer.server` (see `src/viewer/`).

## Run 1 (complete)

Hosting site `auto-psych-2c5da-0926-run1`; 120 people (2 of them took part
twice, below); Opus about $188, Prolific about $224.

| experiment | trials pooled | best model at the end | ELPD-LOO | carried |
|---|---|---|---|---|
| 1 | 2,560 | `individual_streak_aversion_lapse` | −931.4 | 6 (after the re-prune below) |
| 2 | 5,120 | `most_lopsided_stretch_aversion` | −2255.7 | 8 |
| 3 | 7,680 | `personal_pattern_detection_gain` | −3451.1 | 8 |

The best model changed 4 times in experiment 1, 5 times in experiment 2 and
4 times in experiment 3, every time to a model the loop discovered. All the
leading models have person-specific parameters (each person's ideal
switching rate, sensitivity, imbalance and streak weights, guessing rate) and
accumulate shared cues for designed-looking sequences (final streak, mirror
and complement symmetry, run-length variety, the most lopsided stretch). In
experiment 3 the top four are tied within 0.6 nats: the incumbent,
`capacity_blur_pattern_gain` (a person-specific working-memory capacity),
`returns_to_balance_expectation` (the running heads–tails tally should keep
returning to even) and `sensitivity_linked_consistency`.

### What happened, in order

The run recorded the commits below in its manifests and `code_provenance.json`
under their **pre-rewrite** hashes (`main`'s history was rewritten on
2026-10-01 to scrub Prolific IDs; the mapping is in the checklist).

1. **Experiment 1** was designed, deployed and collected on `2ba041f` (2026-09-29).
   Its model loop ran without a fit cache, re-sampling every model at every
   admission (4.5 hours per round); it was cancelled and rerun from scratch
   on `f846b11` (the fit cache).
2. Experiment 1's pruning (at 2·clustered SE) kept a **single** model, and the
   experiment-2 design stopped: it dropped every model with person-specific
   parameters, and with one model there is nothing to discriminate. No study
   was created. Fixed on `a93563b` (user decisions 2026-09-30): a design after
   data predicts a person-level model as 40 **new** participants, marginalizing
   its person-specific parameters over its own population distribution
   (`docs/person_level_models.md`), and the series prunes at **4**·clustered
   SE. Experiment 1 was then **re-pruned at 4**
   (`src/pipelines/outer_loop/reprune.py`; nothing before pruning depends on
   the threshold, so this is what a loop run at 4 would have carried; checked
   on a copy, where re-pruning at 2 reproduced every decision), before any
   experiment-2 data existed. The record is
   `experiment1/model_loop/repruned.json`.
3. **Experiments 2 and 3** ran on `a93563b` (the fit cache, the novelty gate's
   saved predictions, the person-level design, pruning at 4). Every carried
   model was used in both designs; none was screened out.
4. Experiment 3's model loop ran out of memory in round 2 (67 GB of 64; the
   loop holds every loaded fit, about 1.5 GB a model at 7,680 trials). It was
   rerun from scratch at 128 GB (peak 77 GB) and completed. `run_live.sbatch`
   now asks for 128 GB (`bcdb44d`).
5. **Two participants took part in both experiment 1 and experiment 2** (the
   same Prolific account). Collection recognised them and kept their
   experiment-1 ids, so the pooled data treat them as the same two people,
   but they saw the task twice. From runs 2 and 3 on, every live study
   excludes the participants of every earlier study (below).
6. Prolific held back 31 of run 1's submissions (13, 15 and 3 in
   experiments 1–3) as `AWAITING REVIEW` despite automatic approval: every
   one finished in under about 3.5 minutes of the 7 estimated, Prolific's
   threshold for approving on its own. Their data are in `responses.csv`
   like everyone's; they are paid once reviewed in the Prolific dashboard.

The one-off scripts that did these recoveries are in `run1_recovery/`
(Sherlock paths and job ids as they were; study IDs redacted). The general
procedure is the runbook's § 9 and the checklist's step 9.

## Runs 2 and 3: before launching

They run on the current `main` (everything above, with 128 GB, and the
exclusion of earlier participants). From a clone made after 2026-10-01 (or
re-cloned; never push a pre-rewrite clone's branches):

```bash
cd $REPO && git pull
CONFIG=$REPO/full_run_sr-oct26.yaml RUNS=2 AUTO_PSYCH_COLLECTION_OWNER=<you> \
  bash scripts/outer_loop_live/start_full_run.sh
# once run 2's experiment-1 study is full (about an hour later), the same with RUNS=3
```

Every live study now excludes everyone who took part in an earlier published
study of the pipeline in the Prolific account (run 1's three, the pilot's,
and the other run's so far), read from the account just before the study is
created. Prolific applies that list when the study is published, so two
studies published together cannot exclude each other's participants:
launching both runs at once (`RUNS="2 3"`) would let the same people take
both runs' first experiments, which is why run 3 starts after run 2's first
study has filled. Later experiments publish at times set by the model loops;
someone taking one run's study at the moment the other run's is published is
still not excluded.

Never include run 1 in `RUNS`: relaunching an index deletes its results.
After each experiment, sync the results here (checklist step 11, with
`RUNS="run1 run2 run3"`).

Open decisions:
- **Generalization to new people.** The loop selects and prunes by ELPD-LOO
  that leaves out one *trial*, which rewards fitting known people. A
  new-subject ELPD (each participant scored under a fresh draw from the
  model's population) is planned (`docs/person_level_models.md`, plan 3);
  each later experiment's new participants are a natural test set.
- Model-loop memory grows with the zoo and the data (77 GB at experiment 3);
  freeing each fit's per-trial arrays once its scores are computed would cap
  it. 128 GB is enough for 3 × 40.
