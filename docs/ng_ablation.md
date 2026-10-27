# Ablating the number-game final: how much of the final model is needed

2026-10-27. The number-game dataset run (`data/results/number_game_dataset-oct26/`) ended on
`even_rule_digit_size_principle_counting`: a logistic model of P(yes) with 13 terms and 14
parameters. The question is how much of that the data need. This is the sr-oct26 ablation
(`docs/sr_ablation.md`) applied to this one final: the same rule and code, with these
choices (user decisions, 2026-10-27):

- **One final**, the run's best model at its end
  (`collected/run1/number_game/experiment3/cognitive_models/even_rule_digit_size_principle_counting.py`).
- **The run's own data**: experiment 3's cumulative 7,560 responses
  (`model_loop/responses.csv`: 840 (example set, target) items × 9 responses, on 150 example
  sets), not the full Bigelow & Piantadosi bank.
- **Grouped CV over whole example sets** (`cv_group: [number_set]`), the unit the run itself
  selected models by (`scripts/number_game/configs/dataset_run.yaml`, the run's
  `harness/run1/holdout.json`). A term must predict example sets it was never fitted on.

The rule below is fixed before any fit, so the lean model comes from a fixed procedure, not
from judgement.

## 1. The switchable final

`src/number_game/ablation/final.py` rewrites the final with every mechanism switchable
(`build(on)`). As for sr-oct26, a variant model file is that source plus a footer building it
with some mechanisms on (`src/number_game/ablation/variants.py`). Its name lists what was
removed: `ng_full`, `ng_minus_parpre_even`. The variant code is shared with sr-oct26
(`src/ablation/variants.py`).

**Tests** (`tests/test_ng_ablation.py`):
- with every mechanism on, the variant **is** the final: the same free parameters and shapes,
  and the same log density and `p_yes` at four jittered points (logp to 1e-10, `p_yes` to
  1e-12);
- each mechanism at its neutral value (`NEUTRAL`: its weight at 0) gives exactly the model
  without it, and away from that value it moves some prediction;
- the full variant has the final's 14 parameters;
- the full and the smallest variant meet the data contract.

**`even_rule_digit_size_principle_counting`** (13 mechanisms). Every term of the logit is a
mechanism, the intercept included. There is no core: a model only has to keep one term. With
none, P(yes) is 0.5 on every item and there is nothing to fit. The final has no lapse or
noise term (the intercept and the similarity gain calibrate it). It has no person-level
parameters either: the run hid `participant_id`, so there is no person-level variation to
step down.

| mechanism | parameters | what it does | needs |
|---|---|---|---|
| `intercept` | intercept | a bias toward yes or no | |
| `similarity` | gain | credit for the target's final-digit similarity to the examples | |
| `parity` | s_parity | partial similarity for a shared parity without a shared last digit (off: last digits only) | `similarity` or `mutual_standard` |
| `parity_preempt` | preempt | a shared final-digit rule pre-empts the parity credit | `parity` |
| `mutual_standard` | standard | a penalty for the shortfall of the target's similarity below the examples' similarity to each other | |
| `common_factor` | factor_boost | multiples of the examples' common factor (≥ 3, not a multiple of 5) | |
| `round_factor` | round_boost | multiples of a round common factor (a multiple of 5) | |
| `decade_band` | band_boost, band_width | familiarity for targets in or near the examples' decade band | |
| `magnitude_preempt` | magnitude_preempt | an explicit rule (shared last digit, common factor ≥ 3) pre-empts the band credit | `decade_band` |
| `counting` | count_boost | support for targets reached by counting up from one example | |
| `count_step_cost` | step_cost | that support decays per extra step (off: every reach counts alike) | `counting` |
| `size_principle` | size_strength | each confirming example beyond the second makes breaking a shared final-digit rule cost more | |
| `even_rule` | even_rule_penalty | an all-even set rejects odd targets by an extra amount | |

"Needs" works like person-level variation in sr-oct26: a modifier goes before the term it
modifies. `parity` acts only through the similarity and the mutual standard, so it needs one
of them. The band's width is part of `decade_band`: it has no value at which the band term
is absent.

## 2. The rule (fixed 2026-10-27, before any fit)

**Data:** the run's 7,560 responses (`src.number_game.ablation.data`; a test checks the file
and its count).

**Criterion:** ELPD from 5-fold CV grouped by example set (`src.model_comparison.cv`,
`cv_group=("number_set",)`, fold seed 0):
- every response to an example set falls in one fold (the folds have 1,512 responses each);
- each held-out response is scored by its log posterior predictive density;
- the SE of a difference is clustered by example set, the CV unit
  (`src.models.clustered_se.cluster_dse` over `example_set_clusters`).

**Fits:** the sr-oct26 ablation's settings, unchanged:
- NUTS without jitter, 4 chains, 4,000 tuning steps, 1,000 draws, `target_accept` 0.95;
- sampler seeds 20261018, then 6. The first seed whose fit converges within 1 h is used.

**Converged** means the repository's gate (≤ 0.1% divergent transitions, R-hat ≤ 1.05,
bulk ESS ≥ 100) on every parameter, on every fold. All of them are population-level here,
so `--gate population` and `--gate all` are the same.

**Backward elimination** (`src.ablation.eliminate`, the sr-oct26 rule unchanged):
1. Start from the final, with every mechanism on.
2. Each round, score the current model and every model with one mechanism removed. Only
   removals that leave a valid model count (the "needs" column; one term stays).
3. Of the converged candidates, take the one with the highest ELPD-CV. Its cost is the
   current model's ELPD-CV minus its own. Drop its mechanism if the cost is ≤ 0, or below
   **2 clustered SEs**.
4. Stop when every removal costs ≥ 2 SE, or none converged, or nothing is left. The current
   model is then the lean model.

The starting model must converge, or the elimination stops with an error. No other model is
considered, and nothing is added back. `elimination.json` also records the lean model's
total cost against the full final, with its SE: drops that each fall below 2 SE can add up.

**Where the settings differ from what the run used, and why.**
- **5 folds, not the run's 4.** The run's folds served selection inside the loop. Here the
  sr-oct26 rule (5 folds) applies, and 150 example sets split evenly into 5 folds.
- **Production NUTS, not the run's cheap `warm_start_nuts`** (200 tuning steps, 500 draws).
  Those settings were for selection inside the loop, not for a final analysis.
- **The SE is clustered by example set.** The run's own comparison (`model_posterior.json`)
  clustered by (set, target) item although it held out example sets. Clustering by the CV
  unit is coarser, so its SEs are larger, and more drops pass the 2-SE bar.
- **Measured cost allows the sr-oct26 settings as they are.** One fold fit of the full final
  (6,048 training responses) took 345 s on 4 cores, compilation included (2026-10-27,
  Sherlock, seed 20261018). It converged cleanly: 0 divergences, R-hat 1.00, bulk ESS ≥
  1,500 on every parameter. Nothing argues for cheaper settings.
- **A shorter time limit per attempt** (1 h, not 22 h): 10× the measured fit.
- **Several fits per array task.** One fit is shorter than the 10 minutes Sherlock asks a job
  to last, so each worker array task fits `TASKS_PER_JOB` (5) of the round's (model, fold)
  tasks in turn.

## 3. Running it on Sherlock

1. Pull, then on a login node:

   ```bash
   cd $HOME/auto-psych && bash scripts/number_game/slurm/submit_ablation.sh
   ```

   It submits a **setup** job, then a **driver** chain
   (`scripts/number_game/slurm/ablation.sbatch`, modelled on the sr-oct26 one).
   - **Setup:** builds the shared venv, stages the code once into `$WORK_ROOT/code`, and
     copies the run's responses to `$WORK_ROOT/data/responses.csv`. `$WORK_ROOT` is
     `$SCRATCH/auto-psych/ng_ablation`. Every later job runs the staged code, so the checkout
     can move on. A setup on other code refuses.
   - **Driver** (1 core): replays the rule over the saved fits. It then submits the round's
     missing (model, fold) fits as one array of 5 fits per task (4 cores, 8 GB, ≤ 8 h,
     `normal,owners`, `--requeue`) and queues the next driver after it.
   - **Stopping:** when nothing is missing, the driver writes `elimination.json`,
     `summary.md` and `lean/` under `$WORK_ROOT/ng/`. A round still missing fits after 2
     arrays stops with an error.
   - **Knobs:** `SEEDS`, `WORKER_PARTITION`, `WORKER_CPUS`, `WORKER_MEM`, `WORKER_TIME`,
     `TASKS_PER_JOB`, `MAX_PARALLEL`, `MAX_ATTEMPTS`.

2. **Cost**, from the measured fit (~6 min on 4 cores, ~0.4 core-hours):
   - Round 1 is 55 fits: (1 + 10) models × 5 folds. That is 11 array tasks of ~30 min, about
     22 core-hours.
   - Each later round costs one model fewer, plus any it newly opens (a term whose modifier
     went). It runs until the rule stops. If every round dropped something down to one term,
     85 models would be scored: 425 fits, about 170 core-hours.
   - The wall time is set by the rounds, which run in order: up to 13, each ~30–45 min of
     fitting plus queue waits. Expect hours to a day.
   - Smaller variants may sample faster or slower than the full model; the attempt records
     (`cv/<model>/fold<k>.json`, `seconds`) show it.

3. **If something stops:**
   - **The starting model did not converge on some fold:** add seeds and resubmit, e.g.
     `SEEDS="20261018 6 2 1" bash scripts/number_game/slurm/submit_ablation.sh`. Only the
     untried seeds are run, and only on unconverged folds.
   - **A stuck round:** see `next.json` and `logs/ng_abl_worker_*`, then resubmit the same
     way. Every finished fold is kept (`cv/<model>/fold<k>.{json,npz}`).
   - **Never run two chains at once** (`squeue -u $USER | grep ng_abl`). Two drivers would
     submit the same round twice.

4. **Bring back** `elimination.json`, `summary.md`, `lean/` and `cv/*/*.json` to
   `data/results/number_game_dataset-oct26/analysis/ablation/`:

   ```bash
   SRC=shr:/scratch/users/<sunetid>/auto-psych/ng_ablation   # Sherlock's $SCRATCH
   rsync -av --prune-empty-dirs --include='*/' --include='elimination.json' --include='summary.md' \
     --include='lean/**' --include='cv/*/*.json' --exclude='*' \
     "$SRC/ng/" data/results/number_game_dataset-oct26/analysis/ablation/
   ```

   These hold sums and per-fold summaries only; no response data. The `cv/*/*.npz` files hold
   held-out log densities per response. They stay on Sherlock unless they are needed.

## 4. Caveats, stated now

- **The final was selected on these same 7,560 responses.** The loop chose its mechanisms
  because they helped on these data (by grouped CV over these example sets, but over many
  candidates). Mechanisms that only fit these data are therefore favoured, and the lean model
  is an upper bound on what is needed. A cleaner test, not set up here, would score the same
  rule on responses the run never drew. The bank has 263,569 of them, on items whose example
  sets largely overlap the run's (README, table B).
- **The run's items were chosen by EIG to be where the models disagree.** They are not a
  sample of the number game. A term can matter a lot on these items and little on typical
  ones, or the reverse. The lean model says what these items need.
- **Grouped CV holds out example sets, not targets or people.** It asks whether a term
  predicts new example sets. The responses come from many people in the bank, without ids,
  so person-level variation is neither modelled nor tested.
- **Drops close to 2 SE** are reported with their cost in SE in `summary.md`.
