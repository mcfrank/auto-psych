# Ablating the sr-oct26 finals: how much of each final model is needed

2026-10-20. The three live runs of the sr-oct26 series each ended on a different final model
with 14–17 population parameters and 4–6 person-level vectors (paper Table 1). The question is
how much of that complexity the human data need. Modelled on the RSA consolidation
(`docs/auto_rsa/CONSOLIDATION.md`), with three differences decided before any fit (user
decisions, 2026-10-20):

- **Each final separately.** Three backward eliminations, each starting from one run's final
  and removing its own mechanisms. There is no cross-run superset. The finals are too unlike
  for one: runs 1 and 2 weigh cue features, run 3 compares Bayesian generators.
- **Grouped CV decides**, as in the RSA work: ELPD from 5-fold CV that holds out whole
  stimulus pairs. A cue must predict pairs it was never fitted on. Trial-level PSIS-LOO
  would let a pair-specific cue learn from the other ~55 answers to the same pair.
- **Person-level variation is complexity too.** Each person-level parameter steps down
  personal → shared → off, and each step is a removal of its own.

The rule below is fixed before any fit, so the lean models come from a fixed procedure, not
from judgement.

## 1. The switchable finals

`src/subjective_randomness/ablation/finals/run{1,2,3}.py` rewrite each final with every
mechanism switchable (`build(on)`). A variant model file is that source plus a footer
building it with some mechanisms on (`variants.py`). Its name lists what was removed:
`r1_full`, `r1_minus_pal_var`.

**Tests** (`tests/test_sr_ablation.py`):
- with every mechanism on, a variant **is** its final: the same free parameters and shapes,
  and the same log density and `p_left` at four jittered points (logp to 1e-10, `p_left`
  to 1e-12);
- each mechanism at its neutral value (`NEUTRAL`: a weight at 0, a person-level SD at 0, a
  lapse at 0, …) gives exactly the model without it, and away from that value it moves some
  prediction;
- the full variants have Table 1's parameter counts (17 + 6p, 15 + 6p, 14 + 4p);
- the full and the smallest variant of each final meet the data contract.

"Personal" below means one value per participant from a population (non-centred). Removing
it gives everyone the population value.

**Run 1, `personal_pattern_detection_gain`** (15 mechanisms). Core: the switching rate
judged against an ideal rate.

| mechanism | what it does | needs |
|---|---|---|
| `ideal_personal`, `beta_personal` | personal ideal rate, personal sensitivity | |
| `asymmetry` | over-alternation penalised by a fitted fraction | |
| `imbalance` / `imbalance_personal` | weight on H/T imbalance | / `imbalance` |
| `longest_run` / `longest_run_personal` | weight on the longest run | / `longest_run` |
| `final_run`, `palindrome`, `variety`, `lopsided`, `antisymmetry` | shared structural cues | |
| `gain_personal` | personal gain on every structural cue | one structural cue |
| `lapse` / `lapse_personal` | guessing rate | / `lapse` |

**Run 2, `edge_streak_primacy_recency_pattern_rule`** (12 mechanisms). Core: the tally span
judged against an ideal span.

| mechanism | what it does | needs |
|---|---|---|
| `span_ideal_personal`, `span_beta_personal` | personal ideal span, personal sensitivity | |
| `length_scaling` | sensitivity scaled by a power of the length | |
| `switch` / `switch_ideal_personal`, `switch_weight_personal` | switching rate against an ideal rate | / `switch` |
| `variety`, `rule_built` | three-flip chunk variety; rule-built sequences | |
| `pattern_gain_personal` | personal gain on both | `variety` or `rule_built` |
| `final_run`, `opening_run` | penalty per flip of the last / first run | |
| `side_personal` | personal left/right lean | |

**Run 3, `mild_symmetric_trick_coin_streak_weary`** (15 mechanisms). Core: fair-coin
evidence against the regular generators, weighed by a sensitivity. At least one generator
stays.

| mechanism | what it does | needs |
|---|---|---|
| `markov`, `trick_coin`, `motif` | the regular generators | (one must stay) |
| `trick_coin_spread` | the trick coin's spread fitted (off: uniform) | `trick_coin` |
| `motif_weight` / `motif_weight_personal` | the motif's prior weight among the generators | `motif` / `motif_weight` |
| `beta_personal` | personal heavy-tailed signed sensitivity | |
| `switch_belief` / `switch_belief_personal` | believed switch rate of a fair coin | / `switch_belief` |
| `length_scaling` / `length_padding` | evidence per flip; pseudo-flip padding | / `length_scaling` |
| `second_order`, `gamblers_run` | second-order and run-length switch beliefs | |
| `side_bias` / `side_personal` | left/right lean | / `side_bias` |

## 2. The rule (fixed 2026-10-20, before any fit)

**Data:** all three runs' final pooled data (`src.subjective_randomness.ablation.pool`):
23,104 trials of 359 people. This is the file behind the paper's pooled fits (Table 1), and a
test checks they are identical.

**Criterion:** ELPD from 5-fold CV grouped by stimulus pair (`src.model_comparison.cv`,
`cv_group="stimulus"`, fold seed 0):
- both presentations of a pair fall in the same fold;
- each held-out trial is scored by its log posterior predictive density;
- people are not held out: a person's parameters are learned from their other pairs;
- the SE of a difference is clustered by stimulus pair (`src.models.clustered_se`).

**Fits:** the paper's settings for fits on all runs pooled (`scripts/paper/sr_figure_data.py`)
at 1,000 draws instead of 1,500:
- NUTS without jitter, 4 chains, 4,000 tuning steps, 1,000 draws, `target_accept` 0.95;
- sampler seeds 20261018, then 6. The first seed whose fit converges within 22 h is used.

**Converged** means the repository's gate (≤ 0.1% divergent transitions, R-hat ≤ 1.05,
bulk ESS ≥ 100) on every **population-level** parameter, on every fold.
- Person-level vectors are recorded (`problems_all`) but not gated. On the pooled data, run
  2's final has one participant whose posterior is two-humped, which fails the full gate
  however long it samples (sr_figure_data, 2026-10-18).
- `--gate all` gates everything.

**Backward elimination** (`src.subjective_randomness.ablation.eliminate`; the rule's code is
shared with the number-game ablation in `src.ablation`, `docs/ng_ablation.md`), per final:
1. Start from the final, with every mechanism on.
2. Each round, score the current model and every model with one mechanism removed. Only
   removals that leave a valid model count: person-level variation goes before the
   parameter it varies, and run 3 keeps a generator.
3. Of the converged candidates, take the one with the highest ELPD-CV. Its cost is the
   current model's ELPD-CV minus its own. Drop its mechanism if the cost is ≤ 0, or below
   **2 clustered SEs**.
4. Stop when every removal costs ≥ 2 SE, or none converged, or nothing is left. The current
   model is then the lean model.

**One change from the RSA rule:** the starting model must converge, or the elimination stops
with an error. The RSA run costed its first drop against an unconverged full model.

No other model is considered, and nothing is added back. `elimination.json` also records the
lean model's total cost against the full final, with its SE: drops that each fall below
2 SE can add up.

## 3. Running it on Sherlock

1. Pull, then on a login node:

   ```bash
   cd $HOME/auto-psych && bash scripts/subjective_randomness/slurm/submit_ablation.sh
   ```

   It submits a **setup** job, then one **driver** chain per final
   (`scripts/subjective_randomness/slurm/ablation.sbatch`).
   - **Setup:** builds the shared venv, stages the code once into `$WORK_ROOT/code`, and
     writes the pooled data. `$WORK_ROOT` is `$SCRATCH/auto-psych/sr_ablation`. Every later
     job runs the staged code, so the checkout can move on. A setup on other code refuses.
   - **Driver** (1 core): replays the rule over the saved fits. It then submits the round's
     missing (model, fold) fits as one array (4 cores, 32 GB, ≤ 48 h, `normal,owners`,
     `--requeue`) and queues the next driver after it.
   - **Stopping:** when nothing is missing, the driver writes `elimination.json`,
     `summary.md` and `lean/` under `$WORK_ROOT/<final>/`. A round still missing fits after
     2 arrays stops with an error.
   - **Knobs:** `SEEDS`, `WORKER_PARTITION`, `WORKER_MEM`, `WORKER_TIME`, `MAX_PARALLEL`,
     `MAX_ATTEMPTS`, `FINALS`.

2. **Cost.** Round 1 alone is 65 + 60 + 50 = 175 fits: (1 + 12, 1 + 11, 1 + 9) models × 5
   folds.
   - Each round after that has one model fewer, and runs until the rule stops.
   - Each fit is ~18,500 trials. The pooled winners took 3.5–7.4 h at the runs' settings,
     so expect several hours to ~15 h per fit here.
   - In all, roughly 10–30 thousand core-hours, spread over days of wall time.
   - `MAX_PARALLEL` caps how many of a round's fits run at once.

3. **If something stops:**
   - **The starting model did not converge on some fold:** add seeds and resubmit that final,
     e.g. `SEEDS="20261018 6 2 1" FINALS=run2 bash .../submit_ablation.sh`. Only the
     untried seeds are run, and only on unconverged folds.
   - **A stuck round:** see `next.json` and `logs/sr_abl_<final>_worker_*`, then resubmit
     the same way. Every finished fold is kept (`cv/<model>/fold<k>.{json,npz}`).
   - **Never run two chains for one final at once** (`squeue -u $USER | grep sr_abl`).
     Two drivers would submit the same round twice.

4. **Bring back** each final's `elimination.json`, `summary.md`, `lean/` and `cv/*/*.json`
   to `data/results/human_experiment_sr-oct26/analysis/results/ablation/<final>/`:

   ```bash
   SRC=shr:/scratch/users/<sunetid>/auto-psych/sr_ablation   # Sherlock's $SCRATCH
   for f in run1 run2 run3; do
     rsync -av --prune-empty-dirs --include='*/' --include='elimination.json' --include='summary.md' \
       --include='lean/**' --include='cv/*/*.json' --exclude='*' \
       "$SRC/$f/" data/results/human_experiment_sr-oct26/analysis/results/ablation/$f/
   done
   ```

   These hold sums and per-fold summaries only; no trial data. The `cv/*/*.npz` files hold
   held-out log densities per trial. They stay on Sherlock unless they are needed.

## 4. Caveats, stated now

- **Each final's mechanisms were chosen on its own run's data,** a third of the pooled data.
  Mechanisms that only fit that run are therefore favoured, and the lean models are an upper
  bound on what is needed. A cleaner test, not set up here, would run the same rule on the
  two runs that did not produce each final (paper Table S1's datasets), passing a pool of
  those two runs as `--data`.
- **Grouped CV holds out stimulus pairs, not people.** It asks whether a mechanism predicts
  new pairs for people already seen. It does not ask whether it predicts new people
  (`docs/person_level_models.md`, plan 3).
- **Drops close to 2 SE** are reported with their cost in SE in `summary.md`. The RSA run had
  two at 1.87 and 1.92 SE.
