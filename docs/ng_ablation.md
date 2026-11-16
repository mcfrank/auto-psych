# Ablating the number-game final: how much of the final model is needed

> **Status (2026-11-15): the November final is rewritten, not yet run.** The paper's
> number-game runs are the November sweep (`data/results/number_game_dataset-nov26/`, from
> the 2016 paper's seeds; user decision 2026-11-12). The final to ablate is the November
> winner best on all runs pooled: run 2's `neighbourhood_doubt_three_example_memory`
> (grouped-CV ELPD −12,805 on the 22,680 pooled responses, 135 ± 40 nats ahead of run 1's
> winner; all its pooled fits converged; to be confirmed against the figure data's
> `pooled.best_model`). Its switchable rewrite and mechanisms are in section 1 (every
> parameter in a removable mechanism, the intercept and slope included: user decision
> 2026-11-15); the rule (section 2) is unchanged. The result goes to `data/results/number_game_dataset-nov26/analysis/ablation/`,
> where `scripts/paper/ablation_figure.py` reads it. The October final's mechanisms and its
> elimination are in git history (before `c891612f`).

2026-10-27. The number-game dataset run (`data/results/number_game_dataset-oct26/`) ended on
`even_rule_digit_size_principle_counting`: a logistic model of P(yes) with 13 terms and 14
parameters. The question is how much of that the data need. This is the sr-oct26 ablation
(`docs/sr_ablation.md`) applied to this one final: the same rule and code, with these
choices (user decisions, 2026-10-27):

- **One final**, the run's best model at its end
  (`collected/run1/number_game/experiment3/cognitive_models/even_rule_digit_size_principle_counting.py`).
- **All three runs' data pooled** (user decision, 2026-11-04): each run's experiment-3
  cumulative `model_loop/responses.csv`, in run order, 22,680 responses on 241 example sets.
  This is the file behind the paper's pooled number-game fits (Figures 2 and 3), as the
  sr-oct26 ablation uses all its runs pooled. Not the full Bigelow & Piantadosi bank.
  - The first elimination (2026-10-27) used run 1's 7,560 responses alone, set before runs 2
    and 3 had finished. It is superseded; its result is in git history
    (`data/results/number_game_dataset-oct26/analysis/ablation/`).
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
- each mechanism at its neutral value (`NEUTRAL`) gives exactly the model without it, and
  away from that value it moves some prediction;
- the full variant has the final's 27 free parameters;
- the full and the smallest variant meet the data contract.

**Computed once per item.** The rewrite computes P(yes) once per distinct (example set,
target) item and indexes each response to its item. The final computes it per response.
The likelihood is the same per-response Bernoulli, so the log density and `p_yes` are the
final's (the first test). The pooled data hold 2,139 items for 22,680 responses: a log
density and its gradient take 4.2 ms instead of 44.7 ms (2026-11-15, one Apple-silicon
core). The final's own pooled fold fits took 8.7–12.3 h on 4 cores at these settings (the
figure data's pooled fits, 2026-11-15); a variant's should take about a tenth.

**`neighbourhood_doubt_three_example_memory`** (25 mechanisms, 27 parameters). Every
parameter is in a removable mechanism (user decision 2026-11-15). What stays is the
structure, which has no parameters: a target is judged by the salience-weighted share of the
examples' shared profile it has, through a logistic. The cues' saliences are one Dirichlet
simplex (`salience`), one component per cue kept; with one cue left, its salience is 1. A cue's neutral value is its
component at 0. The judgment is a ratio of salience-weighted sums, so the other cues' scale
does not matter, and the model without the cue is that point with the rest renormalized.
The final has no person-level parameters: the run hid `participant_id`.

| mechanism | parameters | what it does | needs |
|---|---|---|---|
| `intercept` | intercept (neutral 0) | the logistic's offset: a bias toward yes or no (without it, P(yes) ≥ 0.5 on every item) | |
| `slope` | slope (neutral 1) | the logistic's gain on the share | |
| `parity`, `prime`, `square_cube`, `power`, `multiple`, `last_digit`, `contains_digit`, `round_multiple` | a salience each | the property families the examples share, and whether the target has them (`multiple`: of 3–12 but 5 and 10; `round_multiple`: of 5 and 10; `power`: of 2–10) | |
| `range_low`, `range_high` | a salience each; `band_width` (shared) | the target not below the smallest example / not above the largest, fading as a Gaussian of the gap | (one stays) |
| `seed` | a salience | multiples of the smallest example, when every example is one | |
| `progression` | a salience | the examples' arithmetic progression, continued | |
| `span` | a salience; `span_width` | closeness to the examples' span | |
| `retain` | retain (neutral 1) | a visible shared last digit down-weights every other cue | `last_digit` |
| `recency`, `primacy` | recency, primacy (neutral 0) | with no shared last digit, the last / first example's last digit, relative to the last-digit cue | `last_digit` |
| `doubt_single`, `doubt_multi` | doubt0, doubt_multi (neutral −1000: a doubt rate of 0 in float64); `random_yes` while either is on | rule-ambiguity doubt on one-example / multi-example sets: the target is judged by a flat "yes" rate instead | |
| `doubt_rich`, `doubt_rich_multi` | doubt_rich, doubt_rich_multi (neutral 0) | that doubt rises with the number of rules every example fits | `doubt_single` / `doubt_multi` |
| `parity_keep` | parity_keep (neutral 0) | the doubt's flat "yes" keeps the examples' shared parity | a doubt |
| `doubt_band` | doubt_band_width (neutral ∞) | on multi-example sets the doubt's "yes" keeps their magnitude neighbourhood (the change run 2 ended on) | `doubt_multi` |
| `memory_limit` | drop (neutral 0) | with four examples, one drops out of working memory; the judgment averages over which | |

"Needs" works like person-level variation in sr-oct26: a modifier goes before what it
modifies. The last-digit modifiers act through the last-digit cue's salience: without it,
the down-weighting of every cue is a common scale the ratio ignores. One range cue stays
because every item has the range cues in the judgment's denominator, and no other cue
guarantees a positive one. The band's and the span's widths are part of their cues: they
have no value at which the cue is absent. 22 of the 25 mechanisms are removable at the start,
so round 1 is 23 models × 5 folds = 115 fits. The smallest model is one range cue with its
band width: 1 parameter.

## 2. The rule (fixed 2026-10-27, before any fit; the data changed to all runs pooled on 2026-11-04, after the run-1 elimination, nothing else)

**Data:** all three runs' 22,680 responses pooled (`src.number_game.ablation.data`; a test
checks the file, its count and that it is the paper's pooled file).

**Criterion:** ELPD from 5-fold CV grouped by example set (`src.model_comparison.cv`,
`cv_group=("number_set",)`, fold seed 0):
- every response to an example set falls in one fold, including an example set drawn by
  more than one run (the folds have 4,536 responses each, 48–49 example sets);
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
  sr-oct26 rule (5 folds) applies, and the 244 example sets split into 5 equal folds.
- **Production NUTS, not the run's cheap `warm_start_nuts`** (200 tuning steps, 500 draws).
  Those settings were for selection inside the loop, not for a final analysis.
- **The SE is clustered by example set.** The run's own comparison (`model_posterior.json`)
  clustered by (set, target) item although it held out example sets. Clustering by the CV
  unit is coarser, so its SEs are larger, and more drops pass the 2-SE bar.
- **Measured cost allows the sr-oct26 settings as they are.** One fold fit of the full final
  (6,048 training responses) took 345 s on 4 cores, compilation included (2026-10-27,
  Sherlock, seed 20261018). It converged cleanly: 0 divergences, R-hat 1.00, bulk ESS ≥
  1,500 on every parameter, with a peak memory of ~2.7 GB (1.6 GB in the parent, 0.27 GB per
  chain). A repeat with the compile cache warm took 289 s. Nothing argues for cheaper
  settings.
- **A shorter time limit per attempt** (3 h, not 22 h): about 10× the expected pooled fit
  (the measurement above was on run 1's data, a third of the pooled responses; 1 h when the
  ablation used run 1's data alone).
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
     writes the three runs' pooled responses to `$WORK_ROOT/data/responses.csv`. `$WORK_ROOT`
     is `$SCRATCH/auto-psych/ng_ablation_nov26` (the October ablations'
     `ng_ablation_pooled` and `ng_ablation` hold other code; setup would refuse them). Every later job runs the staged code, so the checkout
     can move on. A setup on other code refuses.
   - **Driver** (1 core): replays the rule over the saved fits. It then submits the round's
     missing (model, fold) fits as one array of 5 fits per task (4 cores, 16 GB, ≤ 8 h,
     `normal,owners`, `--requeue`) and queues the next driver after it.
   - **Stopping:** when nothing is missing, the driver writes `elimination.json`,
     `summary.md` and `lean/` under `$WORK_ROOT/ng/`. A round still missing fits after 2
     arrays stops with an error.
   - **Knobs:** `SEEDS`, `WORKER_PARTITION`, `WORKER_CPUS`, `WORKER_MEM`, `WORKER_TIME`,
     `TASKS_PER_JOB`, `MAX_PARALLEL`, `MAX_ATTEMPTS`.

2. **Cost** of the November final, not yet measured on Sherlock: its own pooled fold fits
   took ~11 h on 4 cores, and the per-item rewrite (section 1) makes a gradient ~10× cheaper,
   so expect ~1–1.5 h and ~5 core-hours a fit. One fold fit of `ng_full` should be measured
   before the chain, and `TASKS_PER_JOB`, `WORKER_TIME` and the per-attempt
   `--time-limit-hours` set from it (the defaults assume ~20-minute fits).
   - Round 1 is 115 fits: (1 + 22) models × 5 folds, about 600 core-hours.
   - Each later round costs one model fewer, plus any it newly opens (a cue whose modifiers
     went). It runs until the rule stops. If every round dropped something down to one cue
     (24 rounds), at most 301 models would be scored (295 in the order that opens the most
     candidates): ~1,500 fits, ~7,500 core-hours.
   - The wall time is set by the rounds, which run in order: up to 24, each ~1.5–3 h of
     fitting plus queue waits.
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
   `data/results/number_game_dataset-nov26/analysis/ablation/`:

   ```bash
   SRC=shr:/scratch/users/<sunetid>/auto-psych/ng_ablation_nov26   # Sherlock's $SCRATCH
   rsync -av --prune-empty-dirs --include='*/' --include='elimination.json' --include='summary.md' \
     --include='lean/**' --include='cv/*/*.json' --exclude='*' \
     "$SRC/ng/" data/results/number_game_dataset-nov26/analysis/ablation/
   ```

   These hold sums and per-fold summaries only; no response data. The `cv/*/*.npz` files hold
   held-out log densities per response. They stay on Sherlock unless they are needed.

## 4. Caveats, stated now

- **The final was selected on run 1's 7,560 responses, a third of these data.** The loop
  chose its mechanisms because they helped there (by grouped CV over those example sets, but
  over many candidates). Mechanisms that only fit run 1's data are therefore still favoured,
  less than when the ablation used run 1's data alone, and the lean model is an upper bound on
  what is needed. A cleaner test, not set up here, would score the same rule on runs 2 and 3
  alone, or on responses no run drew. The bank has 263,569 of them, on items whose example
  sets largely overlap the run's (README, table B).
- **The runs' items were chosen by EIG to be where the models disagree.** They are not a
  sample of the number game. A term can matter a lot on these items and little on typical
  ones, or the reverse. The lean model says what these items need.
- **Grouped CV holds out example sets, not targets or people.** It asks whether a term
  predicts new example sets. The responses come from many people in the bank, without ids,
  so person-level variation is neither modelled nor tested.
- **Drops close to 2 SE** are reported with their cost in SE in `summary.md`.
