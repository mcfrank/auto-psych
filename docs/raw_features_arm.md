# The raw-features arm: making the loop discover the representation

*Added 2026-09-09 on branch `arm-c/raw-features`, off the improvement
campaign's iteration-2 branch. **If you are a review agent in a later
iteration, read this: it changes what a recovery number means.***

## The problem it addresses

The holdout benchmark hands every agent a featurized `data/responses.csv`: 54
columns computed by `src/subjective_randomness/features.py`. Those columns are
close to a union of the seed models' own sufficient statistics, so a candidate
can reach the data by regressing on the column that *is* the mechanism, without
proposing a mechanism at all.

Measured on 3,000 held-out pairs over lengths 4-8, fitting on half and scoring
on the other half — how well a linear function of the provided columns
reproduces each ground truth's own score difference `S(A) - S(B)`:

| Ground truth | best single provided column | \|r\| | R² (held out) |
|---|---|---|---|
| falk_konold_dp | max_run | 0.65 | **1.0000** |
| local_representativeness | multiscale_imbalance | 0.73 | 0.982 |
| finite_experience_occurrence | occ_n50 | 0.96 | 0.944 |
| motif_stack | alts | 0.81 | 0.904 |

`falk_konold_dp` is exact because its difficulty predictor is `n1 + 2·n2` and
both `rep_motifs` and `alt_motifs` are provided columns. So under the full
featurizer, "recovered `falk_konold_dp`" cannot be distinguished from "found the
right weights on two columns we supplied".

Caveats, so nobody over-reads the table. High expressibility does not imply easy
recovery: `local_representativeness` is 0.982 expressible and yet has the worst
measured recovery (0.716 at baseline). And this R² is a linear fit to a score
difference on a uniform pool, whereas the benchmark reports Pearson r on `p_left`
over a pool that is 75% length-8 pairs. The two are related, not commensurable.

Scripts: `$SCRATCH/auto-psych/featurizer_bias/measure.py` and `split_check.py`.

## What the arm changes

One switch, `raw_features: true` in the holdout config:

1. **The agents' CSV carries sequences only.** `strip_to_raw_columns` keeps
   exactly `RAW_RESPONSE_COLUMNS` (`sequence_a`, `sequence_b`, `participant_id`,
   `trial_index`, `chose_left`) and raises if a sequence column is absent.
2. **The EIG design scores models on raw rows too**
   (`run_design_programmatic(raw_features=True)` passes `featurize_path=None`).
   This is not optional: featurizing there as well would put a column name on
   the row that a model's own `compute_features` also returns, and the hook
   raises on a collision.
3. **The seeds compute their own features**, so they remain fittable.

## Two seed sets, and why they must never be merged

| path | used by | models |
|---|---|---|
| `src/subjective_randomness/pymc_model_families/` | every normal run | bind provided columns |
| `src/subjective_randomness/pymc_model_families_raw/` | `raw_features` runs | compute their own columns |

and correspondingly `projects/subjective_randomness/seed_models{,_raw}/` for the
live pool experiment 1 seeds from.

The sets cannot be one set. A model that declares `compute_features` returning
`rep_motifs_a` is unfittable on a CSV that already has that column, and the
inner loop does not fail on it: it prints `[drop] seed model ... cannot be fit`
and carries on with fewer models. The first draft of this arm edited the seeds
in place and silently lost three of four seeds on every featurized run;
`test_raw_features_seeds.py` now pins both sets so that cannot recur:

- each raw seed is byte-for-byte its featurized twin plus an appended block;
- no featurized seed declares `compute_features`;
- the vendored helpers are byte-identical to the originals in `features.py`;
- each raw seed's `compute_features` returns exactly the columns the model
  binds, with the featurizer's own values, over every pair up to length 5.

`motif_stack` is unchanged in both sets: its `prepare_observed` already builds
every array from the raw sequences, and it is the template the others follow.

Columns each raw seed computes for itself:

| model | hook | columns |
|---|---|---|
| falk_konold_dp | compute_features | rep_motifs_{a,b}, alt_motifs_{a,b} |
| finite_experience_occurrence | compute_features | n_{a,b}, occ_n20_{a,b} |
| local_representativeness | compute_features | p_alts_{a,b}, periodicity_{a,b}, multiscale_imbalance_{a,b} |
| motif_stack | prepare_observed | its whole unique-sequence table |

The helpers are copied into each file rather than imported, so a raw seed pulls
in no featurizer and leaves nothing for a candidate agent to import either. They
were extracted with `ast.get_source_segment`, not retyped.

## Why a self-contained model may recompute a harness column

The agents' CSV is raw, but two paths still build *featurized* rows and hand
them to a model: ground-truth generation (`p_left_fixed_params`) and the
held-out trajectory evaluation, both via `model_recovery.feature_rows`. A raw
seed asked to predict there would find its own columns already present.

`_augment_rows_with_features` therefore allows a model to recompute a column the
harness also supplies **when the value agrees** (`rel_tol=1e-9`), and still
fails loudly when it disagrees, which is a model quietly redefining what a
column name means. The response and bookkeeping columns
(`PROTECTED_ROW_COLUMNS`: `chose_left`, `participant_id`, `trial_index`,
`sequence_a`, `sequence_b`) may never be returned at all, whatever the value.

The smoke run found this: the design already ran featurizer-free, and
generation died on `rep_motifs_a` collides. The parity test is what makes the
relaxation safe — the raw seeds' values are equal to the featurizer's by
construction, over every pair up to length 5.

## Known limitation: isolation is by data, not by import

`features.py` is still importable inside a raw-features run, because the parent
process needs it (to generate ground-truth responses and to score the eval
pool), and candidates are loaded in that same process. A candidate that wrote
`from src.subjective_randomness.features import featurize_stimulus` would get
the whole featurizer back in one line.

Nothing enforces against that yet. What exists is measurement: `leakage_check`
records per admitted model the columns it binds with `pm.Data(...)`
(`data_columns`, `n_data_cols`, `max_data_cols`), and in the featurized sweeps
zero of 26 candidates imported the featurizer, so the shortcut was available and
unused. **If you are reviewing a raw-features sweep, check that first**: count
candidates whose source mentions `featurize_stimulus` or
`subjective_randomness.features`. If any do, the arm's numbers are not a
raw-features result, and the fix is to reject such a candidate at admission
(a static check in `_admit_candidate`), which is the natural next step.

## Running it

```bash
SMOKE=1 bash scripts/subjective_randomness/slurm/run_raw_features_arm.sh  # one cheap task
bash scripts/subjective_randomness/slurm/run_raw_features_arm.sh          # 3 x 4 = 12 tasks
```

Output: `$SCRATCH/auto-psych/holdout_raw_features/`. Three replicates with
`BASE_SEED=100`, so repeats 1-3 use seeds 101-103 — the same cells as
`run1`-`run3` of any featurized sweep at that base seed, which is what makes the
comparison paired.

## Reading the result

Compare per ground truth against the featurized sweep from the same commit
(iteration 2's, which is the first sweep with the held-out label stripped and
the export fixed). The prediction the R² table makes:

- `falk_konold_dp` should fall the most, since the featurizer was handing it over
  exactly;
- `motif_stack` should fall the least;
- `local_representativeness` is the interesting case: already the worst recovery
  despite being highly expressible, so if it does *not* fall, its difficulty was
  never about the features.

A collapse across all four is a real result, not a bug, but check the candidate
failure rate first: agents now write feature code inside the fit gate, and 14 of
360 slots already failed the gate under the full featurizer.
