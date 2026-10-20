# Holdout model-recovery plots (design ablation)

Standard recovery plots copied from the four holdout test-retest runs on
`$SCRATCH/auto-psych/` (the `$SCRATCH` copies are the originals and are subject to
the 90-day purge; these are a durable copy). Each condition is one
stimulus-selection design; see `../design_recovery_comparison.{png,csv}` for the
cross-design summary these feed.

## Condition → source run

| dir | condition (n_eig + n_random per experiment) | source run |
|---|---|---|
| `32_eig/`        | 32 EIG, 0 random (pre-random-half)   | `holdout_faithful_test_retest_v8` |
| `64_random/`     | 0 EIG, 64 random                     | `holdout_faithful_64random` |
| `64_random_eig/` | 32 EIG + 32 random                   | `holdout_faithful_32eig_32random` |
| `64_eig/`        | 64 EIG, 0 random                     | `holdout_faithful_64eig` |

## Files in each condition dir

- `test_retest.png` — aggregate test-retest plot over the (up to) 5 repeats.
- `run<N>_<gt_model>.png` — per-repeat recovery scatter for held-out ground-truth
  model `<gt_model>` in repeat `<N>` (flattened from the source
  `run<N>/<gt_model>/holdout.png`).

Held-out models: `falk_konold_dp`, `finite_experience_occurrence`,
`local_representativeness`, `motif_stack`.

## Completeness (of 5 repeats × 4 models = 20 per-repeat plots)

| condition | per-repeat plots | missing |
|---|---|---|
| 32 EIG        | 20/20 | — |
| 64 random     | 20/20 | — |
| 64 random+EIG | 19/20 | finite_experience_occurrence: 4/5 |
| 64 EIG        | 16/20 | finite_experience_occurrence: 2/5; local_representativeness: 4/5 |

Missing per-repeat plots correspond to repeats that did not complete; the
aggregate `test_retest.png` is computed from whatever repeats finished.
