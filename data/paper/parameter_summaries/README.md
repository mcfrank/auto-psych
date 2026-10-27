# Parameter summaries of the winning and seed models

Written by `scripts/paper/parameter_summaries.py` (run on Sherlock with
`sbatch scripts/paper/parameter_summaries.sbatch`). It reads the fits made by
the figure-data jobs (`scripts/paper/README.md`), which live on `$SCRATCH`
under `auto-psych/paper/{sr_figure_data,number_game_figure_data}/`, and
samples nothing.

| `fit_set` | models | data | sampler |
|---|---|---|---|
| `sr_pooled` | the 3 runs' winners + 4 seeds | the 3 live runs' final data pooled (23,104 trials, 359 participants) | 4 × 1,500 draws, 4,000 tune, target_accept 0.95, no jitter |
| `sr_run1`..`sr_run3` | that run's winner + 4 seeds | that run's final data | 4 × 3,000 draws, 2,000 tune, target_accept 0.8 |
| `number_game` | the run's winner + 3 seeds | the dataset run's final data, all 7,560 responses (not the CV folds) | warm-start NUTS, 4 × 500 draws, 200 tune |

`role` is `winner` or `seed`; `fit_fingerprint` names the fit file
(`<model>.<fingerprint>.nc`).

- `population_parameters.csv`: one row per scalar parameter, and one per
  element of a vector parameter that is not per person (e.g.
  `rule_weights[3]`): posterior mean, sd, 94% HDI, bulk ESS, R-hat. Both
  free parameters and the deterministic transforms the model stores (e.g.
  `eps` beside `logit_eps`) are listed.
- `person_level_parameters.csv`: one row per vector indexed by
  `participant_id`. The columns summarise the participants' posterior
  means: their mean, sd and 5/25/50/75/95% quantiles, plus the worst R-hat
  and bulk ESS among those participants. Only participants in the data count
  (the models have 400 slots, and the slots no data reached are draws from
  the population distribution). For non-centred models both the raw offsets
  (`z_*`) and any stored person-level transform (e.g. `ideal`, `lapse`)
  appear.

Per-trial predictions (`p_left`, `p_yes`) are left out.

The pooled fit of `edge_streak_primacy_recency_pattern_rule` failed the
convergence gate on one participant's `z_sw` and `z_kappa` (two-humped
posterior; see `accepted_fit_problems` in
`../figure_data/subjective_randomness.json`), which shows here as their
`max_r_hat` ~1.09 and `min_ess_bulk` ~30. Its population-level parameters
converged (R-hat ≤ 1.01).

The RSA reference-game finals and bar are not here: their fits were made
outside this account's `$SCRATCH`.
