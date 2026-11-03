---
name: trial-index-banned
description: models may not bind trial_index (rejected at admission); session-drift critiques can't be addressed directly
metadata:
  type: feedback
---
Admission rejects any model that binds `trial_index` (2026-10-30, exp2 iter_0 candidate_5). Stimulus pools for the novelty gate and held-out evaluation carry only number_set and target.
**Why:** a model has to be evaluable on bare stimuli.
**How to apply:** ignore the trial_index_rating_correlation critique and use stimulus-only mechanisms. The repair grafted the forward-counting tilt onto final_digit_multiplicative_concepts; its smoke check took ~220 s. See [[session-drift-null-finding]], [[forward-counting-tilt-finding]].
