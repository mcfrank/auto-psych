---
name: exp3-iter3-set-size-criterion
description: Exp3 iter-3 cand_2 decision slot — yes log-odds shift per example shown (aimed at prefix-extension critique), smoke -3193.3, submitted
metadata:
  type: project
---
On 2026-10-31 I submitted `set_size_yes_criterion` for exp3 iter 3, candidate_2. It reuses the inference of drifting_yes_criterion (the digit_sum_open_ended base). The one change is that trial_index drift is replaced by size_shift*(n_examples-2.5) on the logit, with the lapse and guess kept. It targets the prefix_extension_yes_change critique (z=2.5, q=0.11). Smoke LL is -3193.3, against about -3193.9 for the base, and it ran in 12 s. I did not check novelty. If size_shift≈0, it is a duplicate risk against digit_sum_open_ended_range_discount.

See [[exp3-iter0-drifting-criterion]] and [[exp3-iter3-critique]].
