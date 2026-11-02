---
name: sample-size-shifted-criterion
description: exp3 iter_1 c2 repair: criterion shift per example shown (n-2.5) on beta_share learner, replaced trial_index intrusion model; smoke ll -3653.7
metadata:
  type: project
---
exp3 iter_1/candidate_2_repair_1 (2026-10-31), sample_size_shifted_criterion: beta_share learner, logit -= shift*(n_examples-2.5), Normal(0,1). Replaced [[previous-set-concept-intrusion]], which was rejected because prepare_observed read trial_index (admission forbids anything but number_set/target).
Smoke OK in ~2 min, ll -3653.7. Posterior of the shift and novelty vs beta_share were not checked (if shift≈0 it duplicates beta_share).
**How to apply:** never use trial_index or row order in a candidate; cross-trial models can't be admitted. Related: [[kind-uncertainty-shifted-criterion]].
