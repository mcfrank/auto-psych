---
name: decade-boundary-range-extent
description: decade_partitioned_number_line (iter_3 cand_1) — decade crossings add perceived range extent; MAP gain tiny
metadata:
  type: project
---
On 2026-10-21 (experiment3, iter_3, candidate_1), the critique flagged decade_boundary_effect: same-decade near targets get more yeses than the model predicts (q=0.21, not FDR-significant). The candidate decade_partitioned_number_line responds with serial_position_ends_inflated_size plus one change: a range's perceived size becomes width + decade_distance × (decade boundaries crossed), with a HalfNormal(5) prior on decade_distance. Interval groups get an extra crossing key, which raises them from about 450 to 748 per set.

At MAP, decade_distance came out at about 0.96. The log-likelihood was -3188.7, against -3189.1 for the best model, a gain of only 0.4 nats. p_yes RMSE against the best model on the data trials was 0.0032.

**Why:** the decade effect is weak in these data, and a range-prior version (round_landmark_interval_prior) was already pruned.
**How to apply:** do not expect decade or landmark range tweaks to win. See [[map-screening-refinements]].

Convergence repair (2026-10-21, iter_3 candidate_1_repair_1): the original failed R-hat 1.051 / ESS 76. Swapping HalfNormal priors for LogNormal (decade_distance LogNormal(0,0.5); size_inflation LogNormal(log 1.5,0.75)) fixed it. A 4-chain warm-start benchmark (200/250, ta 0.95) gave max R-hat 1.01, min ESS 424 (size_inflation), 0 divergences, 15 steps per iteration, 1230 s on load ~58. Posterior decade_distance 1.06 ± 0.50 (prior-dominated).
**How to apply:** for warm-start NUTS, use LogNormal rather than HalfNormal on positive scale params with posterior mass near 0. The dense Hessian metric fits poorly when the log-space posterior has a long tail.
