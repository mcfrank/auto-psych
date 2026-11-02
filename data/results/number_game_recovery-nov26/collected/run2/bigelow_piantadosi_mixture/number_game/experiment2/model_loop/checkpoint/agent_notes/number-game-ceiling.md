---
name: number-game-ceiling
description: Rule+interval+digit concept model is near its ceiling; single-param tweaks gain only 3-6 nats in-sample
metadata:
  type: project
---
Round 1 (iter_1, candidate_5) MAP screens with equal-weight 44-rule + uniform-interval model (in-sample NLL 912): free example-noise 908.5, lapse 907.9, noise+lapse 906.3, logit response map 908.6, size-principle exponent 911.6, Erlang interval prior 913.5 (uniform is better), rule-family weights about 911, free 45 weights 893 (but the extra params cost it on LOO). Calibration by predicted-p bin is already good. Biggest per-set residuals: 83,38,58,68 (scattered 20-40% yes in the 50s-70s), 85,19,91 (odd numbers overpredicted), 14,47.
Submitted `equal_prior_concepts_with_guess_lapse` (the lapse model with one shared rule prior): smoke loglik -905.8.

**Why:** the next improvement probably needs a new mechanism, not another response or prior parameter. Builds on [[digit-concepts-key]].
**How to apply:** don't spend rounds on noise/lapse/exponent tweaks; look at the residual sets above.

Round 2 cand_5 (2026-10-30) numpy MAP screens on the incumbent (NLL 904.5): strong/weak sampling mix 904.5 (no gain), bonus for round-boundary (decade) intervals 904.5, lapse drifting with trial_index 904.3, guess-yes drift 904.5, rule prior ∝ size^-g 903.3 (g≈0.25), all-hypothesis prior ∝ size^-g 902.7. Submitted `specificity_biased_rule_prior_lapse` (rule-only version, smoke loglik -904.7). Data = 5 sessions × 450 trials, contiguous in the CSV (trial_index resets), n=5 per cell, so most residual is binomial noise. Reusable screener: a numpy set × hypothesis log-lik matrix with Nelder-Mead runs in about 1 min per variant.
