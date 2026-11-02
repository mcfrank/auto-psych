---
name: extend-factor-tweaks-duplicate
description: tweaking the incumbent's span-extrapolation factor (e.g. directional skew) gets rejected as near-duplicate
metadata:
  type: project
---
A directional (ahead/behind) skew on span_extrapolation_hesitant_guess_2's reach fitted ~0 and was rejected as a near-duplicate (p_yes RMSE 0.0016 < 0.002), in iter_4 of experiment3. To get past the novelty gate, change the concept-update dynamics instead, e.g. re-placing a kept range guess from the all-example interval posterior (sticky_kind_replaced_range_guess; smoke-fit log-lik -3114.5).
**Why:** the reach factor only touches a few far targets, so its variants barely move predictions.
**How to apply:** don't refine the answer-stage discount in small ways; change something that affects many stimuli. See [[check_candidate_runtime]].
