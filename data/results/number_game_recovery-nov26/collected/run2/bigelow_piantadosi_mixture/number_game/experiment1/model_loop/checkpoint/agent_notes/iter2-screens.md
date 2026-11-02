---
name: iter2-screens
description: Round 2 MAP screens on equal_prior_concepts_with_guess_lapse (NLL 904.5): new concept families and response tweaks gain at most ~1.8 nats
metadata:
  type: project
---
Round 2, candidate_4 (2026-10-30). Baseline in-sample NLL 904.5. Adding decade, repdigit, digit-sum, single-digit or "contains both digits" rules gave no gain. Logit sharpening gave 0. Free example noise gave 902.7, but lapse then drops to about 0.008, so it would duplicate `digit_concepts_inferred_outlier_rate`. A rule prior proportional to |rule|^gamma gave 903.3 with gamma≈-0.24, a slight preference for narrow rules. I submitted that as `rule_breadth_prior_concepts`. Its smoke loglik was -904.8 and the fit took about 2 s per 200 iterations.

**Why:** this confirms [[number-game-ceiling]]. Most of the residual is 5-response-per-cell binomial noise.
**How to apply:** don't re-screen these concept families. Further gains are unlikely without participant-level structure, and the CSV has no participant column.

Round 2 cand_5 repair (2026-10-30): `specificity_biased_rule_prior_lapse` was rejected as a near-duplicate of `rule_breadth_prior_concepts` (RMSE 0.0003). The repair is `occam_specificity_prior_all_concepts`, which applies the same |c|^-g to intervals too, normalised within each class and collapsed to 100 size groups per set. Smoke loglik was -904.4, about 16 s per 200 iterations. Applying g to intervals moves p_yes by RMSE ≈ 0.01 at g = 0.1, which clears the 0.002 novelty gate.
