---
name: iter4-occam-wor
description: Iter-4 cand_5 refinement slot — occam all-concept size prior + without-replacement sampling, smoke loglik -903.5, ~10 min fit
metadata:
  type: project
---
Iter 4, candidate_5 (2026-10-30). Submitted `occam_prior_without_replacement`. It takes `occam_specificity_prior_all_concepts`, which uses a shared |h|^-g prior over rules and intervals, and changes one thing: example likelihoods use the WOR `set_log_lik` copied from the incumbent. Parameter count is the same as the incumbent (lambda_0, g, lapse, guess_yes). Smoke loglik was -903.5, the best smoke figure so far. A 200-iteration smoke fit takes about 17 s, so the full 7000-iteration fit should take about 10 min.

**How to apply:** check its ELPD next round before trying more prior×sampling combos. See [[iter3-noise-ceiling-wor]] and [[number-game-ceiling]].
