---
name: exp2-iter0-distortion-screens
description: Exp2 iter-0 cand_0 MAP screens on WOR incumbent (NLL 1928.49) — target Weber noise, rule-check errors, strong/weak mix, sample-size neglect all null or much worse; interval-only width power prior +1.2 submitted
metadata:
  type: project
---
Exp2 round 0, candidate_0 (2026-10-30). Numpy MAP screens on `without_replacement_size_principle` (NLL 1928.49; lapse 0.006, guess_yes about 1, lambda_0 0.34):
- Magnitude noise on the target (Weber or constant sd): the noise goes to 0, a tie.
- Errors when checking rule membership (perceived extension (1-e)m + e(1-m)): e=0.002 costs 10 nats and e=0.01 costs 80. Strongly ruled out.
- A mixture of strong and weak sampling assumptions: theta goes to 1 (pure strong sampling). A tie.
- Sample-size neglect (LL × n^(b-1)): sharply pinned at b=1, and b=0.85 or 1.15 costs 20-40 nats.
- Erlang interval-size prior: -2.8. A power-law width prior on intervals only (rules kept flat): gamma -0.12, NLL 1927.33 (+1.2), RMSE 0.010 vs the incumbent.

Submitted `narrow_range_specificity_bias` (the interval-only power prior). Intervals are grouped by (set, width) in prepare_observed. The smoke fit takes 20 s per 200 iterations, and the smoke loglik is -1929.9. The scripts are in experiment2/model_loop/iter_0/candidate_0/scratch/.

**Why:** the model is at the ceiling ([[number-game-ceiling]]). Normative sample-size use and strong sampling are well identified.
**How to apply:** don't re-screen these. See [[exp2-iter0-critique]].
