---
name: exp2-iter4-size-principle-rules-only
description: Exp2 iter-4 cand_0 repair — strong/weak mixture was dup of margin (RMSE 0.0017); weak sampling for ranges only (0 new params) smoke -1988, ~60 nats worse, submitted
metadata:
  type: project
---
Exp2 iter 4, candidate_0_repair_1 (2026-10-30). The strong/weak sampling mixture on the margin incumbent was rejected as a near-dup: the strong share went to about 1, giving RMSE 0.00165.
The repair, `size_principle_for_rules_only`, uses the size principle for rules and digit sums and weak sampling for intervals, with the margin incumbent's range prior. Smoke LL is -1988.2 (incumbent about -1926), the smoke fit took 26 s, and it was submitted as a high-variance loser.
**How to apply:** ranges need strong sampling on exp2, worth about 60 nats. A mixture whose weight sits at the nested boundary gets rejected as a dup, so make the alternative structural. See [[exp2-iter4-strong-weak-sampling]].
