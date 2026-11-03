---
name: exp2-pure-rule-induction
description: Exp2 iter-0 cand_1 — rules + catch-all, no intervals: MAP NLL 2352 vs incumbent 1928 (~420 nats worse), submitted as high-variance probe
metadata:
  type: project
---
Experiment 2, iter 0, candidate 1 (2026-10-30). Submitted `pure_rule_induction_no_ranges`: the 45 rules sampled without replacement, plus a catch-all "any number 1-100" concept in place of the 5050 intervals, with a guess lapse. MAP NLL was 2351.8, against 1928.5 for the WOR incumbent, so the intervals are worth about 420 nats on the exp2 data. The smoke loglik was -2353.
**Why:** the brief asked for a simpler, higher-variance candidate.
**How to apply:** don't re-propose any model that drops ranges. Magnitude concepts are essential. See [[digit-concepts-key]].
