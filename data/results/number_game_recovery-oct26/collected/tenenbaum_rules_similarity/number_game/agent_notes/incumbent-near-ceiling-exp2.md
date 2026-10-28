---
name: incumbent-near-ceiling-exp2
description: Exp2 screen of prime_concept_excludes_one refinements found ≤2 nats available; noise/tempering/shape/lapse all null
metadata:
  type: project
---
2026-10-19, exp2 round 1 (candidate_5): a fast numpy harness (fixed kind weights, rules + plain intervals, scipy optimize) on prime_concept_excludes_one gave a log-lik of about -2054. With each change refit: free epsilon, evidence tempering, gamma shape and response lapse each moved it by ≤0.1 nat. Per-rule free weights gave +14 nats for 37 extra parameters (overfitting). The decade bonus is pushed strongly negative. The only structured residual found was fewer "yes" answers 1-3 above the largest example (z≈-3.3); a one-sided upper-edge bonus gained about 1.6 nats (submitted as upper_edge_marking_ranges). 10 participants of 450 trials each (trial_index resets), in two stimulus blocks; there are no participant or sequential effects. The data look close to simulated from this model family.

**Why:** It saves rescreening the same ideas.
**How to apply:** Expect small gains from parameter-level refinements; look for structurally new stimuli/sets instead. PyMC find_MAP gets stuck on these models (-2574 vs -2054), so use short NUTS or the numpy harness. See [[prime-rule-includes-one]].
