---
name: family-pooled-reliability-graft
description: Grafting family_pooled_rule_prior's hierarchical rule weights onto uncertain_reliability_noise_concepts — smoke OK but deep NUTS trees
metadata:
  type: project
---
iter_3/candidate_3 (2026-10-30): family_pooled_uncertain_reliability = incumbent (Beta-marginalised reliability) + non-centred family-pooled rule prior (softmax(family_level[f] + tau*rule_dev)). Check OK, smoke loglik -834.1 (incumbent smoke about -819 in-sample MAP; not comparable). Same 6 ms/grad as the incumbent, but the smoke fit took 290 s per 200 iters (~240 grads/iter: the tau funnel defeats the fixed Hessian mass matrix).
**Why:** the warm-start dense metric fits a Gaussian at the mode; a hierarchical scale near 0 makes the trees deep.
**How to apply:** if this is rejected as too slow, try a smaller tau prior or a fixed tau. Compare [[number-game-refinement-findings]] and [[warm-start-saddle-priors]].
