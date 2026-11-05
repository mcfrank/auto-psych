---
name: set-specific-criterion-candidate
description: iter_3 candidate_3 — per-set hierarchical criterion on graded-parity incumbent; MAP NLL 1488 vs 1561, smoke log-lik -1517.9
metadata:
  type: project
---
iter_3 candidate_3 (2026-11-02), `set_specific_criterion_exemplar`: aimed at the set_level_yes_rate_sd critique (q=0.016). Adds a non-centred per-set criterion shift (HalfNormal(1) sd; crit_z shape 200, set index from prepare_observed) to graded_parity_variability_exemplar. MAP data NLL 1488 vs base 1561 (MAP sd ~5 when unconstrained, likely driven by sets with a single target, so LOO/Pareto-k is the risk). Other one-change MLE screens on the base: homogeneity-shifted intercept -8 nats, homog^power digit attention -6, spread-shifted intercept -4, mean vs sum and n-examples intercept 0.
**Why:** set-level variation dominates what's left over. **How to apply:** check its LOO/Pareto-k before building more on it; related [[iter3-critique-stats]], [[fixed-parity-incumbent-repair]].
