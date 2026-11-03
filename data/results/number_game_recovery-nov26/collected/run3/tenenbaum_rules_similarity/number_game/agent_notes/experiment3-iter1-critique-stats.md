---
name: experiment3-iter1-critique-stats
description: exp3 iter_1 critique of kind_uncertainty_shifted_criterion — 8 new PPC stats + observed values; numpy MAP refit script for residual scans
metadata:
  type: project
---
iter_1 critique (2026-10-31), incumbent kind_uncertainty_shifted_criterion (iter_0 got 0/8; closest multiple_rule_vs_span_preference p=0.08).
New stats, observed (rough binomial null from a numpy MAP refit, ignoring posterior spread): decade_shared_contrast -0.062 (null -0.040, z≈-2); target_100_yes 0.46 (null 0.39, z≈1.9); participant_set_consistency 0.258 (null 0.265, z≈-1.5, people LESS consistent within a set than the model); prime_set_two_target 0.08; rule_set_member_far_outspan 0.73; odd_set_even_target_yes 0.58; single_example_multiple_yes -0.01; late_session_yes_drop -0.07.
Model fits marginals (set size, span, parity, blocks, halves) almost exactly; residuals look like noise.
Refit script: /tmp/crit3/fit.py (Powell over lam, mean_share, conc, 8 family logits, shift; ll -3646 in ~min) writes per-row p to /tmp/crit3/pred.csv; /tmp/crit3/sim.py <stat files> gives quick z.
**Why:** avoid re-probing; see [[experiment3-iter0-critique-stats]].
**How to apply:** read iter_1/critique/ppc_results.json for hits before choosing iter_2 stats.
