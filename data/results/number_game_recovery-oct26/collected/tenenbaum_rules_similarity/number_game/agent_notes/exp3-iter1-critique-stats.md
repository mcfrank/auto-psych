---
name: exp3-iter1-critique-stats
description: exp3 iter_1 critique (same incumbent as iter_0): 8 new stats on max-not-last asymmetry, far generalization, parity, span width, lag-1, decade, gradient, round targets
metadata:
  type: project
---
Exp3 iter_0 critique was run on the same incumbent (last_example_marks_range_top): only above_below_asym_max_last was significant (obs -0.11 vs null -0.22, p=0.018; people extend upward MORE than the model allows after a max-last set). Example-rejection stats were null.
iter_1 (2026-10-22) submitted: above_below_asym_max_not_last, far_outside_span_yes, parity_match_outside_span_gap, inside_span_width_corr, lag1_response_autocorr, singleton_same_decade_gap, outside_gradient_steepness, round_target_outside_gap. They share a vectorised parse preamble (str.split expand=True to a NaN-padded matrix, about 0.015 s per call).
**Why:** avoid re-proposing ideas already tested. **How to apply:** read iter_1/critique/ppc_results.json before the next critique. Related: [[exp3-example-rejection-critique]], [[last-example-top-marking]]
