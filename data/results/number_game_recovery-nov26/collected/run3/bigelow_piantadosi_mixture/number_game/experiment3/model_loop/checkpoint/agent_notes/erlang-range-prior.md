---
name: erlang-range-prior
description: exp3 iter_1 c1: Erlang size prior on magnitude ranges (on range_limited incumbent) loses 4.1 MAP ll, sigma≈39, pool RMSE .019; narrow-span in-hull non-member yes is 0.58 vs 0.17 for wide spans
metadata:
  type: project
---
exp3 iter_1/candidate_1 (2026-11-01), erlang_sized_magnitude_ranges. Motivated by the critique narrow_rule_inhull_nonmember_span_contrast (obs -0.305 vs null -0.196, q .08). In the raw data, rule non-members inside the span get yes 0.58 for spans ≤40 (90 responses) and 0.17 for spans >40 (520).
Change: the range prior ∝ s·exp(-s/sigma), normalised over the 5050 ranges, with log_sigma ~ N(log 20, .75). Range-limited rules keep uniform ranges. To implement it, keep the run_count (sets, run, size) tables and pt.dot them with w(s)·s^-j inside the model. The smoke check passed in 42 s.
MAP ll was -3324.87 vs the incumbent's -3320.79 (−4.1), at sigma 38.6 and lambda_0 .36 (vs .21). Submitted anyway as an exploratory sharp disagreement.
**How to apply:** the Erlang range prior is now ruled out on top of the earlier width-power collapse (see [[normative-distortions-collapse]]). The span contrast probably needs something other than the range size prior.
