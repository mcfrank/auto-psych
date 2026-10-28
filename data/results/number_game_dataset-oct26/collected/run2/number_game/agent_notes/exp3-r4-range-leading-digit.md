---
name: exp3-r4-range-leading-digit
description: Exp3 r4 — leading-digit graft onto sample_size_neglect_upward_start beats incumbent MAP by 2.4; log-scale range inference hurts badly
metadata:
  type: project
---
Exp3 round 4 (2026-10-18), MAP log-lik: incumbent leading_digit_string_similarity −4630.2; sample_size_neglect_upward_start −4638.9; the same model plus a leading-digit dimension −4627.8 (submitted as range_inference_leading_digit, iter_4/candidate_5). Predictions correlate 0.996 with the incumbent's (max |Δp| 0.06).
Measuring range inference's gap and span on a log scale: −4661.7 (−23), or −4645 with the leading digit added. Don't log-compress the range model to chase the log_magnitude critique.
**Why:** the leading-digit gain carries over between the span-filling and range-inference size accounts (+11 in each).
**How to apply:** the scripts are in iter_4/candidate_5/scratch (map.py, varA–C). See [[leading-digit-similarity-gain]].
