---
name: range-band-single-leniency
description: exp3 iter_0 c5: n=1 leniency grafted on range_tightened_magnitude_band; smoke loglik -4695.7 vs base -4717.8 (+22)
metadata:
  type: project
---
Exp3 iter_0 candidate_5 (2026-11-02), range_band_single_leniency. In exp3, one-example sets get 0.58 yes against 0.40 for multi-example sets, and the n=1 yes rate is flat with distance from the example (0.52-0.61 across 10-wide bins). The incumbent span_interval_similarity fails single_vs_multi_yes_gap and single_example_distance_gradient. Adding a logit shift for n=1 sets to range_tightened_magnitude_band: smoke -4695.7 vs -4717.8 for the base.
**Why:** single-example leniency keeps paying off across experiments.
**How to apply:** if the incumbent lacks n=1 leniency, it's a cheap graft. See [[band-plus-leniency]], [[single-example-leniency]].
