---
name: position-weighted-top-marking
description: exp3 iter_0 c5: top-marking where P(max is the mark) = last_bias*[max last] + (1-last_bias)/n-share; smoke LL -3094.7, 187 s/run
metadata:
  type: project
---
On 2026-10-22, exp3 iter_0 candidate_5 (`position_weighted_top_marking`) was a copy of top_marking_range_sampling with one change. The marked share is last_bias*top_is_last + (1-last_bias)*top_share, with last_bias ~ Beta(2,2). This nests both the incumbent (last_bias=1) and its parent (last_bias=0). It targets the critique above_below_asym_max_last: the incumbent over-sharpens the top edge (q=0.14, so it did not survive FDR). The check passed, with smoke LL -3094.7 and about 187 s per sampling run.
**Why:** It sits between the two order rules. The risk is novelty: if the posterior puts last_bias near 1, its predictions match the incumbent's.
**How to apply:** Check the outcome and last_bias. If it was rejected as a duplicate, the order cue is already settled. Related: [[last-example-top-marking]], [[novelty-gate-tiny-tweaks]]
