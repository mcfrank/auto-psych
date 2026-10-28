---
name: generative-top-marking
description: exp2 iter_3 c4 replaced constant top_bonus with generative top-marking credit log1p(odds*share_at_max/p_draw(width)); check outcome
metadata:
  type: project
---
2026-10-19, exp2 iter_3 candidate_4 (`top_marking_range_sampling`): the incumbent upper_edge_marking_ranges, with its constant top_bonus replaced by a non-nested, width- and set-size-dependent credit. A range ending at the max example gets log1p(top_odds * (fraction of examples == max) / ((1-eps)/w + eps/100)). It's a small edit, using pm.Data top_share per set. I chose it to get past the novelty gate, since nested tweaks collapse ([[graded-top-edge-duplicate]]).
Check passed: smoke LL -2054.7 (incumbent about -2057), 163 s per sampling run.
**Why:** Records which form of top-edge marking was tried.
**How to apply:** Check its admission and LOO before trying other functional forms of the edge mechanism. See [[novelty-gate-tiny-tweaks]].
