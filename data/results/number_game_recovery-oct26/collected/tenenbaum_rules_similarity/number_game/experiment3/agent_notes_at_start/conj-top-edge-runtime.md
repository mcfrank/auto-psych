---
name: conj-top-edge-runtime
description: Exp2 iter_2 cand_4 applied the top-edge bonus to conjunctions too; doubling the conj columns gives a ~274s smoke fit
metadata:
  type: project
---
2026-10-19, exp2 iter_2 candidate_4 (`top_edge_marking_all_ranges`): this copies upper_edge_marking_ranges and splits the conjunction groups by "interval top == max example", sharing top_bonus. The check passed: smoke LL -2056.7, the 100+100 single-chain draws took 274 s, and the whole check took about 10 min (the incumbent takes about 200 s). Watch the 30-min sampling limit if more columns are added.

**Why:** Gives a cost and benefit baseline for extending the edge marking. **How to apply:** See [[incumbent-near-ceiling-exp2]] and [[write-candidate-first]].
