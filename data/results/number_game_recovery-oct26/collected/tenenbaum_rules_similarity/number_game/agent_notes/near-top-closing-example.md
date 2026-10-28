---
name: near-top-closing-example
description: exp3 iter_0 c3: incumbent's top-marking credit spread over ranges ending 0..9 above max (geometric decay rho); targets max-last over-sharpening critique; slow check
metadata:
  type: project
---
2026-10-22, exp3 iter_0 candidate_3 (`near_top_closing_example`): this is a copy of last_example_marks_range_top with one change. The marking credit is (1-rho) rho^d for a range whose top lies d = 0..9 above the largest example, and rho ~ Beta(1,3). The intervals are partitioned into 11 distance bins, where the incumbent had 2. It was motivated by the critique above_below_asym_max_last (p=0.018, q=0.144): the model over-sharpens the top edge. The first version used a gather over posterior[cell_set] and took 316 s per smoke run. Switching to blocks of 8 targets per row with batched_dot (from [[top-marking-uncertain-sampling]]) brought it to 0.066 s per gradient and 100-115 s per smoke run. Check OK: smoke LL -3094.3.
**Why:** It is a softer top edge, and the incumbent is its rho=0 special case.
**How to apply:** Check the LOO and the admission speed. If it is too slow, reduce NEAR to 5. Always use batched_dot blocks for this model family. Related: [[last-example-top-marking]], [[exp3-example-rejection-critique]]
