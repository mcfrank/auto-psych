---
name: exp3-iter4-span-filling-evidence
description: Exp3 iter-4 cand_3 — refinement of position_decaying incumbent; interior (span-filling) example evidence weight gamma≈-0.06, +0.58 MAP; open-ended +0.2, rule breadth +0.06; submitted
metadata:
  type: project
---
Exp3 iter 4, candidate_3 (2026-10-31), refinement slot on position_decaying_example_evidence. The base MAP is -3189.70 with free rho (≈0.014).

MAP screens (script at iter_4/candidate_3/scratch/map.py):
- Weight exp(-gamma) on an example lying inside the span of the earlier examples: -3189.11, so +0.58. Gamma ≈ -0.06, meaning interior examples count slightly MORE. That runs against the prefix-broadening critique (31,3,1→31,3,1,15). Observed-cell RMSE vs gamma=0 is 0.0058.
- Open-ended range discount (class key width+100 flag): +0.2.
- Rule breadth size^h: +0.06.

Submitted `span_filling_example_evidence`. Smoke -3192.1, 18 s.

The prefix critique rests on one set with 5 responses per cell, so it is probably noise. See [[exp3-iter4-critique]], [[exp3-iter3-position-decaying-evidence]].
