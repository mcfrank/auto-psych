---
name: pedagogical-sampling-exp3
description: exp3 iter4 cand2 — teacher-chosen (distinctiveness-weighted effective size) likelihood ~null, +0.8 MAP; full 100x100 interval grid trick
metadata:
  type: project
---
2026-10-31, exp3 iter_4/candidate_2 (`pedagogical_example_selection`), on forward_continuation_response_bias (MAP loglik about -3411.6).
- Pedagogical sampling in closed form: a teacher with P(x|h) ∝ [x∈h]·w(x)^α, where w(x) = 1/(size-weighted hypothesis mass containing x). Then w(x_i) cancels, and the effective size is Z_h = Σ_{x∈h} w^α. α=0 gives back the plain size principle exactly.
- Results by how distinctiveness w is defined:
  - rules + intervals: α -0.08, loglik -3411.9 (null)
  - rules only: α -0.11, -3410.8
  - intervals only (number-line edges are diagnostic): α +0.10, -3410.9. Submitted.
- Check OK, smoke loglik -3419.8, 3 min.
- RMSE against α=0 on the novelty pool is 0.008, so it should clear the 0.002 duplicate gate.
- Speed: the 100x100 grid version took 0.106 s per gradient against the incumbent's 0.026, and the check timed out.
- Exact fix: in a set, an interval's example count k is constant over at most 25 rectangles of (a, b), and _blocks() enumerates them. Keep tables per (n, k) of the "intervals containing [a, b]" sum (cumsum over a, reverse cumsum over b), each with its own max shift. Each rectangle is then 4 corner gathers, at 0.020 s per gradient.
- Gotcha: a plain prefix sum from (1, 1) suffers catastrophic cancellation (p errors up to 0.4), and so does a single shift that includes the k>n entries.
**Why:** another null on exp3: the strong-sampling likelihood is fine.
**How to apply:** don't revisit teacher or sampling-assumption variants. Scripts are in iter_4/candidate_2/scratch (map.py, rmse.py). Related: [[exp3-iter1-exploratory-screens]].
