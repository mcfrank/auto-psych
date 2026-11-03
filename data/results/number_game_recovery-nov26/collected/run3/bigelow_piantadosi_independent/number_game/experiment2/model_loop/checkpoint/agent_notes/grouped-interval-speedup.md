---
name: grouped-interval-speedup
description: exact 2x faster incumbent: sum intervals by (size,row,#examples contained) groups + distinct (set,target) pairs; round-3 refinements were all dropped as too slow in CV folds
metadata:
  type: project
---
2026-10-30, exp2 iter_4/candidate_4 (`fives_bonus_chart_concepts`). All round-3 refinements of prime_over_odd_reading, the fives bonus included, were admitted. They were then dropped, "fold 0 too slow to fit": the CV folds are fitted concurrently at target_accept 0.99 under a 30-minute limit.
The fix is exact. An interval's log posterior depends only on its size, whether it is a whole-rows interval, and the count c of examples it contains. So the 5050 intervals become 200*5 groups, with counts precomputed per set and per distinct (set, target) pair. exp2 has only 863 distinct pairs among 4500 trials. Results: p_yes matches the incumbent to 1e-13, and the gradient takes 18 ms vs 37 ms.
Gotcha: a gather over trials×groups (4500×1000) was slower (98 ms) than the original. Gather over pairs instead.

**Why:** speed now decides admission, not just fit.
**How to apply:** copy candidate.py from that dir as the base for any interval-model refinement. Related: [[fives-bonus-finding]], [[forward-counting-tilt-finding]].
