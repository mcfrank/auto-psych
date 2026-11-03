---
name: concept-space-saturated-exp3
description: exp3 iter0 cand1 screen — every new concept family on the repdigit incumbent gains ≤2 nats MAP; digit-reversal pairs chosen (+1.5)
metadata:
  type: project
---
2026-10-31, exp3 iter_0/candidate_1 (`digit_reversal_pair_concepts`). On 6750 rows, the incumbent repdigit_pattern_chart_concepts has MAP nll 3465.8 (lapse→0, beta≈0.86).
Gains from one-parameter additions: reversal pairs {ab,ba} 1.5; contains-d bonus 1.9; big-step multiples 13..50 1.2 (log-graded 0.6); decade rules 0.9; digit-set pairs 0.2.
The rest came out at 0: digit sum, near-multiples, "any number", two-digit, halves, digit patterns (odd-odd, ascending), first ten multiples, contains-either-of-2, bounded APs (init k=14, or Powell diverges), and a 5-aligned interval bonus.
The singleton critique: the model predicts near 0.49 / far 0.24, observed 0.375 / 0.31. No concept addition fixed it.
Pool RMSE vs incumbent: rev 0.016, big 0.018. Both pass novelty.
Scratch: iter_0/candidate_1/scratch/w/scr2.py (full 5050-interval numpy, Powell).

**Why:** future concept-space slots should look elsewhere (response stage, interval mechanism for singletons).
**How to apply:** don't re-screen these families. Related: [[prime-salience-finding]], [[big-step-multiples-finding]].
