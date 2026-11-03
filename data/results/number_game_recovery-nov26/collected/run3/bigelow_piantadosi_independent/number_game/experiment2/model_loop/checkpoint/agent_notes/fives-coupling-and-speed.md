---
name: fives-coupling-and-speed
description: exp2 iter4 cand3 — fives bonus alone best (MAP NLL 2107.9 vs 2126.6); coupling with mult-3 cost or ends-in-0 adds nothing; grid-cumsum/pair-gather speedups are slower
metadata:
  type: project
---
2026-10-30, exp2 iter_4/candidate_3 (`counted_by_fives_privileged`), on base prime_over_odd_reading. Powell MAP NLL:
- incumbent 2126.6; fives bonus alone 2107.9 (bonus≈2.65, row_bonus→9.7, lapse→0)
- one signed "legibility" weight (+5, −3): 2110.8; (+5, −3/6/9/12): 2108.9; fives + ends-in-0: 2108.1
The round-3 fives model (`fives_readable_multiples`) was admitted, then dropped because fold 0 was "too slow" at target_accept 0.99. Four round-3 models were dropped that way, so machine load may be the cause.
Speed per logp+grad: dense dot graph 0.046 s. Gathering per distinct (set, target) pair takes 0.16 s (the advanced-index grad is slow). A 100×100 interval grid with cumsums takes 0.157 s (CumOp and Softmax over 1.8M cells). The dense BLAS dot is the fastest form.

**Why:** fives is the strongest single residual. The mult-3 cost does not stack usefully on it through a shared weight.
**How to apply:** don't try these speed rewrites again; the slowness is in sampling geometry/steps, not the graph. Related: [[fives-bonus-finding]], [[hidden-triples-finding]].
