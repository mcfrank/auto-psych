---
name: prime-salience-finding
description: exp2 prime-numbers prior bonus on factor_graded incumbent +20 loglik MAP (bonus~2.1); new concept families (residues, progressions, digit sums, rule∩range, ending unions) all null
metadata:
  type: project
---
2026-10-30, cell_1/experiment2 iter_1/candidate_2 (`prime_salient_concepts`). Incumbent MAP loglik -2173.5.
- Fitted prime_bonus: -2153.4 (+20; bonus 2.06). The gain comes from 2,47 / 53,2 / 67,31,17,61. Making primes cost one primitive with no new parameter gave only +10.
- Extra concept families, each with one fitted cost, came out at about 0 loglik: residue classes mod 3-12, unbounded progressions (1023 rules), digit sum = k, "ends in d1 or d2", parity/multiples ∩ decade ranges or half-bounded ranges. Repdigits 11,22,...,99 gave +6.
- Scratch fam.py appends EXTRA rules through string patches of the incumbent source. Note: the extra rules there cost -2*cost, not -cost.

**Why:** shows which parts of concept space are worth exploring on exp2.
**How to apply:** don't re-screen these families. See [[factor-graded-multiples-finding]].

Update 2026-10-30 (exp2 iter_2 cand_4): grafted prime_bonus onto incumbent hundred_chart_row_ranges as `prime_salient_hundred_chart`; check OK, smoke loglik -2141.8, ~90s for 100+100 draws. Motivated by critique prime_set_prime_target_lift (z=4.3, survives FDR).
