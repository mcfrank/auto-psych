---
name: parity-salience-asymmetry-exemplar
description: exp3 iter0 c3 repair: even/odd-majority asymmetric parity weight on round_number_averaging incumbent, +7.8 MAP; smoke -4107.5
metadata:
  type: project
---
Repair of the c3 slot (2026-11-02). round_set_kinship_averaging was rejected as a near-duplicate of whole_set_round_category (RMSE 0.0018), so the round-set idea is used up. Change: c_par × exp(parity_asym × set_evenness), where set_evenness is ±1 for an even or odd majority (0 for a tie; a single example's own parity). MAP parity_asym -0.40, meaning odd sets get the stronger parity penalty. MAP loglik -4100.4 vs incumbent -4108.2.
**Why:** aimed at the odd_vs_even_set_parity_violation critique ([[exp3-iter0-critique-stats]]); follows [[progression-lattice-exemplar]] screen.
**How to apply:** the critiques still open are near-target set-size dilution and trial drift.
