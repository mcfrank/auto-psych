---
name: incumbent-null-changes-round3
description: Changes to erlang_interval_simplicity_search that fit to ~0 at MAP (novelty-rejected risk) vs the digit-notation discount that gained ~21 nats
metadata:
  type: project
---
Round 3 (2026-10-30), cell_1/experiment1 MAP screens on erlang_interval_simplicity_search (baseline LL -873.7):
- divisibility salience bonus: fit ~0, rejected as near-duplicate (RMSE 0.0009).
- fitted stray rate: stays at 0.10, no gain. "Any number 1-100" concept: weight -> 0. Decade-aligned interval bonus: +1.5 only. Extra cost on contains-digit rules alone: +2.6.
- Separate fitted log-prior shift for all 20 digit-notation rules (ending-in/contains): -852.3 (+21), fits as a BONUS (~2) while primitive cost rises to ~4.6. Submitted as `notation_cheap_simplicity_search`. It makes shared-divisor under-extension worse (0.56 vs obs 0.71).

**Why:** a change whose parameter fits to ~0 gets rejected by the novelty gate.
**How to apply:** screen at MAP first, and only submit changes whose parameter moves away from its null. See [[exponent-concept-cost-finding]].
