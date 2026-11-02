---
name: squares-cost-split-finding
description: On the ends-bonus + exponent-cost model, a separate cost for perfect squares gained ~4 nats MAP (squares ~5.4, powers ~2.5); parity ranges and upward interval bias gained ~0-1
metadata:
  type: project
---
Round 4 repair (2026-10-30), cell_1/experiment1, iter_4/candidate_5_repair_1. The base was final_digit_multiplicative_concepts, with MAP loglik -799.5.
- Splitting the exponent cost so perfect squares pay their own cost: -795.5. The squares cost fit at about 5.4, while powers and cubes fell to about 2.5. Grouping squares with cubes gave -796.1. Submitted as `costly_squares_final_digit_concepts`; smoke-fit loglik -799.6.
- Parity-restricted intervals: the share fit to 0, so no gain.
- An interval prior tilted by interval midpoint (the "upward" critique stat): +1.2, and the tilt fit slightly downward (-1.3).
- The first attempt (ends bonus + exponent cost on the final_digit_salient base) was rejected as identical to final_digit_multiplicative_concepts (RMSE 0.0004). The same mechanism reached from two different bases counts as a duplicate.
- The largest remaining misfits are on these sets: 92,82,27,42; 22,62,26,82; 83,38,58,68 (85/87/89 yes); 20,32,92 (people answer multiples of 4); 100,1,16; 8,64.

**Why:** saves re-screening, and shows that how the exponent cost is broken down still matters.
**How to apply:** screens are in iter_4/candidate_5_repair_1/scratch (sq.py, par.py, screen.py, resid.py). See [[ends-bonus-plus-exponent-cost]] and [[exponent-concept-cost-finding]].
