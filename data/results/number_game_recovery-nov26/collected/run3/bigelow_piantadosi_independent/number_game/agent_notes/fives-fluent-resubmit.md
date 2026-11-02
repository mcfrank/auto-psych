---
name: fives-fluent-resubmit
description: exp2 iter4 cand0 resubmitted fives bonus (fives_fluent_skip_counting); MAP +18.7; contrast/opaque variants worse; per-pair gather 4x slower than dense dot
metadata:
  type: project
---
2026-10-30, exp2 iter_4/candidate_0. The fives bonus on prime_over_odd_reading (rule row 18) gives MAP NLL 2107.9 vs base 2126.6.
- A single contrast (+d on mult-5, -d on the other times tables): 2117.9. Opaque-only cost: 2124.2.
- Check printed OK, smoke loglik -2114.1, 4.6 min.
- Hessian at MAP: lbeta/lcost have curvature ~2900 (very stiff). lapse→0, row_bonus and sq_cost sit on flat plateaus. That geometry may explain why round-3 refinements timed out.
- Computing p_yes only over the 863 distinct pairs (a gather of w rows) took 147 ms per gradient vs 35 ms for the dense dot(w, MEMBER). Keep the dense version.

**Why:** the earlier fives model was dropped only for speed, so this tests the idea again.
**How to apply:** if it times out again, try reparameterising (tighter priors on row_bonus/sq_cost, or drop lapse) instead of making the graph faster. Related: [[fives-bonus-finding]], [[hidden-triples-finding]].
