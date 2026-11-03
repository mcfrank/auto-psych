---
name: previous-set-concept-intrusion
description: exp3 iter_4 c2: inter-trial interference (prev other set's concept judged with w) on variable_min_max: w→.004 free; fixed .02/.04 cost 5/13 MAP ll; submitted Beta(10,990)
metadata:
  type: project
---
exp3 iter_4/candidate_2 (2026-11-02) is previous_set_concept_intrusion. Mechanism: p = (1-w) p(cur set, t) + w p(most recent *other* set in the session, t). Sessions are 450-row blocks, and trial_index resets at each new session. Rows are in trial order, and the same set repeats on consecutive rows 10% of the time.
- Implementation: add the (prev set, target) cells to the cell list in prepare_observed, which costs almost nothing.
- Free w goes to .004, so it is a dup (pool RMSE .0014).
- Fixed w=.02 gives MAP ll −3323.2, against −3317.9 for the incumbent, and pool RMSE .0063. Fixed w=.04 gives −3330.5.
- So there is no inter-trial carryover in exp3 (this agrees with the residual-vs-previous-response correlation of ≈0 in [[exp3-decision-rules-collapse]]).
- Submitted with a stated prior Beta(10,990).
- The novelty pool has no trial_index and is sorted by set, so "the previous other set" is just the previous set in the pool's order.
**How to apply:** sequential or inter-trial families are ruled out for exp3; don't retry priming variants.
