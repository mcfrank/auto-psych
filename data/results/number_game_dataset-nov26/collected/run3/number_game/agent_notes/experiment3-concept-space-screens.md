---
name: experiment3-concept-space-screens
description: iter_4 cand_0 MAP screens of added concept families on the exp3 incumbent magnitude_threshold_concepts — all ~null; what carries the leak
metadata:
  type: project
---
iter_4 cand_0 (2026-11-03), incumbent magnitude_threshold_concepts MAP loglik -3958.2. Added concept families (own prior share, Beta(1,9)) at MAP:
- residue classes r mod k (k 3-12 except 10, r>=1; "arithmetic progressions") with lone-example discount: -3957.4 (+0.8, share .03, RMSE vs incumbent .007) -> SUBMITTED as arithmetic_progression_concepts (check OK, smoke ll -3984.7, 193 s per 100/100 chain).
- the same residues also taking part in explaining away (nested-rule preemption): -3973 (WORSE by 15; preemption drops .78 -> .10).
- examples' span [min,max] concept: +0.4; rules-need-majority-of-examples penalty: 0 (MAP penalty 0); digit-sum = s: 0; decade unions (1-3 decades): 0; multiples of 13-50: +0.6 to +1.0; divisors-of-x: +1.4, but only by absorbing lone example "1" (artifact).
- Decomposition at MAP: lambda_0 .97, two-sided intervals carry ~nothing; mixed-parity pairs' out-of-range leak (~.25) comes from rules covering ONE of the two examples (tempered beta ~.56 makes strays cheap) plus "above x" thresholds (weight e^3.8, acting as a yes-floor).
**How to apply:** concept-space additions are saturated for this data; aim at evidence strength / stray handling or readout instead. Scratch: experiment3/model_loop/iter_4/candidate_0/scratch (mapdec.py decomposition, postdump.py + show.py per-set top hypotheses).
Related: [[experiment3-data-patterns]], [[experiment3-threshold-concepts]]
