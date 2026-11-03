---
name: exp3-iter2-critique-stats
description: exp3 iter_2 PPC stats on set_size_evidence_privileged_fives; observed values (upward asymmetry 0.14, singleton multiples 0.22)
metadata:
  type: project
---
2026-10-31, exp3 iter_2 critique of set_size_evidence_privileged_fives. iter_1's 8 stats were all null (singleton near-far p=0.066 was the closest), so I picked new angles. Observed values: on 2+ example sets, yes above the max minus yes below the min is 0.136. On singletons (example 2-50), multiples minus non-multiples is 0.217. Yes rate on 3-example sets minus 4-example sets is 0.04. Between-set variance of yes rate is 0.044. Within-set target-magnitude slope is -0.045. Nearest-example gap (distance 1-3 vs 4-15) is 0.025. Round targets when no example is round: -0.057. Same tens digit with no shared final digit: 0.046.
**Why:** these are candidate discrepancies. ppc_results.json in iter_2/critique says which are significant.
**How to apply:** if the upward asymmetry is significant, look again at the forward tilt ([[forward-counting-tilt-finding]]) on exp3. Related: [[exp3-iter1-critique-stats]].
