---
name: exp3-iter4-critique-stats
description: exp3 iter_4 PPC stats on forward_continuation_response_bias; order-split forward lean -0.11, even-set gap 0.38, next-term +0.12
metadata:
  type: project
---
2026-10-31, exp3 iter_4 critique of forward_continuation_response_bias. iter_3's 8 stats were all null. Observed values: forward lean on ascending-shown sets minus descending-shown sets is -0.108, so the lean is stronger when the examples were shown descending. The design has 41 ascending and 39 descending sets. Above-max targets 1-10 above minus more than 20 above is +0.057. Arithmetic next term (last + step) minus other targets is +0.123. Digit reversal is -0.066. Double or half of an example is +0.10. All-even sets, even minus odd targets, is 0.376. Span centre minus edges is 0.003. Same-set lag-1 response agreement is 0.617.
**Why:** these are candidate discrepancies; iter_4/critique/ppc_results.json says which are significant.
**How to apply:** if the order split is significant, make the forward lean depend on the order the examples were shown in. Related: [[exp3-iter3-critique-stats]], [[forward-continuation-response-bias]].
