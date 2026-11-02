---
name: range-restricted-rules
description: exp2 iter_1 cand_2: rule∩interval concepts on the incumbent; lambda_c→0.014 (prior-held), +0.2 MAP, RMSE 0.0075; sparse CSR over (m,f,k,s) made it 4 ms/grad
metadata:
  type: project
---
experiment2 iter_1/candidate_2 (2026-11-01): range_restricted_rule_concepts = first_example_anchored_reliability plus "rule cut to an interval" concepts (proper, non-empty cuts only), share lambda_c ~ Beta(2,38), split by rule weight then equally over cuts. MAP ll -2031.25 vs -2031.46 for the incumbent; lambda_c=0.014; p_yes RMSE vs the incumbent was 0.0075 on the training cells.
Speed: dense (sets,rules,400) dots took 35 ms/grad. Flattening the count tables over (m, first, k, size) with m included, then one ps.structured_dot mat-vec with lik_table.reshape(-1), gave 4.2 ms/grad, 15 steps/iter, 0.13 s/iter.
**Why:** this is the same as [[concept-additions-collapse]]: the data don't ask for new concepts.
**How to apply:** the m-in-column CSR trick applies to any table model. Check the ELPD outcome in iter_1 admission.
