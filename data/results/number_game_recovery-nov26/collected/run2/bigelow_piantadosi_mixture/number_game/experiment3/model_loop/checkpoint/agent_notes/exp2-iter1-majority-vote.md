---
name: exp2-iter1-majority-vote
description: Exp2 iter-1 cand_1 decision-rule slot — majority vote over k posterior samples via betainc, smoke -1931.3, ~16 s/200 iters
metadata:
  type: project
---
Exp2 iter 1 candidate_1 (2026-10-30), `sampled_concept_majority_vote`: WOR incumbent inference plus p = I_q((k+1)/2,(k+1)/2), k-1 ~ HalfNormal(4). pt.betainc has gradients and works under NUTS. Smoke loglik -1931.3, tied with the field. Submitted.

**Why:** brief asked for a decision-rule mechanism; as in [[soft-criterion-decision-rule]], response-level rules tie.
**How to apply:** don't re-propose sharpening/criterion decision rules on exp2; the betainc trick is reusable for smooth sharpening.
