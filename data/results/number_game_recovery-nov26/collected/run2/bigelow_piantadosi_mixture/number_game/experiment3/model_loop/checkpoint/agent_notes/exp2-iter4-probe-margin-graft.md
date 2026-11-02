---
name: exp2-iter4-probe-margin-graft
description: Exp2 iter-4 cand_5 refinement — margin-beyond-examples prior grafted onto probe_as_weak_example; critique 0/8 significant
metadata:
  type: project
---
Exp2 iter 4 cand_5 (2026-10-30), refinement slot. Critique of the margin incumbent gave 0/8 significant, so there was no gap to target. I chose probe_as_weak_example (rank 3) and grafted the margin Taylor-moment prior onto it. The probe boost for intervals is dot(member_moments, coef * (1+90/w)^w_probe per group) / evidence. Submitted as `probe_with_margin_range_prior`; smoke LL -1926.0 (passed), each 100+100 chain took ~75 s. I did not run a MAP screen. I chose probe over a heading graft because the incumbent slots were likely to graft heading themselves (duplicate risk). See [[exp2-iter3-probe-as-weak-example]], [[exp2-iter3-margin-beyond-examples]].
