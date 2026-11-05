---
name: coincidence-rule-violation-route
description: exp3 iter3 c3: breaking the set's most specific coincidence costs fixed log-support on parity_preserving_decisive_route, +38 nats MAP; smoke -4036.0
metadata:
  type: project
---
exp3 iter_3 candidate_3 (2026-11-02): incumbent parity_preserving_decisive_route plus a flag for targets that break the set's most specific coincidence (same digit > all mult-5 > same parity, 2+ examples). Each flagged target loses c_viol from log_summed, before the decisive slope. MAP -4027.9 vs incumbent -4066.0 (c_viol≈0.43). Smoke fit -4036.0.
Variants that did worse: penalty scaled by coincidence -4036.9, penalty as a decision offset outside the slope -4037.3, summing every broken shared property -4039.4. Incoherent polarization got worse (pred 0.112 vs observed 0.216).
**Why:** targets the critique stat coherent_set_far_violation_rate (q 0.059); follows [[exp3-iter3-critique-stats]], [[parity-preserving-decisive-route]].
**How to apply:** the polarization of single-example and incoherent cells is still unfixed. It is target-level: multiples get too low a p (7→14, 33→99) and far unrelated targets sit near 0.3 where people answer 0.
