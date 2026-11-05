---
name: coincidence-violation-exclusion
description: exp3 iter3 c4: fixed similarity cut for targets breaking the set's coincidence on parity_preserving_decisive_route, +37 MAP; smoke -4037.4
metadata:
  type: project
---
exp3 iter_3 candidate_4 `coincidence_violation_exclusion` (2026-11-02). One change to the incumbent [[parity-preserving-decisive-route]]: log_sim -= c_violation·breaks. `breaks` = 1 if the target breaks the most specific shared property of a 2+ example set (same digit, else all ×5, else same parity).
MAP log-lik: incumbent -4066.0; fixed cut -4028.8 (c≈0.38, and c_par_max fell to 0.09); cut scaled by coincidence -4029.1; criterion shift on violators -4037.3; criterion × coinc -4065.8 (no gain). The predicted far-violation rate went from 0.153 to 0.128, against 0.123 observed. Smoke fit -4037.4 vs incumbent smoke about -4073.
**Why:** aimed at coherent_set_far_violation_rate, the strongest iter3 critique. It echoes the pruned shared_regularity_exemplar, but this time on the strong incumbent.
**How to apply:** if it wins, the next critique should check whether the polarization stats moved. Related: [[exp3-iter3-critique-stats]].
