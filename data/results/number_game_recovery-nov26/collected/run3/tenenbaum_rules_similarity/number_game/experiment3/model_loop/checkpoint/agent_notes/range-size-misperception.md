---
name: range-size-misperception
description: exp2 iter_3 c2: ranges' felt size = L**rho in the likelihood only (rules true size); smoke OK, lppd -2268.4
metadata:
  type: project
---
experiment2 iter_3 candidate_2 (2026-10-30), exploratory "normative + one distortion" slot. Incumbent beta_share_width_exception_concepts with interval likelihood ALPHA/L**rho + 0.001, rho ~ LogNormal(0,0.3). Implemented by grouping intervals per (set, size L, k examples inside) — tensor (sets,100,5) of log counts plus per-cell share containing target; smoke fit 41 s/200 draws, lppd -2268.4. rho posterior not inspected; risk of novelty dup if rho≈1 (cf. evidence exponent null in [[experiment2-null-refinements]]).
**How to apply:** check its admission outcome before retrying likelihood-side size distortions.
