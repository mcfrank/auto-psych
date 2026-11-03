---
name: exp2-iter3-open-ended-heading
description: Exp2 iter-3 cand_3 — heading-direction graft onto digit_sum_open_ended_range_discount, smoke -1925.7, submitted
metadata:
  type: project
---
Exp2 iter 3, candidate_3 (2026-10-30), a refinement slot on the incumbent digit_sum_open_ended_range_discount (-1928.2). The critique found 0/8 significant.
The change grafts heading_direction_range_extrapolation's last-step heading factor onto the incumbent. It reuses the Taylor-moment code, with intervals grouped by (width, open) into 200 groups. Coefficient: w^gamma * exp(log_open*open) * beta^k/k!.
Named `open_ended_ranges_heading_direction`. Smoke loglik -1925.7, versus -1927.1 for heading alone. About 20 s per 200 iterations. Siblings had not yet written their files when I checked, so a duplicate graft is possible ([[exp2-iter2-digit-sum-location-prior]]).
See [[exp2-iter2-heading-direction]].
