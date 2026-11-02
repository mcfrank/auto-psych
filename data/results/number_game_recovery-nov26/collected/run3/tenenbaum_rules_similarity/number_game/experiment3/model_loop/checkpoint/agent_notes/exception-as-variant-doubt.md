---
name: exception-as-variant-doubt
description: exp3 iter_3 c3: incumbent's rule-family doubt with exception family pooled into powers; smoke OK ll −3653.0
metadata:
  type: project
---
exp3 iter_3/candidate_3 (2026-10-31), exception_as_variant_doubt_criterion: [[rule-family-doubt-criterion]] with only the doubt measure changed, so that exception rules (family 7) count as powers (family 3) and the entropy is normalised by log 7. The prior is unchanged.
At a typical parameter setting, its doubt correlates 0.94 with the incumbent's across the 170 exp3 sets. The changes are concentrated in powers-of-2 sets: for 16,32,2,8 doubt falls from 0.29 to 0.00, and for 32,4,16 from 0.30 to 0.03.
Smoke check OK (17 s), ll −3653.0. Novelty was not checked. The probe script is /tmp/c3i3/probe.py.
**How to apply:** if it is rejected as a duplicate, the exception-family part of the doubt does not matter.
