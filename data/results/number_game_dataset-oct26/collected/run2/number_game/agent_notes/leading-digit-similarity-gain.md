---
name: leading-digit-similarity-gain
description: Exp3 r3 — adding a leading-digit match dimension to the best model gains ~12.6 nats at MAP
metadata:
  type: project
---
In experiment3 round 3 (2026-10-18), a MAP fit of upward_count_start_boundary showed that targets sharing their first written digit with the examples were under-predicted. Adding a 5th similarity dimension (an all-or-none leading-digit match, with coherence tuning) took the MAP log-likelihood from −4642.8 to −4630.2. A decade match (t//10) gained only 5.6 and an any-shared-digit match 10.1. This became candidate leading_digit_string_similarity. At MAP the size weight is small (w_base[0] ≈ 0.6 on distance/100), so the size gradient is shallow, and parity drives the near=1 versus near=2 contrast.
**Why:** [[residual-ascending-below-start]] workflow worked again.
**How to apply:** if this candidate wins, look next at the remaining residuals (last_share in (0, 0.3] under-predicted by 0.056; 4-example sets under-predicted).
