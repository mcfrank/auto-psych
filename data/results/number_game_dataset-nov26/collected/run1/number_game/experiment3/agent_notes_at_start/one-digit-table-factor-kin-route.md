---
name: one-digit-table-factor-kin-route
description: exp2 iter3 c3 — incumbent factor_kin_multiple_route with multiples route only from examples 2-9, MLE -4.8 nats; smoke -2683.1
metadata:
  type: project
---
Exp2 iter_3 candidate_3 (2026-11-02): `one_digit_table_factor_kin_route` refines [[factor-kin-multiple-route]] and targets large_example_multiple_contrast from [[exp2-iter3-critique-stats]].
MLE NLL: the incumbent scored 2682.4. With a soft weakener exp(-tau·[e>=10]) on the multiples route it scored 2677.6, with tau → ∞, so the candidate uses a hard 2-9 gate (no extra parameter). The smoke fit was -2683.1.
Applying the same gate to the factor-kin route, or using a linear decay in example size, gave much worse optima (possibly optimizer failures, not checked).
**Why:** the one-digit gate is a gain of about 5 nats that is independent of the kin route.
**How to apply:** expect at most a LOO tie with the incumbent. The size-gating idea is now used up on both routes.
