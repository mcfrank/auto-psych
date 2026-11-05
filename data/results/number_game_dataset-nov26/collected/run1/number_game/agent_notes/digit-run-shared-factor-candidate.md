---
name: digit-run-shared-factor-candidate
description: iter_4 c5 refinement — digit-run criterion model + coprime penalty, MLE -6.6 nats; run-sharpened digit attention and mod-5 credit ~0
metadata:
  type: project
---
iter_4 candidate_5 (2026-11-02), `digit_run_criterion_shared_factor`: digit_run_strict_criterion_exemplar + fixed coprime penalty grafted from the incumbent. In-sample NLL: base 1545.0, + coprime 1538.4 (submitted, check OK, smoke LL -1542.5). Alternatives with no gain: the run sharpening final-digit attention in place of the criterion shift (1544.1), and partial credit for final digits in the same mod-5 class, aimed at the multiples-of-5 critique (1545.0).
**Why:** saves re-screening. **How to apply:** the multiples-of-5 contrast and set-level SD still need a mechanism; see [[digit-run-strict-criterion]], [[iter4-critique-stats]], [[shared-factor-exemplar-screen]].
