---
name: residual-ascending-below-start
description: Exp3 round 2 MAP residual finding — targets below the first example of ascending sets are over-predicted by the best model
metadata:
  type: project
---
In experiment3 round 2 (2026-10-18) a MAP fit of saturating_criterion_relaxation showed that targets below the first example of sets whose last example is larger than the first were over-predicted by about 0.1. Adding one all-or-none logit penalty for those targets improved the MAP log-likelihood by about 28 nats (−4671 to −4643). The symmetric "behind the start" version gained only about 6, and a graded version about 19. Targets 1 and 2 are also over-predicted by about 0.11 overall. This became candidate upward_count_start_boundary.
**Why:** a quick MAP fit plus grouped residuals (scratch/resid.py) finds targeted single mechanisms much faster than reading the critique alone.
**How to apply:** before proposing a candidate, MAP-fit the incumbent and group residuals by set and target structure.

Round 3 (2026-10-18): grafting the same penalty onto sample_size_neglect_range_inference raised its MAP log-likelihood from −4667 to −4639, about 4 better than the incumbent (−4643). The penalty fit at about 0.63 in both models, so it is roughly additive with range inference. This became candidate sample_size_neglect_upward_start.
