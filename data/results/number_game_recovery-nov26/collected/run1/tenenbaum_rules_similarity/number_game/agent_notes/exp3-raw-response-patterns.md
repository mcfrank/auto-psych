---
name: exp3-raw-response-patterns
description: Raw exp3 response patterns seen while writing iter_1 critique stats (target-is-example yes only 0.65, no parity effect for all-odd sets, flat proximity)
metadata:
  type: project
---
Exp3 responses.csv (6750 rows, 154 sets, trial_index 0-449, no participant_id), seen 2026-11-01 while writing iter_1 critique stats:
- Target equal to an example: only 0.65 yes (250 trials).
- All-odd multi-example sets: odd targets 0.49 vs even 0.48, so no parity effect. All-even sets: even 0.36 vs odd 0.06.
- Yes rate is flat in distance to the nearest example (0.40-0.44 from 1 to 100 away).
- Targets 71-90 get the lowest yes rate (0.35 vs about 0.46 elsewhere).
- iter_0 critique: 0 of 8 statistics were significant against centre_weighted_range_sampling.
**Why:** these point to likely model misfits (endorsing examples, the odd rule) to test.
**How to apply:** check iter_1 ppc_results.json to see which of these the graded digit-sum incumbent misses. See [[exp3-concept-family-probes]].
