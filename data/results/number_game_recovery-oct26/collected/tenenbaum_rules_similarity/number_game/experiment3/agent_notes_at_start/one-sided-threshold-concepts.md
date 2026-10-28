---
name: one-sided-threshold-concepts
description: exp2 iter_4 c0: rules + one-sided "up to N"/"N and above" thresholds instead of ranges; smoke LL -2459.8 (~400 nats worse), check OK, ~20 s/run
metadata:
  type: project
---
2026-10-19, exp2 iter_4 candidate_0 (exploratory "simpler/high-variance" brief): `one_sided_magnitude_threshold_concepts`. Two-ended ranges are replaced by half-lines, with a round-decade threshold boost and an upper/lower share. The check passed, but the smoke LL was -2459.8, against the incumbent's ~-2057. So two-ended ranges are essential.
The iter_4 critique was all null, including the session SD stats: the model reproduces the session yes-rate spread, so [[session-heterogeneity]] is not a misfit.
**Why:** so this is not re-proposed. **How to apply:** don't drop bounded ranges. Related: [[ppc-critiques-null]]
