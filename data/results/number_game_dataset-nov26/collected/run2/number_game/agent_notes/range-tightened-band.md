---
name: range-tightened-band
description: band width narrower for multi-example sets (tighten~0.24) on exp2 incumbent progression_band_first_digit_primacy: grouped LOO +9.3
metadata:
  type: project
---
Exp2 iter_4 c3 (2026-11-02), range_tightened_magnitude_band: graded-band width = 0.02 + band_width*(tighten if n>=2 else 1). Own harness (iter_3/candidate_5/scratch/ev.py, 2x300 NUTS): incumbent grouped LOO -3142.2 -> -3132.9; band_width rises 0.61->0.87 for n=1, multi width ~0.21. Smoke -3124.3, 0 divergences. Motivated by inrange_nonshared_rate critique (in-range vs out-of-range nonshared targets).
**Why:** one shared band width was a compromise between broad n=1 and tight multi-example generalization.
**How to apply:** set-size-dependent band width is a real effect; stacks with [[band-plus-leniency]]? untested. See [[graded-magnitude-band]].
