---
name: band-plus-leniency
description: single-example leniency on progression_graded_magnitude_band: grouped LOO +8.1 (−3142.9 vs −3151.0); coincidence gate on same base +9.0
metadata:
  type: project
---
Exp2 iter_3 candidate_4_repair_1 (2026-11-02), progression_band_single_leniency. Own harness (candidate_5/scratch/ev.py, 2x300 NUTS): incumbent −3151.0 grouped LOO, +lenience (n=1 logit shift, ~0.77) −3142.9, +coincidence gate −3142.0. Leniency and the band stack. Smoke −3131.5. The first try (first-digit primacy on the incumbent) was rejected as an exact duplicate of progression_band_first_digit_primacy.
**Why:** every primacy variant on the incumbent already exists (digit-only, all-props anchoring, coincidence+primacy).
**How to apply:** before grafting onto the incumbent, check models/ for an existing graft. See [[progression-leniency-graft]], [[coincidence-plus-primacy]].
