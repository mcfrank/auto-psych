---
name: span-sharpened-band
description: separate band width for multi-example sets on progression_band_first_digit_primacy: grouped LOO +9.3; hard in-span property +5
metadata:
  type: project
---
Exp2 iter_4 candidate_4 (2026-11-02), span_sharpened_magnitude_band. On the incumbent progression_band_first_digit_primacy, own harness (scratch/ev.py, 2x300 NUTS): incumbent grouped LOO -3142.2; separate band width for n>=2 (span_width ~0.19 vs single ~0.92) -3132.9; hard "within span" property with HalfNormal salience -3137.3. inrange_nonshared_rate stat: obs 0.066, inc -0.016, A 0.010, B 0.034. Smoke -3123.9.
**Why:** single sets want a broad band and multi sets want a sharp one; one shared width compromises.
**How to apply:** B closes the critique stat better but fits worse; an in-span property on top of A may be the next graft. See [[graded-magnitude-band]], [[progression-band-primacy]].
