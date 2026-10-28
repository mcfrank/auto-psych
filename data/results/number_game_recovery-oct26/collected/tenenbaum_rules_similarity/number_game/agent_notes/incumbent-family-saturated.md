---
name: incumbent-family-saturated
description: MAP probes of prime_concept_excludes_one (exp2 round1) — calibration perfect, most single tweaks gain <1 nat; rule-list edits hurt
metadata:
  type: project
---
MAP probes on `prime_concept_excludes_one` (exp2, iter_1 candidate_4, 2026-10-19), MAP LL -2054.1 baseline:
- Calibration by predicted-p bin is near-perfect; residuals by distance/span/parity/set size show no trend.
- Free gamma shape on width: no gain. Rule-membership fade with distance: no gain.
- Rule-size prior tilt: +0.35. Range-only tempered size principle (sharpness 0.85): +0.2 (submitted as `range_discounted_size_principle`).
- Epsilon 0.05: -37, 0.2: -58 (0.1 near optimal). Multiples 3-25: -65 (kind dilution); multiples 3-10/11: -3; powers 2-5 only: -25; drop contained-digit: -0.2.
- Data: 10 participants x 450 trials (trial_index resets), 5 responses per cell, so cell noise dominates.

**Why:** Saves re-running these probes. Most live models' ~50-nat deficits are just the missing prime fix.
**How to apply:** Look for gains outside this family's parameter tweaks; see [[prime-rule-includes-one]].
