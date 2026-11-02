---
name: exp3-iter1-saturation-probes
description: Exp3 iter_1 cand_1 MAP probes on graded_prior_digit_sum_concepts (LL -3513.54) — decades, geometric seqs, digit gaps, "anything" concept, odd/even split all dead; range-location slope +0.18 (submitted); incumbent at noise ceiling
metadata:
  type: project
---
iter_1 candidate_1 (2026-11-01, exploratory). Incumbent graded_prior_digit_sum_concepts MAP LL -3513.54 (fw parity .025, special .15, mult .125, ends .14, contains 0, exceptions 0, dsum .025, intervals .53; scale 79, centre .017, decay -0.25).
- Saturated-cell half-deviance 646 over 1227 cells (median 5 obs) ≈ noise level, so at most a few tens of nats are left to gain. Calibration gain ≈1 in every trial_index block, no drift.
- Each of these as its own family: decade concepts (n//10 = d) weight 0; geometric r*q^k weight 0; |tens-units| = k weight 0; "anything 1-100" weight 0; odd and even as separate families stay symmetric (+0.02).
- Multiples of 13-50 family +0.34 (sibling cand_2 took it). Range-location prior exp(slope*midpoint) within width: slope +0.27, +0.18, RMSE 0.004 → submitted as location_biased_range_prior.
Probe scripts are in /tmp/c1i1 (map.py = MAP LL + saved p_yes for any candidate file; perset.py = per-set deviance). They may not persist.
**How to apply:** stop adding concept families or prior tweaks to exp3; see [[exp3-concept-family-probes]], [[exp3-iter0-centre-weighted-probes]].
