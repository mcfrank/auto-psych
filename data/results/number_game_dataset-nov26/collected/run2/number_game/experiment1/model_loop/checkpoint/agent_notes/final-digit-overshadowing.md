---
name: final-digit-overshadowing
description: Overshadowing by a visible shared final digit gained ~7 nats over the one_sided_magnitude_bound_overlap incumbent (iter_4)
metadata:
  type: project
---
Tested 2026-11-02 (iter_4/candidate_0, final_digit_overshadowing_overlap). When at least 2 examples share a final digit, every other family and both bounds get their salience multiplied by `retain`. Posterior retain was about 0.07, so the overshadowing is strong. Plain arviz LOO with 2 chains: +7.4 ± 2.5 nats over the incumbent. 7 of 51 sets (486 trials) have a shared final digit. The idea came from the critique of shared_final_digit_match_rate.
**Why:** it is the first non-split tweak that gained, unlike the ones in [[overlap-model-tweaks]].
**How to apply:** set-dependent salience gating (one visible pattern overshadowing the others) may also work for other conspicuous patterns, such as all multiples of 10.
