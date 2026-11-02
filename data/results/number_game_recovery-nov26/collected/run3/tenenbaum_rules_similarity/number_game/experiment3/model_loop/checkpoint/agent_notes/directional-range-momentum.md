---
name: directional-range-momentum
description: exp2 iter_4 c1: range placement tilted toward direction of travel (first→last), tau on incumbent mixed_strong_weak; dense (width,start) interval layout + cumsum trick
metadata:
  type: project
---
exp2 iter_4/candidate_1 (2026-10-31), directional_range_momentum = mixed_strong_weak incumbent + within-width placement prior exp(tau·sign(last−first)·(centre−examples' midpoint)/100), normalised per width. Motivated by sequence_continuation_asymmetry (+0.10 obs vs 0.03 null, p=0.41, not significant — weak).
Speed: a dense per-set (width, start) layout cost 70–680 ms per gradient, too slow for admission. What worked was 17 ms (as fast as the incumbent): per set and width, group runs of starts with equal k into segments (at most 2n+1 per width). A segment's tilted mass is the prefix-sum difference Psum[d,a2] − Psum[d,a1−1] with Psum = cumsum exp(tau·d·(a−50)/100), d ∈ {desc, none, asc}. P(target in range) uses 2 queries per width: offset[j] + ratio[j]·Psum[a], with ratio = post/mass and offset = cumulative mass before j minus ratio·Psum[a1−1]. It matches the dense version to 1e-13. Fancy 3-D indexing and per-cell gathers over all 5050 intervals were slower than the dense dot.
Smoke check OK (ll −2268.9, about 2 min). tau = 1 moves p_yes by RMSE 0.02 compared with tau = 0.
**Why:** order effects are hard to identify (only one permuted pair, see [[serial-position-reliability]]); novelty risk if tau≈0.
**How to apply:** check the result in the iter_4 manifest before spending more order-effect slots.
