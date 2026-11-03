---
name: arbitrary-opening-example
description: exp2 iter_4 c1 repair: first example weakly sampled (membership only), later ones mixed sampling; built so it can't collapse to the incumbent
metadata:
  type: project
---
exp2 iter_4/candidate_1_repair_1 (2026-10-31), arbitrary_opening_example. The first try, [[directional-range-momentum]], was rejected as a novelty duplicate of mixed_strong_weak: the fitted tau was ≈0, giving RMSE 0.00147. The repair adds no new parameter. The first example's likelihood is the fixed weak term ALPHA/100+(1−ALPHA)/100. Later examples keep the incumbent's θ mixture. Segments break on 2k+k1, where k1 marks whether the interval contains the first example. Uniform placement means psum = arange(101). Gradient takes 24 ms. Initial logp is −2932. Smoke check OK with ll −2405.6, against about −2269 for the incumbent, so it loses ~135 nats: people do apply the size principle to a lone first example.
**Why:** an inferred order parameter collapses to null on exp2 (only one permuted pair), so order mechanisms need to be fixed and structural to pass the novelty gate.
**How to apply:** for future order-effect slots, use parameter-free structural changes, or skip them.
