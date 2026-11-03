---
name: exp2-memory-and-prior-anchoring-probes
description: Exp2 iter_3 probes on without_replacement_size_principle — conservatism, recency memory, family sharpening, digit confusion pin at normative; round-number range edges -0.64 (+0.58 nats)
metadata:
  node_type: memory
  type: project
  originSessionId: dbdb6df5-37b7-41e3-bcb6-72900301d106
  modified: 2026-10-30T20:00:42.261Z
---

iter_3 candidate_2 (2026-10-30), base without_replacement_size_principle MAP LL -2245.12 (family weights ranges 0.55, scale 72):
- Conservatism: belief = (1-c) posterior + c prior predictive membership → c = 0.0, RMSE 4e-5. Dead.
- Recency-decay working memory (last example always held, earlier one at lag L held w.p. r^L, average posterior over surviving subsets; 527 subsets for 131 sets) → r = 1.0, RMSE 5e-5. Dead (the exp1 leaky-memory idea doesn't help in exp2 either).
- Family-level sharpening (softmax over 8 family evidences with exponent b, then within-family posterior): b 0.988, pool RMSE 0.0015 (dead); fixed b=3 costs 537 nats, b=20 ~2000. Fixed working-memory capacity (last 3 examples only) costs 290 nats, last 2 costs 715.
- Positional digit confusion ("ends in d" also accepting tens digit d) at response: eps 0.0.
- Round-number range boundaries (bonus per bounded-range edge that is a multiple of 10, groups by width x #round edges): bonus -0.64, LL +0.58, data RMSE 0.0023, pool RMSE 0.0028 → submitted as round_number_range_boundaries. Other conventions: lo%10==1 & hi%10==0 -0.60; lo%10==0 & hi%10==9 -0.11 (nothing); lo%10==1 & hi%10==9 -0.07 (nothing); multiples of 5 -0.16 (nothing). So it is ranges ending ON a multiple of ten that are disfavoured.
- Aggregate residuals of the base (target decile, set size, member, below/in/above, distance, pred bin) all |z|<1.4: calibration is essentially perfect.
**Why:** learned-strength distortions all collapse to the ideal learner on this data, so they are rejected as near-duplicates.
**How to apply:** prefer a fixed-strength (parameter-free) distortion or a new concept family if a candidate must differ from the incumbent; see [[exp2-map-probe-results]], [[exp2-pedagogical-sampling-probe]].
