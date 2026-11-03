---
name: exp2-iter3-heading-open-graft
description: Exp2 iter-3 cand_5 refinement — open-ended range discount grafted onto heading_direction: MAP NLL 1921.98 vs 1923.1/1923.3 parents; submitted
metadata:
  type: project
---
Exp2 iter 3, candidate_5 (2026-10-30), refinement slot. Submitted `heading_extrapolation_bounded_ranges`: the heading_direction model with intervals grouped by (width, open) in its Taylor moments (200 groups) and a log_open prior.
Numpy MAP NLL: base digit_sum 1925.26, +heading 1923.26, +open 1923.10, both 1921.98. The two effects are mostly additive but overlap, so each adds only about 1.2 nats on top of the other. Smoke loglik -1925.8; check about 100 s. guess_yes goes to 1 in MAP (lapse is tiny).
Sibling slots cand_3/4 refine the incumbent and may graft heading onto it too, which would be a duplicate risk. See [[exp2-iter2-heading-direction]], [[exp2-iter2-open-range-graft]].
