---
name: interval-space-is-load-bearing
description: Swapping the Bayesian interval hypothesis space for a single range/neighbourhood estimate costs ~230-260 nats
metadata:
  type: project
---
iter_4 candidate_2 (2026-11-01) replaced the sticky learner's interval hypotheses with a single range estimated from the example extremes, extended by a margin of base + share*span/n (Gaussian, Cauchy or exponential edge), or with a nearest-example neighbourhood. At MAP the log-lik was -3347 to -3378, against -3110 for span_extrapolation_hesitant_guess_2. Most of the loss is in widely spread 4-example sets (92,14,20,5; 11,41,38,92): people do not fill the gaps. The Bayesian intervals manage this because they can cover a sub-cluster and treat the remaining examples as noise.

**Why:** order-free or symmetric range mechanisms (aimed at the first_example_side_extension critique) cannot work unless the range can leave out outlying examples.
**How to apply:** keep the averaging over intervals with noisy examples, and target primacy effects in the sticky dynamics instead. A fast MAP check (/tmp/map_c2.py, /tmp/diag_c2.py pattern, about 1 min) catches a large loss before the 7-min check. These models run fast (about 25 s per smoke fit) with only 46 hypotheses.
