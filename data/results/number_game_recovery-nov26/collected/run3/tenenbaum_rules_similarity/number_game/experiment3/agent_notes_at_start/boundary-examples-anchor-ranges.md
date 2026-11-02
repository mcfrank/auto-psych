---
name: boundary-examples-anchor-ranges
description: exp2 iter_3 c1: interior (non-min/max) examples weighted w in range likelihood only; exact at w=1, 8 ms grad
metadata:
  type: project
---
exp2 iter_3 candidate_1 (2026-10-30), boundary_examples_anchor_ranges = beta_share incumbent + interior examples' range likelihood raised to w = 2*Beta(2,2); rules use all examples.
Trick: per (set, size, #interior examples inside ∈{0,1,2}) precompute logsumexp of boundary-example ll and target membership; the model adds w*(c*log(.9/L+.001)+(k-c)*log(.001)). At w=1 logp equals the incumbent's exactly (after the prior on w is removed). 65 of 119 sets have interior examples.
Smoke: ll -2269.2; 27 s for 200 iters on 1 chain; logp+grad 8 ms.
**Why:** an attention mechanism that is not position based (cf. [[serial-position-reliability]]) and not a global evidence exponent (cf. [[experiment2-null-refinements]]).
**How to apply:** check its fitted w and LOO before trying other within-set attention ideas.
