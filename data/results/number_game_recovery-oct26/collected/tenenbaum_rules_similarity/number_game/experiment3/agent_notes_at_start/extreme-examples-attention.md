---
name: extreme-examples-attention
description: exp2 iter_3 c1 submitted extreme_examples_attended_sampling (inner examples partly weak-sampled, no-conj base); check passed, smoke LL -2058.5, ~120 s/run
metadata:
  type: project
---
2026-10-19, exp2 iter_3 candidate_1 (exploratory, process-level brief): `extreme_examples_attended_sampling`. Min and max examples are strong-sampled; inner examples mix strong and weak sampling with Beta weight `inner_attention`. Built on a no-conj base (rules plus intervals, with an upper-edge bonus). Intervals are pooled exactly by (at_top, width-group, extremes covered, inner covered), which works because size is constant within a width group. Check OK: smoke LL -2058.5, about 120 s per sampling run, about 5 min in total.
**Why:** record the outcome so it is not re-proposed. **How to apply:** check the admission and novelty result before reusing it. Pooling cells by covered counts is a cheap way to give interval models a parameter-dependent likelihood. Related: [[pymc-model-speed-patterns]], [[novelty-gate-tiny-tweaks]]
