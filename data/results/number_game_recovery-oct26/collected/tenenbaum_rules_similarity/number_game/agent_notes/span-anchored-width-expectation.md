---
name: span-anchored-width-expectation
description: exp3 iter_2 c1 (retry): Erlang width scale × ((span+1)/20)^power per set; cheap per-set prior; smoke -3094.1, 215s
metadata:
  type: project
---
2026-10-22, exp3 iter_2 candidate_1 was retried because the first attempt ([[bottom-to-top-demonstration]]) left no files. It is now `span_anchored_width_expectation`: a copy of compressed_rule_extension in which the range width prior is made per-set, scale = width_scale*exp(power*log((span+1)/20)), with power ~ N(0, 0.5). This double-counts the spread and targets set_size_near_outside_gap. Check OK: smoke LL -3094.1, 215 s.
Per-example evidence tempering or sublinear n is expensive here, because group/conj evidence is precomputed by logsumexp, so tempering needs (k, size) decomposition or a grid (about 3x cost). Per-set *prior* changes are cheap.
**How to apply:** check the outcome and novelty (power near 0 → duplicate risk). Related: [[compressed-rule-extension]], [[pymc-model-speed-patterns]]
