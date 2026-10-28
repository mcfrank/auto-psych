---
name: first-shown-top-marking
description: exp2 iter_4 c1 order slot: top-marking credit only when first example is the max; smoke LL -2057.5, 236 s
metadata:
  type: project
---
2026-10-19, exp2 iter_4 candidate_1 (exploratory, order brief): `first_shown_top_marking` — a copy of top_marking_range_sampling with top_share replaced by 1[first example == max] (the program opens by showing its top). It's a one-line edit. Raw data: for multi-example sets, yes rates 1-10 above the max were 0.39 when the first example was the max and 0.48 when it was the min, but set identity confounds this. Check passed: smoke LL -2057.5, 236 s per run.
**Why:** records which order-based forms were tried. **How to apply:** check its LOO before trying "last example marks top" variants. Related: [[generative-top-marking]]
