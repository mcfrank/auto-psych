---
name: bounded-arithmetic-progressions
description: exp3 iter_0 c1: bounded APs (step>=2, 16371 of them, equal prior) on rule_count_familiar_strays; eta→0, costs ~1 nat at .003, 5 at .01
metadata:
  type: project
---
exp3 iter_0/candidate_1 (2026-11-01): bounded_arithmetic_progressions = incumbent rule_count_familiar_strays plus every AP a..b step d>=2 (>=2 members), share eta ~ Beta(1,19) split equally. Incumbent MAP ll -3321.79. Fixed eta: .003 → -3322.95 (RMSE .0039), .01 → -3327.3 (.013), .03 → -3348.7, .1 → -3468. Main damage: 23,35,95,83 (step-12 AP) pulls target 25 from .64 to .16.
Implementation: AP counts by (set, example-subset bitmask, size) as CSR over the flattened sub_size_lik (sets*16*100), structured_dot for set evidence and cell numerators: 9.7 ms/grad, smoke 33 s/200 iters. The dense (cells,16,100) gather version took 238 s. Check OK.
Gotcha: find_MAP returns p_yes directly; compile_fn(p_yes) on these models fails or mis-maps, so use mp['p_yes'].
**How to apply:** APs are ruled out as a heavy-share concept, like [[concept-additions-collapse]]. The hull critique (q=.29) is not enough to motivate them. See [[range-restricted-rules]].
