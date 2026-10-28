---
name: spread-sampled-ranges
description: exp3 iter_1 c1 repair: stratified (evenly spread) range sampling mixture; pure spread rejected by ~350-480 nats; MAP spread_prob ~0.003 but RMSE 0.0086 vs incumbent
metadata:
  type: project
---
2026-10-22, exp3 iter_1 candidate_1_repair_1 `spread_sampled_ranges`. The first attempt (compressed rule size) was rejected as a near-duplicate of compressed_rule_extension (RMSE 0.0009).
The repair adds a second block of plain ranges whose likelihood uses stratified sampling: the range is cut into n equal strata, each example must sit alone in its own stratum, and each gets probability n/W. The prior is the incumbent's times spread_prob (Beta(2,2)).
At MAP, spread_prob is about 0.003 and LL is -3092.3 (incumbent -3090.5). The n^n boost still moves 4-example spread sets: RMSE 0.014 for n=4 and 0.0086 overall.
Fixing spread at 0.5 gives LL -3440; at 0.999 it gives -3574. People do not assume evenly spread examples.
Check OK: smoke LL -3094.8, 330 s per sampling run (grad 0.082 s).
**How to apply:** Don't retry stratified/spread sampling as a main mechanism. Check its novelty/ELPD outcome. Related: [[compressed-rule-extension]], [[last-shown-top-discounted-ranges]]
