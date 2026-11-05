---
name: parity-preserving-random-doubt
description: exp3 iter_1 c1 — n=1 doubt mode keeps the example's parity (yes floor discounted for opposite parity); smoke -4678.2 vs incumbent -4682.4
metadata:
  type: project
---
exp3 iter_1 candidate_1 (2026-11-02): the incumbent single_example_random_program_doubt with one change. Its flat yes-floor is multiplied by (1 - parity_keep) on opposite-parity n=1 targets, with parity_keep ~ Beta(2,2). The smoke fit gave -4678.2, about +4 over the incumbent's -4682.4, so the gain is small.

The lowest n=1 set means (85, 95, 55, 15) are odd examples asked only about even targets. So n1_set_mean_variance and n1_parity_match_gap are largely the same parity effect.

**Why:** the critique's n=1 variance comes mostly from parity, not from how distinctive each seed is.
**How to apply:** don't treat n=1 between-set variance as a separate mechanism from parity. Related: [[single-example-random-doubt]], [[exp3-n1-data-facts]]
