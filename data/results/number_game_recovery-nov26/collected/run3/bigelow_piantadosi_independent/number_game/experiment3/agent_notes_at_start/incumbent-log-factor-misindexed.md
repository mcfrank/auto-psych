---
name: incumbent-log-factor-misindexed
description: factor_graded_multiples_concepts LOG_FACTOR[18:27] is off by 2; multiples rules sit at rule indices 16:25
metadata:
  type: project
---
In exp2's `factor_graded_multiples_concepts.py`, `log_m[18:27]` is shifted by two rows. Multiples of 3/4/5 get no cost, multiples of 6-12 get the cost meant for 3-11, and "ends in 0" and "ends in 1" get log(11/3) and log(4). The multiples rules are actually at indices 16:25 (odd, even, primes, squares, cubes, powers 2-10, two p2 variants come first). Its +34 gain ([[factor-graded-multiples-finding]]) may partly come from penalizing ends-in-0/1.

**Why:** a refinement that copies the slice inherits the bug, and comparisons against the incumbent are confounded.
**How to apply:** print rule sizes and first members for each index before reusing an index slice. iter_1 candidate_5 (forward_counting_graded_multiples) uses the correct 16:25.
