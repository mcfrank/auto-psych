---
name: data-coverage-quirks
description: cell_1/experiment1 responses.csv coverage gaps that limit which test statistics are computable
metadata:
  type: project
---
Found 2026-10-30 (iter_4 critique). responses.csv has 2250 rows and 172 sets, with about 5 answers per (set, target) cell and no participant_id.
- Power-of-2 sets never ask about a power-of-2 target that isn't an example, so you can't test "powers vs multiples" there directly. Use the multiples-of-4 yes rate (0.44) instead.
- Sets whose examples all share a last digit: targets with that last digit get 0.96 yes (n=55), other targets 0.06 (n=505).
- 92,82,27,42 and 22,62,26,82 get high yes rates across the 20s. 83,38,58,68 gets 85/87/89 yes and 50-76 no.

**Why:** saves re-checking coverage when designing statistics.
**How to apply:** check that a stimulus slice has targets before building a statistic on it. See [[exponent-concept-cost-finding]].
