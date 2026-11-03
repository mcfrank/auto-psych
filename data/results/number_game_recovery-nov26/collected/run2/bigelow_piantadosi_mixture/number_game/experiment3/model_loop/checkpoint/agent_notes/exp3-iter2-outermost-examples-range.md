---
name: exp3-iter2-outermost-examples-range
description: Exp3 iter-2 cand_0 — ranges weighed by min/max examples only (interior = uniform in span) loses ~172 nats MAP; interior examples do tighten ranges
metadata:
  type: project
---
Exp3 iter 2 candidate_0 (2026-10-31), `outermost_examples_range_evidence`: incumbent heading_weber_with_upward_counting with range likelihood from the extreme examples only. MAP LL -3362 vs incumbent -3190 (same /tmp/mapchk.py find_MAP script). Submitted as a 0-param falsification test.
Pitfall: dropping interior examples' factors from range evidence alone (without a matching constant) lets ranges escape the per-example cost rules pay. Ranges then always win and LL is about -4150. Fix: give each interior example a likelihood of being uniform within the span.

**Why:** people do count interior cluster examples toward the size principle for ranges.
**How to apply:** don't propose sample-size neglect / extremes-only attention for ranges. See [[number-game-ceiling]], [[exp2-iter4-size-principle-rules-only]].
