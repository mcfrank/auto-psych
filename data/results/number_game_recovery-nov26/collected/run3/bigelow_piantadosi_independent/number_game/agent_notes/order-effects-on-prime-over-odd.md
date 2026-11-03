---
name: order-effects-on-prime-over-odd
description: exp2 iter_4 order-effect screens on prime_over_odd_reading all ≤1.6 nats MAP; sticky-first-hypothesis worse; graded stray-trust chosen
metadata:
  type: project
---
2026-10-30, exp2 iter_4/candidate_1 (`serial_position_trust_gradient`). Fast MAP NLL screen (base 2126.6):
- first-example stray rate fitted: 2125.4 (e1≈0.14)
- last-example stray: 2125.9 (≈0.08); first/last evidence-weight multipliers: 2126.2 / 2125.7
- win-stay-lose-shift sticky hypothesis from example 1 (or last): stickiness → 0, no gain
- logit-linear stray gradient across position, centred at 0.1: 2125.5 (slope ≈ -0.46); with free level +0.6 more
Only one reordered set pair in the data (8,16 / 16,8), so order effects are barely identifiable.
Speed: the per-position likelihood in the graph costs about 1.5x the incumbent's gradient time when you sum slices of the flattened S×(5H) array. A reshape-broadcast version costs about 2.2x.

**Why:** order-matters briefs keep recurring and give almost nothing on this data.
**How to apply:** don't expect more than about 1 nat from order mechanisms. Related: [[position-stray-rate-finding]], [[set-size-amplified-evidence]].
