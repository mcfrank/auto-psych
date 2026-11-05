---
name: exp3-order-effects-saturated
description: exp3 iter_3 c0 — order mechanisms on parity_bounded incumbent all ~0 MAP (momentum band +0.8, first/last forgetting +1, whole-profile anchoring 0)
metadata:
  type: project
---
exp3 iter_3 candidate_0 (2026-11-02), an order-effect brief, on the parity_bounded_ambiguity_doubt incumbent (MAP -4685.8). Changes tested:
- Direction-of-travel momentum: log band width ±momentum on the side the last-minus-first trend points to. Band only +0.8 (submitted as `sequence_momentum_extrapolation`), band and span +0.5, span only 0.
- Forgetting the first or the last example, on n>=3 or on n=4 only: +1.2 at best.
- Non-shared properties of the first or last example lingering: 0.
In the raw data, n=2 sets show ahead .375 vs behind .30, but the paired difference within sets is ~0, and n=3/4 do not show it.

**Why:** don't spend another round on order effects for exp3.
**How to apply:** an order brief gets a tie at best. Related: [[leaky-memory-recency]] [[working-memory-three-example-limit]] [[parity-bounded-ambiguity-doubt]]
