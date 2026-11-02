---
name: exp3-iter2-disjunction-probe
description: exp3 iter_2 cand_2 (2026-11-01) — two-rule union family (2,000+ distinct unions) gets weight ~0 with 10% noise; with 0.1% noise unions take 15% but MAP LL -4334 (~820 nats worse); submitted the no-noise version as exploratory
metadata:
  type: project
---
Base: rules + digit sums (equal share) + Erlang ranges, no centre weighting, width-grouped precompute (prepare_observed sums ranges per width, so graph is tiny; smoke check ~2 min).
- ALPHA 0.9 + union family: union weight 2e-6, MAP LL -3514.7 (incumbent -3513.5) → dead, likely near-duplicate.
- ALPHA 0.99: -3910.6; ALPHA 0.999: -4333.9 (unions 0.15 weight). Treating outliers as noise beats explaining them by disjunctions by ~800 nats.
- Submitted `two_rule_disjunction_concepts` with ALPHA 0.999 (high-variance slot).
**How to apply:** don't retry disjunctive/union concepts on exp3; the 10% noise channel is essential. See [[exp3-iter1-process-probes]].
