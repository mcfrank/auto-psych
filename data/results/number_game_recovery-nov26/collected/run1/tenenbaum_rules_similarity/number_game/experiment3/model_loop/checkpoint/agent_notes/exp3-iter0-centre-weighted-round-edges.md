---
name: exp3-iter0-centre-weighted-round-edges
description: Exp3 iter_0 candidate_5 — round-number range-edge bonus on centre_weighted_ranges_with_thresholds: MAP +1.0 nats, bonus -0.76 (same sign as exp2)
metadata:
  type: project
---
iter_0 candidate_5 (2026-11-01, refinement slot), exp3 data (6750 rows). Parent centre_weighted_ranges_with_thresholds MAP LL -3518.55 (scale 81, centre_weight 0.024). Adding round_bonus per bounded-range edge on a multiple of 10 (Normal(0,1), normaliser over (width, #round edges) cells): MAP -3517.56, bonus -0.76. Submitted as centre_weighted_round_edge_ranges. Probe script /tmp/c5i0/map.py (find_MAP on any candidate file; may not persist).
**Why:** the round-edge disfavour replicates across exp2 and exp3 (~-0.7), so it is a stable but small (~1 nat) effect.
**How to apply:** don't re-propose this graft; see [[exp2-iter4-round-number-refinements]], [[exp2-memory-and-prior-anchoring-probes]].
