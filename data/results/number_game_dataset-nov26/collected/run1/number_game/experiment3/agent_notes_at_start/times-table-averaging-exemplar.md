---
name: times-table-averaging-exemplar
description: exp2 iter3 c5 — similarity averaging (−kappa·log n) grafted on times_table_multiple_route, MAP -2671.7 vs incumbent-equivalent -2682.5; smoke -2677.5
metadata:
  type: project
---
Exp2 iter_3 candidate_5 refinement slot (2026-11-02): `times_table_averaging_exemplar`.
MAP log-lik screen (my reimplementation):
- factor_kin incumbent: -2682.5
- times_table base: -2686.0
- times_table + averaging: -2671.7, kappa 0.71 (submitted)
- multiple_route + averaging: -2675.7
- incumbent + averaging: -2668.6
- times_table + averaging + factor-kin: -2664.6
Smoke fit -2677.5, 11 s.
**Why:** averaging stacks on top of the multiples-route models (about 11–14 nats), roughly independently of the route's variant. See [[similarity-averaging-exemplar]] and [[times-table-multiple-route]].
**How to apply:** next round, try averaging on whichever route model is incumbent. The tt+avg+fk version is the best seen so far.
