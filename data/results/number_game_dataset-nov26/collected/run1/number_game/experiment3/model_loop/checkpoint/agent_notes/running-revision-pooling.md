---
name: running-revision-pooling
description: exp3 iter3 c0: delta-rule serial-position pooling (lr 1/k^rho) on parity_preserving_decisive_route; rho MAP 0.96, +0.2 nats — no order effect in data
metadata:
  type: project
---
Candidate `running_revision_pooling` (2026-11-02): example weights from a delta rule with learning rate 1/k^rho (rho=1 = incumbent's equal weighting), on parity_preserving_decisive_route. Smoke -4073.8. MAP log-lik -4065.8 vs incumbent -4066.0, log_rho ≈ -0.04 → likely near-duplicate rejection.
**Why:** brief asked for an order mechanism; data show essentially no primacy/recency (critique first_vs_last stat also n.s., see [[exp3-iter3-critique-stats]]).
**How to apply:** don't spend more slots on serial-position weighting of the pooled similarity; order-effect ideas need a different locus (or none exist). Compare [[recency-gated-arithmetic-route]], [[running-guess-coincidence-exemplar]].
