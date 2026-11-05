---
name: five-family-multiple-route
description: exp3 iter4 c4 — parity-switch cut on multiples route exempts 5-ending examples, on coincidence_rule_violation_route; +6 nats MAP loglik, 0 extra params
metadata:
  type: project
---
exp3 iter4 candidate_4 (2026-11-02): refinement of [[coincidence-rule-violation-route]]. The multiples-route parity cut applies only to odd examples not ending in 5, so 5→20/70/100, 25→50 and 15→90 keep the route. MAP loglik went from -4027.9 to -4022.0 with no new parameter. Smoke log-lik -4030.4.
Residual findings on the incumbent (MAP): its magnitude gradient is nearly flat (c_mag≈0.07 per 10), and far targets get p≈0.3-0.5. Incoherent sets over-credit routes from one example (24,35→4/72/96 observed ≈0, predicted ≈0.5). Grafting set-kin consensus (rho≈1.6) on top of this change gave a further +9.5 nats (-4012.4), but that is a second change, so it was not submitted. It is a good next refinement.
**Why:** addresses the single-example polarization critique (5-family round multiples were under-predicted by +0.25-0.46).
**How to apply:** if this wins, try the set-kin consensus graft next. The scratch MAP script is at /tmp/c4s/fitmap.py (it may be gone).
