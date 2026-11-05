---
name: set-kin-consensus-violation-route
description: exp3 iter4 c3: set-kin consensus route scaling grafted onto coincidence_rule_violation_route, +6.9 nats MAP; smoke -4029.2
metadata:
  type: project
---
Grafted the set_kin_consensus route gain exp(-rho*share of examples not kin to the target) onto coincidence_rule_violation_route (2026-11-02). MAP response log-lik -4021.0 vs the incumbent's -4027.9, rho≈1.4, c_viol≈0.51. Smoke fit -4029.2.
Polarized incoherent-set cells are mostly factor-kin or multiples of one example (25 or 55 after "10,3"; 72 or 96 after "24,35"; 16 after "48,63,53"). Polarized single-example cells are multiples: yes for round or even examples, no for parity-switched multiples of odd examples (34 after "17", 54 after "3", 6 and 12 after "9"). This graft leaves the single-example cells unaddressed.
**Why:** answers the incoherent_multi_cell_polarization critique in [[exp3-iter4-critique-stats]]. The base is [[coincidence-rule-violation-route]] and the graft comes from [[set-kin-consensus-exemplar]].
**How to apply:** single-example polarization is still open. A cut on the multiples route for odd examples, steeper than the current parity cut, may help (7 → 14 and 5 → 100 are exceptions).
