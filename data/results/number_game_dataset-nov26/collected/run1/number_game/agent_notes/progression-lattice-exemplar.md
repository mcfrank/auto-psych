---
name: progression-lattice-exemplar
description: exp3 iter0 c1: distance-free route for targets on examples' step lattice (gcd of gaps 2-20) on round_number_averaging incumbent, +10 nats MAP loglik
metadata:
  type: project
---
exp3 iter0 c1 (concept-space slot) submitted `progression_lattice_exemplar`: incumbent round_number_averaging_exemplar + additive route exp(log_w_step)*on_step, on_step = n>1, step = gcd of sorted gaps in [2,20], (t-ex0)%step==0, t not an example. MAP loglik -4097.8 vs incumbent -4108.2; smoke -4104.6.
Screens on the same incumbent (MAP loglik): rule-with-one-exception coincidence sharpening → weight ≈0.05, no gain (exceptions don't count); odd-vs-even parity salience asymmetry +7.8 nats.
Raw checks: shared tens digit / contains-any-digit / equal digit-sum give no raw effect on far unrelated targets.
**Why:** saves re-screening these concept additions. **How to apply:** parity-salience asymmetry is an untried +8 option; see [[exp3-iter0-critique-stats]].
