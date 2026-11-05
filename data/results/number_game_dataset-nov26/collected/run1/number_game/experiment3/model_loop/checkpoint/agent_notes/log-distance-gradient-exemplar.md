---
name: log-distance-gradient-exemplar
description: exp3 iter1 c1: log(1+gap/delta) magnitude gradient on whole_set_round_category, +7.3 MAP; power-law +5.9; singleton/size tightening ~0
metadata:
  type: project
---
Exp3 iter_1 candidate_1 (2026-11-02): `log_distance_gradient_exemplar` = incumbent whole_set_round_category with c_mag*dist replaced by c_mag*log1p(dist/delta). MAP log-lik: incumbent -4098.3, log-distance -4091.0 (delta≈0.41 i.e. ~4 units), power dist^q -4092.4 (q≈0.43). Smoke fit -4098.1.
No gain (<0.5 nats): c_mag × n^gamma set-size tightening, free lone-example spread, extra log(spread) term on c_mag, times-table co-membership replacing multiples+factor-kin (-60, bad).
Incumbent MAP has spread_tuning>0 (wide sets generalize *more* broadly) and very flat magnitude (c_mag≈0.06/10 units). Polarization critique not moved by any of these (exp 0.326→0.328 vs obs 0.373); residual misfit is item-level multiples (17→34, 43→86, 49→98 rejected; 7→14, 33→99, 16→64 all yes).
**Why:** follows [[exp3-iter1-critique-stats]]; refines [[whole-set-round-category]].
**How to apply:** if admitted, expect small LOO gain; don't retry the null variants above.
