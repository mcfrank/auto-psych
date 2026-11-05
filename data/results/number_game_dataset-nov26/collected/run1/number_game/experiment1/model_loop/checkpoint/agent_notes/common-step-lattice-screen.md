---
name: common-step-lattice-screen
description: iter_4 c1 MLE screen on shared_factor incumbent (base NLL 1554.5) — common-step lattice -5.2 submitted; most other new-info features ~0
metadata:
  type: project
---
iter_4 candidate_1 (2026-11-02), in-sample NLL added to `shared_factor_parity_exemplar` (base 1554.5):
- off-step penalty (>=3 examples, gcd of gaps 2..20, target not reachable in whole steps): 1549.3 → submitted as `common_step_lattice_exemplar` (check OK, smoke LL -1553.0). Without the n>=3 restriction: 1549.8.
- digit_run binary criterion on this base: 1539.1 (still the strongest single fix); a size-principle rule-evidence criterion (n-1)*log(100/|smallest rule|): 1545.8, but it only works via the digit runs
- multiple-of-5 mismatch dim 1554.5, its tuned version 1551.6; 0/5 digit kinship 0; leading-digit match 1554.2; any shared digit 1553.5; tens digit 0; inside-span bonus 1554.1
The biggest residuals are 3,63→36/39/30, 6,66→60 and 83,38,58,68→8x, which are under-predicted.
**Why:** saves re-screening. **How to apply:** see [[shared-factor-exemplar-screen]], [[digit-run-strict-criterion]], [[iter4-critique-stats]].
