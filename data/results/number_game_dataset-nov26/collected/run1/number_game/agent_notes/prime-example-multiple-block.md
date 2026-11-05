---
name: prime-example-multiple-block
description: exp3 iter4 c1 — two-digit prime examples weaken the multiples route on coincidence_rule_violation_route, +9.4 LL / +6.8 logp MAP; smoke -4026.7
metadata:
  type: project
---
Submitted prime_example_multiple_block (2026-11-02): multiples-route weight × exp(-c_prime) when the example is a prime >10. MAP c≈0.95 (route ×0.39).
Variants on the incumbent (MAP LL gain): prime>10 +9.4; prime>7 same; all primes incl 3,5,7 +1.7; "not a 12×12 times-table number" gate +9.4 (same fit); multiplier>5 gate +0.5; two-digit example gate +3.5.
Single-example cells: incumbent predicts ~0.4–0.6 for nearly all of them; the polarization critique comes from item-level variation with only 9 responses per cell.
**Why:** aimed at [[exp3-iter4-critique-stats]] single-example polarization (17→34, 43→86, 23→92 at 0.11 vs 7→14, 16→64, 33→99 at 1.0).
**How to apply:** the prime gate and the times-table-membership gate can't be told apart on this data; a design with composite two-digit examples that aren't table products (26, 34, 38, 46) would separate them.
