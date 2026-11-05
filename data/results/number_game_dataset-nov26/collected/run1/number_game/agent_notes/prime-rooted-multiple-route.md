---
name: prime-rooted-multiple-route
description: exp3 iter4 c2: multiples route cut for examples with a prime factor >11, on coincidence_rule_violation_route, +9.4 MAP; smoke -4026.8
metadata:
  type: project
---
Exp3 iter_4 candidate_2 (2026-11-02, exploratory slot, but it is refinement-shaped). `prime_rooted_multiple_route` = the incumbent with its multiples route × exp(-c_rough·[largest prime factor of example > 11]).
MAP log-lik: incumbent -4027.9, gate at >11 -4018.5 (c_rough≈0.83), gate at >7 -4024.7. Replacing the multiples route with times-table co-membership (k 2-12, ×≤10 or 12) scored -4089 and -4096, so it is bad (as in [[log-distance-gradient-exemplar]]).
Polarization critique barely moves: expected single-example polar 0.126→0.141, observed 0.248. The single-example cells are very noisy item by item (13→91 at 0.67 but 13→78 at 0.11), which looks like cell-level overdispersion rather than a missing systematic feature.
Screening scripts: /tmp/c2scr/screen.py and pol.py (may be gone).
**Why:** follows [[exp3-iter4-critique-stats]].
**How to apply:** don't expect a feature to close the single-example polarization gap. Cell-level random effects would probably fail the novelty check, because they predict the same as the base model on new stimuli.
