---
name: round-base-multiples-prior
description: Exp3 r4 MAP screen on leading_digit_string_similarity — round-base prior for "multiples of g" gains ~10 nats; primes/powers/repdigit/lattice null
metadata:
  type: project
---
Exp3 round 4 (2026-10-18). Incumbent leading_digit_string_similarity has a MAP log-likelihood of −4630.2. Its residuals show the multiples-of-g rule under-applied for g = 2, 4, 5, 8, 10 (on-target residual +0.05 to +0.13) and over-applied for g = 3, 6, 9 (−0.03 to −0.06).
- Extend the rule to g ≥ 2 and add a signed "round base" prior term, where round means g divides 100 or g is a power of 2: −4619.7 (+10.5). Submitted as round_base_multiples_prior (iter_4/candidate_0).
- The round-base prior alone with g ≥ 3: +6.8. Extending to g ≥ 2 alone: +3.6.
- Null or small: a prime rule (+1.3); a powers-of-b rule (0); repdigit sets (too few data); a step-lattice rule for uncovered sets (roughly 0).
Leftover residuals: target 1 is still over-predicted in many sets; the set-level SD is 0.147 observed vs 0.118 predicted.
**Why:** saves re-screening concept additions.
**How to apply:** the scripts are in iter_4/candidate_0/scratch (mapv.py, grp.py, gg.py, lat.py). See [[exp3-r3-residual-scan]].
