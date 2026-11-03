---
name: concept-family-map-screen
description: MAP screen (exp3 iter_4/cand_0) of new rule families added to salient_contained_digit_rules — repdigits +11 nats, decades +6, others null
metadata:
  type: project
---
2026-10-31, experiment3 iter_4/candidate_0 (concept-space brief). Base salient_contained_digit_rules MAP log-lik -2536.19. Each family was added as rules with its own kappa_x (variant generator + outputs in iter_4/candidate_0/scratch/v/, run with iter_3/candidate_3/scratch/map.py):
repdigits {11..99} -2524.99 (kappa_x 4.8); decades (tens digit t) -2530.5 (kappa_x 2.8); digit sums -2534.8; multiples of 15/20/25/50 -2535.8; triangular+fibonacci -2536.5; two-digit-set pairs -2536.2. Cued equal-step progressions (gcd of differences 3..12, share theta) fit theta=0 -> null.
Submitted salient_repdigit_concept (check OK, 54 s sampling, smoke log-lik -2529.0).
**Why:** saves future slots from re-screening dead families.
**How to apply:** decades are the next-best untried family; progressions/residues are dead on this data. Related: [[contained-digit-salience-gain]], [[interval-prefix-sum-speedup]].
