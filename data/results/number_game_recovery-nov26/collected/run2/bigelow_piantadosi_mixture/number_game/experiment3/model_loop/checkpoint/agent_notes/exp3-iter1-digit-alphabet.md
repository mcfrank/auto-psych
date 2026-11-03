---
name: exp3-iter1-digit-alphabet
description: Exp3 iter-1 cand_1 — digit-alphabet concept family (numbers using only digits D, |D|≤4, 384 concepts) on the Weber-heading incumbent; smoke -3195.0, submitted
metadata:
  type: project
---
Exp3 iter 1, candidate_1 (2026-10-31), an exploratory slot. The critique found 0 of 8 stats significant.

Raw check against the incumbent at its MAP:
- reversal targets: 255 rows, obs 0.463 vs pred 0.432
- targets built from the examples' digits: obs 0.387 vs pred 0.392

Both are basically fit already.

I submitted `digit_alphabet_concepts`: the incumbent plus a third finite family with its own log_alpha_share ~ N(-3, 1.5), mixed alongside the rules and the digit sums. Smoke loglik was -3195.0 and the check took about 2 s per run. The MAP share screen (scratch/s/fam.py, Nelder-Mead) was too slow (>5 min), so I stopped it with no result. Novelty is unknown, and there is a risk it is a dup if the share shrinks; see [[exp3-decade-stated-prior]] and [[exp3-iter0-concept-family-screens]].
