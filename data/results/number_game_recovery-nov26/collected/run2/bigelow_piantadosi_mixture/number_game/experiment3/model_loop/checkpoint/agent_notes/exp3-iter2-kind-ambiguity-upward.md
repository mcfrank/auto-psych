---
name: exp3-iter2-kind-ambiguity-upward
description: Exp3 iter-2 cand_5 refinement — kind_ambiguity_aversion + upward counting (incumbent + kappa ambiguity term), smoke LL -3194.0, submitted
metadata:
  type: project
---
Exp3 iter 2, candidate 5 (2026-10-31): `kind_ambiguity_with_upward_counting`. To build it I copied the incumbent `heading_weber_with_upward_counting.py` and added the kappa·4p(1-p) ambiguity factor taken from `kind_ambiguity_aversion`. It adds no extra compute, and the check took about 1.5 min. The smoke fit gave LL -3194.0, and the iter-2 critique found 0/8 significant. I did not check novelty. If kappa→0 it duplicates the incumbent.
**How to apply:** if it is rejected as a dup, the ambiguity term is null at the ceiling; see [[exp3-iter1-kind-ambiguity-aversion]], [[number-game-ceiling]].
