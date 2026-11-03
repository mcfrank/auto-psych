---
name: exp3-iter0-concept-family-screens
description: Exp3 iter-0 concept-family MAP screens on heading incumbent; near-misses/named sequences null, decades +0.5, submitted decade_named_concepts
metadata:
  type: project
---
Exp3 iter 0, candidate 1 (2026-10-31). The incumbent `heading_direction_range_extrapolation` has a MAP NLL of 3193.6 over 1307 cells. Its MAP lapse goes to 0, and that matters for screening: a family screen with the lapse held at 0 grossly exaggerates the losses. Refit lapse and guess_yes jointly with the family share. The lapse refit alone is worth about 2.5 nats, so subtract that from any family's gain.

Net gains with the lapse refit:
- Near-misses (rule ∪ {x} and rule \ {x}): -0.01, null.
- Named sequences (triangular, Fibonacci, 2^n±1, n²±1, factorials): 0.
- Decades: 0.5.
- Digit product: 1.35.
- Digit difference: 2.1, but almost all of it comes from the one set 85,19,91, so it is spurious.

I submitted `decade_named_concepts`: the incumbent plus 10 decade concepts (1-9, 10-19, ..., 90-99) with log_decade_share ~ N(-5, 1.5). Smoke loglik was -3194.4 and the check took about 1 min. Novelty RMSE against the incumbent is 0.0057 at share -6.2. The screener is at experiment3/model_loop/iter_0/candidate_1/scratch/screen/ (screen.py and fam4.py).

**Why:** this confirms [[number-game-ceiling]] and [[exp2-iter1-digit-sum-concepts]].
**How to apply:** don't re-screen these families on exp3. Exp3 sets include 2^n±1 sequences (33,17,5,9 and 31,3,1,15), but the targets don't probe those concepts.
