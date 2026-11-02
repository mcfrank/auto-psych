---
name: exp3-iter4-reversal-pair-concepts
description: Exp3 iter_4 cand_0 concept-family probes on the equal-prior no-centre base (MAP -3514.51) — digit-reversal pairs {ab,ba} +0.83 (submitted); Fibonacci, triangular, digit product, digit vocabulary dead
metadata:
  type: project
---
2026-11-02, exp3 iter_4 candidate_0 (exploratory, concept-space brief). Each family was added as a 4th Dirichlet kind, with mass shared equally among its members.
- Fibonacci: weight 0. Triangular: +0.44. Fib+tri: +0.2. Digit-product classes: weight 0. Digit vocabulary (numbers using only 1-3 given digits): weight 0.
- Reversal pairs {ab, ba} (36 pairs): LL -3513.68 (+0.83), weight 0.025. MAP pool RMSE is 0.020 vs the incumbent and the round-edge model, so it is clearly novel. Submitted as digit_reversal_pair_concepts.
- Real pairs in the data: 83,38,58,68; 47,2,74; 73,35,53,59. Raw yes rates for reversal targets are mostly ~0.
- Smoke check: 200 iterations in 38 s (speed similar to the incumbent).
Probe: iter_4/candidate_0/scratch/p/fam.py (set the FAM env var) with iter_2/candidate_5_repair_1/scratch/nov.py.
**How to apply:** don't re-probe these families. See [[exp3-concept-family-probes]], [[exp3-iter1-saturation-probes]].
