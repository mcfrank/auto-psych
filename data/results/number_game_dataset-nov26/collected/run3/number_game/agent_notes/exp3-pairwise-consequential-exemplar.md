---
name: exp3-pairwise-consequential-exemplar
description: iter_4 cand_1 exemplar model with Shepard consequential-region pairwise similarity — best similarity-family fit so far (~4306 nll), still ~170 behind concept learners
metadata:
  type: project
---
iter_4 cand_1 (2026-11-03, brief = exemplar/prototype): pairwise_consequential_exemplar. s(t,e)=P(t in C | e in C) over 45 rules (5 family weights) + all intervals (prior ∝ L^p, count tensor PAIR_COUNT 10000x100), size principle 1/|h|^tau; p=sigmoid(c + b·log mean_e s). 9 params.
- Cell-level MLE nll: mean 4305.7, power-mean 4303.6 (q .58), geometric 4336.8, max 4342, min 4583. Previous similarity models ~4368; incumbent MAP ~3958.
- Check OK, smoke ll -4310, 22 s per 100/100 chain (very fast). Scratch iter_4/candidate_1/scratch/mle.py.
Related: [[experiment3-data-patterns]]
