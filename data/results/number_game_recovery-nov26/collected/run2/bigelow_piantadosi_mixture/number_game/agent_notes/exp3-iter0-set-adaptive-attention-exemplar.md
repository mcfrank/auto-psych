---
name: exp3-iter0-set-adaptive-attention-exemplar
description: Exp3 iter-0 exemplar slot — GCM with set-adaptive attention (softmax of -focus × example dispersion over 12 number dimensions); smoke LL -4240 vs best ELPD -3200
metadata:
  type: project
---
Exp3 iter-0 candidate 2 (2026-10-31): `set_adaptive_attention_exemplar`. There are 12 mismatch dimensions: magnitude 1-exp(-|d|/10), parity, mod 3/4/5, last digit, tens digit, digit sum, shared digit, prime, square, pow2. Attention = softmax(-focus·mean pairwise example mismatch). Response = sigmoid(bias + gain·10·(mean sim-0.5)). It has 4 params, prepare_observed takes (trials×4×12) arrays, and the smoke fit of 200 iters takes 7 s. Smoke LL is -4240, about 1000 nats behind the Bayesian models, which matches the pattern that similarity models lose badly.
I submitted it without tuning. Related: [[exp2-iter4-prototype-ratio-similarity]], [[concept-exemplar-summed-similarity]].
