---
name: exp2-iter4-strong-weak-sampling
description: Exp2 iter-4 cand_0 — the margin incumbent with a mixture of strong and weak sampling (logit_strong), smoke -1925.9, submitted; N_TERMS 24→14 halves gradient time
metadata:
  type: project
---
Exp2 iter 4, candidate_0 (2026-10-30). Submitted `uncertain_strong_weak_sampling`. It takes the margin_beyond_examples_range_prior learner and makes the likelihood of the examples under every concept a mixture: strong sampling with weight pi, weak sampling with weight 1-pi. Weak sampling gives an example probability 1/(100-k) if the concept contains it, with 10% strays. Strong and weak each get their own evidence and moment tables, mixed in log space. Smoke LL -1925.9 (incumbent -1925.8). There was no MAP screen, so the effect size is unknown.
Speed: cutting the Taylor terms from 24 to 14 took the gradient from 34 ms to 18 ms, the incumbent's speed, with an unchanged logp. The smoke run took 40 s. The incumbent's 24 terms were overkill for |beta|<3. See [[exp2-iter3-margin-beyond-examples]], [[fit-speed-budget]].
