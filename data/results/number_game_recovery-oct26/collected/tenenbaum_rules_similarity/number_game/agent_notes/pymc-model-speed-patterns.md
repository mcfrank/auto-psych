---
name: pymc-model-speed-patterns
description: Number-game PyMC speed tips — admission needs smoke sampling well under ~200 s/run; use count-matrix matmul, not (set,slot,hyp) tensors; SAMPLER_SETTINGS target_accept allowed
metadata:
  type: feedback
---
2026-10-19, exp2 iter_2 candidate_1 (pedagogical_teacher_sampling, likelihood depends on parameters for all 5092 hypotheses). First attempt: a (set, slot, hyp) tensor, 0.13 s per gradient, smoke sampling 394 s. It passed the check but was REJECTED at admission as too slow: still sampling after 30 min at target_accept 0.99. So passing the smoke check does not show a model is fast enough.

The repair: a per-number gain table gain[x, h] = log1p(K * appeal_x / norm_h) * member, then set_ll = counts(set x 100) @ gain. This is exact, because the epsilon floor is a per-set constant. That got about 0.07 s per gradient. float32 barely helped. A module-level `SAMPLER_SETTINGS = {"target_accept": 0.9}` is honoured, and the warm-start dense metric makes it safe. Result: smoke sampling about 110-130 s per run, about 0.55 s per iteration, with 282 s for the whole check.

**Why:** Admission's fit must finish in 30 min, and a parameter-dependent likelihood over every interval can't be precomputed. Models with smoke runs of about 190 s per sampling run have passed admission, and 394 s has not.
**How to apply:** Time compile_dlogp before the check (aim for <0.05 s). Keep each smoke sampling run under about 150 s. Don't name scratch scripts after stdlib modules (timeit.py broke the pymc import). Run long checks in the background. Related: [[write-candidate-first]].
