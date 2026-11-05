---
name: representativeness-gated-doubt
description: exp3 iter_1 c0 — n=1 flat-rate doubt gated by how conspicuous the lone example is; smoke -4683.6 vs incumbent -4682.4, ~0
metadata:
  type: project
---
Tried 2026-11-02 (experiment3 iter_1/candidate_0, representativeness_gated_random_doubt). On single_example_random_program_doubt, I replaced the Dirichlet mixture with doubt = sigmoid(base - gain*patternedness). Patternedness is log(100/size of the smallest conspicuous category the example is in: square/cube/power/x10/x11/1,100). The model passed the check. Smoke loglik was -4683.6, against the incumbent's -4682.4, so no visible gain.
**Why:** it targeted the n1_set_mean_variance critique. Most n=1 sets have only 1-3 targets x 9 answers, so set-mean variance is mostly item/binomial noise.
**How to apply:** don't expect example-level gating of n=1 doubt to pay off. Related: [[single-example-random-doubt]], [[exp3-n1-data-facts]]
