---
name: best-explanation-contrast
description: exp3 iter_2 c2: yes = sigmoid(gain·(best concept incl. target − best excl.)); MAP ll −3858 vs ~−3653 incumbent; hard max broke warm-start Hessian
metadata:
  type: project
---
exp3 iter_2 candidate_2 (2026-10-31), best_explanation_contrast: beta-share learner's hypotheses, but p_yes = sigmoid(gain·(max log joint over concepts containing target − max over concepts excluding it)), no averaging.
Hard pt.max → "Hessian not PD" in warm_start_nuts (kinks). Fix: fixed-sharpness soft max (logsumexp(4x)/4) over groups; per-size interval maxima precomputed parameter-free (np.maximum.reduceat over size-sorted intervals).
Smoke check OK (~2 min, ll −3863.9). MAP: gain≈0.6, lambda_0→0.01, ll −3858 (≈200 nats worse than marginal averaging), so max-vs-sum loses loudly.
**How to apply:** don't retry MAP/max-based readouts of the concept learner; marginalisation over concepts is strongly supported. Related: [[generative-next-output-likelihood]].
