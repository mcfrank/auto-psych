---
name: iter4-sampling-weight-screens
description: Iter-4 cand_1 screens of non-uniform production weights and session drift on the WOR incumbent (NLL 902.77); best +0.56 nats; submitted round_number_salient_sampling
metadata:
  type: project
---
Iter 4, candidate_1 (2026-10-30). MAP screens on concepts_sampled_without_replacement (NLL 902.77). Each changes how the program is believed to emit numbers, with weighted WOR draws.
- Magnitude weight x^-beta: uniform is sharply best (beta ±0.25 costs 2-4 nats).
- Round numbers (multiples of 5) weighted exp(b), with weighted strays: b≈-0.4 gives 902.21.
- Multiples of 10 and single-digit weights: at most +0.3.
- Drift of rule-vs-interval log-odds with trial_index: no gain. The novelty pool also has no trial_index, so trial-dependent models can't pass the novelty gate.

Submitted `round_number_salient_sampling`. It uses Chebyshev tabulation in b (degree 16, error 1e-10), takes about 2 s per 200 iterations, has smoke loglik -904.1 and novelty RMSE 0.011.

**Why:** this confirms [[iter3-noise-ceiling-wor]].
**How to apply:** don't re-screen production-weight variants. Scripts are in iter_4/candidate_1/scratch.
