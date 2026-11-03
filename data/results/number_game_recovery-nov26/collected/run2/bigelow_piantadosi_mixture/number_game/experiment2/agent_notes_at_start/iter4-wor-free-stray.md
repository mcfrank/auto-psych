---
name: iter4-wor-free-stray
description: Iter-4 cand_4 — free stray rate on the WOR incumbent gives +1.2 nats MAP (stray≈0.12); in-set-target discount +0.3; exact rules + Chebyshev intervals at 3 ms/grad
metadata:
  type: project
---
Iter 4, candidate_4 (2026-10-30). I MAP-screened variants of `concepts_sampled_without_replacement`, which has NLL 902.77:
- Free stray rate: 901.56, with stray≈0.12 and lapse falling to ≈0.013.
- Discount on "yes" for targets that are themselves examples: 902.44. These are 50 trials with an observed yes rate of 0.68.

I submitted `without_replacement_inferred_stray_rate`. Its prediction RMSE against the incumbent is 0.0125 at stray 0.12, and its smoke loglik was -903.9. It computes the rule likelihoods exactly in pytensor, as (sets, pos, rules) tensors. The interval log-evidence per set and the interval mass per row are tabulated as a degree-24 Chebyshev series in log(stray). That costs 22 s of prep and 3 ms per gradient. The screener is in iter_4/candidate_4/scratch/s/.

**Why:** consistent with [[iter3-free-outlier]] and [[number-game-ceiling]]: about 1 nat of gain at best.
**How to apply:** don't re-screen these; reuse the Chebyshev template ([[iter3-process-screens]]).
