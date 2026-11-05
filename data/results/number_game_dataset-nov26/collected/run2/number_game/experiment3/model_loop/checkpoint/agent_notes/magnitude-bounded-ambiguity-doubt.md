---
name: magnitude-bounded-ambiguity-doubt
description: exp3 iter_4 c5 — on exception_tolerant_parity_doubt, multi-set doubt fallback fading with distance outside the range: +2.7 loglik MAP; tolerance variants all ~0
metadata:
  type: project
---
exp3 iter_4 candidate_5 (2026-11-02), a refinement slot. Base: exception_tolerant_parity_doubt, MAP logp -4685.4, loglik -4650.2. Exception tolerance was fit at only ~0.07.
- Separate tolerance for n=3 and n=4: +1 loglik.
- Size-principle weighting of near-shared properties: 0, RMSE 0.0007, so it would be a duplicate.
- Per-example decline in multi-set doubt: 0.
- Doubt fallback also keeps the shared final digit: 0.
- Doubt fallback on multi sets fades with a Gaussian of distance outside the range (width ~0.31): +2.7 loglik, RMSE 0.013. Submitted as `magnitude_bounded_ambiguity_doubt`.
The base model's residuals are dominated by individual sets (n=4 set z^2 72.6 over 31 sets). The n3/n4 patterns are carried by single big sets such as 48,99,3 with 558 trials.

**Why:** the exception tolerance mechanism is exhausted.
**How to apply:** don't refine tolerance further. Related: [[exception-tolerant-parity-doubt]] [[parity-bounded-ambiguity-doubt]]
