---
name: exp2-iter1-open-ended-range
description: Exp2 iter-1 cand_5 — 10 participants in 2 cell groups (exp1 cells + new); edge/fuzzy/endpoint screens null; open-ended range discount +1.6 nats submitted
metadata:
  type: project
---
Exp2 iter 1, candidate_5 (2026-10-30). This was a refinement slot built on `relative_width_range_prior`, whose MAP NLL is 1925.84.

**Data layout:** exp2 has 10 participants, 450 trials each. Participants 0-4 answer the exp1 cells (yes rate about 0.25; NLL 903, the same as exp1). Participants 5-9 answer new cells (yes rate about 0.58). Only 5 cells are shared between the two groups. The model already gets both groups' means right, so the critique's "excess disagreement" is not evidence of person-level differences.

MAP screens, each adding one parameter (NLL):
- symmetric fuzzy range edges: 1925.6
- below-only fuzzy edges: 1925.4
- Weber fuzzy edges: 1925.2
- range prior on log(lo): 1924.9
- mid^-k prior added: 1926.4 at best
- lo==1 boost: 1924.6
- prior weight on ranges starting at 1 or ending at 100: **1924.26**, with a = -0.95 (a discount)

Submitted `open_ended_range_discount`, the last of these. Prediction RMSE vs the parent is 0.012. The smoke loglik was -1927.3, and the check took about 75 s. Scripts are in /tmp/c5 (not persistent).

**How to apply:** the range-shape slot is saturated; don't re-screen edge fuzziness or endpoint priors. See [[exp2-iter1-critique]] and [[exp2-iter0-relative-width-prior]].
