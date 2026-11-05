---
name: rule-ambiguous-parity-doubt
description: exp3 iter_2 c5 — parity_preserving doubt + richness-gated doubt share (graft of incumbent): smoke -4674.5 vs incumbent -4675.4, ~0
metadata:
  type: project
---
exp3 iter_2 candidate_5 (2026-11-02): parity_preserving_random_doubt with doubt = sigmoid(d0 + b*(category count - 6)) and a separate random_yes Beta, replacing the Dirichlet n1_mix. Smoke fit -4674.5, only ~1 nat better than [[rule-ambiguity-single-doubt]] (-4675.4). Sampled fast (25s for 200 iterations).

**Why:** stacks the two near-tied n=1 doubt refinements. The gain is small because parity and richness are both proxies for the same odd-multiple-of-5 sets ([[parity-preserving-random-doubt]]).
**How to apply:** expect n=1 doubt refinements to have mostly run out. Look for gains on multi-example sets instead.
