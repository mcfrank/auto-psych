---
name: uncertainty-gated-default-answer
description: exp3 iter_3 c2 — default answer w.p. k*4b(1-b) on span_interval_similarity inference: smoke -4715.5, ~50 behind incumbent
metadata:
  type: project
---
exp3 iter_3 candidate_2 (2026-11-02), `uncertainty_gated_default_answer`, a decision-rule slot. Inference is span_interval_similarity unchanged. Decision: p = (1-k u) b + k u g, with u = 4b(1-b), k ~ Beta(2,2) and default yes g ~ Beta(2,2). The check passed in about 30 s per chain. Smoke log-lik -4715.5, against -4664.8 for the incumbent parity_bounded_ambiguity_doubt.
**Why:** a decision rule on top of simple inference does not replace the n=1 doubt mechanisms.
**How to apply:** don't expect decision-rule-only slots on simple inference to beat the incumbent. Related: [[guessing-lapse-ceiling]] [[span-interval-similarity]]
