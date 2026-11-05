---
name: common-divisor-sharpened-exemplar
description: exp2 iter_0 c3 — set-level gcd>=3 divisor penalty replacing per-example coprime on coincidence_sharpened incumbent, +16 nats MAP
metadata:
  type: project
---
Experiment 2, iter_0 candidate_3 (2026-11-02): refined `coincidence_sharpened_exemplar` by replacing its per-example coprime penalty with a set-level one. In multi-example sets whose GCD g is ≥3, a target with t%g≠0 loses similarity to every example. In-sample MAP log-lik was -2714.4, against -2730.4 for the incumbent (+16 nats, no extra parameters). Using g≥2 instead gave only +1.7 more. The smoke fit was -2718.6. This targets the critique stat common_divisor_nodigit_contrast. Note that "90, 45" has g=45, but people endorse 30 and 60 there, so strict use of the GCD fails on that set.
**Why:** a set-level arithmetic-rule reading fits better than pairwise coprimality.
**How to apply:** the next step could be a GCD-coincidence term inside the sharpening. See [[coincidence-sharpened-exemplar]] and [[exp2-iter0-critique-stats]].
