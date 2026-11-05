---
name: multiple-route-exemplar
description: exp2 iter1 c0 — additive "target is a multiple of an example" similarity route on common_divisor_sharpened incumbent, ~+24 nats MAP
metadata:
  type: project
---
Experiment 2, iter_1 candidate_0 (2026-11-02), `multiple_route_exemplar`. Change to the incumbent: per-example similarity = exp(log_sim) + exp(log_w*gain)*[t is a multiple of e, t!=e, e>1]. This is an additive route that works at any distance, not a penalty. MAP log-lik was -2690.5, against -2714.4 for the incumbent. The smoke fit was -2695.2.
Screen results:
- Ungated version: -2693.7.
- Multiple-or-divisor: -2715.
- gcd>=3 additive: -2711.
- Graded log-gcd additive: -2707.
- Weight ∝ e/10: -2706.
- Dropping the set-divisor penalty: -2707.5.
- Penalty variants (single-example gcd<3 penalty, with or without its own coefficient; pairwise gcd penalty): at most +2.
Residuals showed the incumbent predicted background yes ≈0.01 after single examples, while people said yes 0.3–0.7. A penalty can't lift far targets.
**Why:** this targets the critique stats single_example_shared_factor_contrast and background_set_rate_sd. See [[exp2-iter1-critique-stats]] and [[common-divisor-sharpened-exemplar]].
**How to apply:** many sets ("28,98", "68,58", "84,56", "69,9,39,21") are still under-predicted on background targets.
