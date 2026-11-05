---
name: shared-feature-contrast
description: exp2 iter_0 c1 exploratory — pure Tversky common-feature model (no exemplar similarity), 3 params, smoke log-lik -2833.4
metadata:
  type: project
---
experiment2 iter_0 candidate_1 (2026-11-02), `shared_feature_contrast`: features all examples share (parity, mult 3-12, final digit, decade, square/cube/pow2/prime), each weighted n*log(100/|f|); p_yes = sigmoid(a + w_m*match - w_v*violation). No magnitude distance beyond decade. Smoke in-sample log-lik -2833.4 vs incumbent coincidence_sharpened_exemplar ELPD -2761.6 (~9 params).
**Why:** brief asked for simpler/high-variance; aimed at set-level SD and gcd-divisible contrast critiques ([[exp2-iter0-critique-stats]]).
**How to apply:** if it does respectably, a refinement adding graded magnitude may be worth trying; distinct from pruned [[shared-regularity-screen]] which grafted onto exemplar.
