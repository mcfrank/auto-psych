---
name: exp2-set-feature-contrast
description: Exp2 iter-1 Tversky set-level feature-contrast similarity (shared-rule features + hull distance) ~-2470 LL, ~540 nats behind Bayesian models
metadata:
  type: project
---
Exp2 iter-1 cand_0 (2026-10-30): `set_common_feature_contrast`. Logit = a + w_c·Σkept shared-rule info^rho − w_d·Σlacked info^rho − w_mag·dist-outside-span, lapse at 0.5. MAP: rho=1 → -2516, rho=2 → -2474; smoke fit LL -2466 (6 params, 3 s). Hull-distance variants (d/10, d/(span+10), log1p d) all within 2.5 nats of each other. Lapse fits about 0.3.

**Why:** similarity even to set-shared features can't give the size-principle sharpening; still far behind the ~-1931 ELPD models.
**How to apply:** when a similarity slot comes up, expect a loss of about 500 nats or more; submit it as a valid contrast model and don't spend time tuning it. Related: [[concept-exemplar-summed-similarity]], [[all-or-none-rule-commitment-fails]].

Repair (2026-10-30): the free exponent rho traded off against w_common/w_distinct (R-hat 1.19, ESS 14). Fixing rho=2, precomputing the summed weights per trial (weights scaled to max 1) and leaving 5 params gave R-hat 1.0 and ESS >4000 in 19 s with 4 chains. Posterior: w_common≈7, w_distinct≈10, lapse≈0.28. **How to apply:** in similarity models, don't free an exponent on top of linear gain weights.
