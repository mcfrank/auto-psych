---
name: coincidence-sharpened-exemplar
description: iter_4 c0 — global similarity sharpening by suspicious coincidence of shared property, -9.7 nats MLE on shared-factor incumbent; smoke -1549.0
metadata:
  type: project
---
iter_4 candidate_0 (2026-11-02), `coincidence_sharpened_exemplar`: on shared_factor_parity_exemplar (MLE NLL 1554.5), all dimension penalties are multiplied by exp(k*S). S = (n-1)*log(size ratio) of the most specific shared property (parity 2, mult-of-5 5, final digit 10), divided by 3 log 10. NLL 1544.8 (k≈0.84). Other one-change screens on the same base: log-scale magnitude 1551.0, mod-5 digit kinship 1552.8, mean/max similarity ~0, digit-run count-scaled attention worse. Digit-run criterion shift still gives 1539.1 on this base, but that is a graft of the existing digit_run model.
**Why:** aimed at the iter_4 critiques (digit-run nonmatch, mult-of-5 contrast, set-level SD). **How to apply:** see [[iter4-critique-stats]], [[digit-run-strict-criterion]].
