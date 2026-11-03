---
name: set-level-strong-weak-sampling
description: exp2 iter_4 c4: whole-set (not per-example) strong/weak sampling mixture on the incumbent — theta≈0.86±0.11, LOO tied, RMSE 0.0055
metadata:
  type: project
---
exp2 iter_4/candidate_4 (2026-10-31), submitted as set_level_strong_weak_sampling. P(X|h) = theta·∏strong + (1−theta)·∏weak (strays 0.1 kept per example), using pt.logaddexp on the incumbent's (size, k) grouped tables, so it is as fast as the incumbent (smoke ≈3–4 min, 2x300 fit 126 s).
Quick 2x300 NUTS: LOO −2273.50 vs incumbent −2273.49; theta 0.86±0.11; p_yes RMSE vs incumbent 0.0055, max 0.014 (on data rows).
**Why:** another sampling-assumption null, as [[mixed-strong-weak-sampling]] predicted; the exp2 data can't separate these.
**How to apply:** don't spend more slots on sampling-mixture variants on exp2; differences would only show on 4-example nested-concept sets in new designs.
