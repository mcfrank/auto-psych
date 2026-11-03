---
name: beta-share-without-replacement
description: exp2 iter_2 c5 repair: inferred stray rate on beta-share collapses (alpha 0.898, RMSE 0.0021 at gate); submitted without-replacement + beta-share (RMSE ≥0.0106, ll −2.8)
metadata:
  type: project
---
exp2 iter_2/candidate_5_repair_1 (2026-10-30). First idea: inferred_stray_rate + beta-share width. Alpha 0.898 ≈ fixed 0.9, so posterior-mean p_yes RMSE vs incumbent was 0.0021 (right at the 0.002 gate). Dropped.
Submitted beta_share_without_replacement_concepts (without_replacement_sampling + beta-share width). 1×300 NUTS: in-sample ll −2267.1 vs incumbent −2264.3; RMSE vs live beta-share models 0.0106–0.0154. Smoke fit 11 s.
**Why:** on exp2 every inferred add-on parameter collapses to the incumbent and fails novelty ([[multiple-base-specificity]], [[beta-share-parity-range]]).
**How to apply:** in a repair slot, use a fixed structural change. Check RMSE against all live models with /tmp-style quick 1-chain fits (~2–5 min each, run in parallel).
