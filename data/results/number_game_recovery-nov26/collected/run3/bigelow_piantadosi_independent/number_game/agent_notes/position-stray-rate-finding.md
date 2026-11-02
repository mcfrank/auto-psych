---
name: position-stray-rate-finding
description: separate stray rate for first example vs later ones, +3.7 MAP log-post on costly_squares incumbent; first example MORE often dismissed
metadata:
  type: project
---
Exp2 round 0 (2026-10-30): on costly_squares_final_digit_concepts base, splitting the fixed 0.1 stray rate into first-example vs later-example fitted rates gave MAP log-post -2228.0 vs -2231.6. Fit: stray_first≈0.12 > stray_later≈0.07 (primacy *discount*, not anchoring). Last-vs-earlier split (recency) gave nothing (-2232.6).
**Why:** order-matters slots keep recurring; this tells which direction works.
**How to apply:** fast trick — loglik = member*log((1-e)/size+e/N) + (1-member)*log(e/N), so precompute only membership counts per set. Related: [[forward-counting-tilt-finding]].
