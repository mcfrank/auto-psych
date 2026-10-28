---
name: decade-and-recency-screen
description: MAP screen on conservative_updating_inflated_size base — decade-aligned interval priors are null; recency graft +0.5 nats
metadata:
  type: project
---
On 2026-10-21 (experiment3 iter_3 candidate_5), MAP screens on the conservative_updating_inflated_size base (MAP log-lik -3189.8):
- Interval prior boost for decade-aligned intervals (1-10, 11-20, ... and also the 50-59 "tens digit" style): boost ~0.1-0.8, gain 0.0 nats, p_yes RMSE 0.0002. The critique's decade_boundary_effect cannot be closed via interval priors. Do not retry.
- Grafting the incumbent's last-example recency multiplier: recency ~1.05, gain 0.5 nats, p_yes RMSE 0.0058 vs base. Submitted as conservative_updating_recency_boost.
- Per-cell residuals are dominated by n=5 noise; no clear structure.

**Why:** saves re-screening dead ideas.
**How to apply:** the variant-switch screen script is /tmp/c5s/model_v.py + mapfit.py (may be gone); see [[map-screening-refinements]].
