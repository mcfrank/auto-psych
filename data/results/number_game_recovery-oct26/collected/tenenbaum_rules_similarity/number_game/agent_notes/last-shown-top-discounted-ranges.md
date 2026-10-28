---
name: last-shown-top-discounted-ranges
description: exp3 iter_0 c5 repair: order-only top-marking tweaks fail novelty (RMSE 0.0007-0.0014); graft onto discounted-range model gave MAP RMSE 0.0073 vs incumbent
metadata:
  type: project
---
2026-10-22, exp3 iter_0: two candidates were rejected as near-duplicates of last_example_marks_range_top. Candidate 3 (near_top_closing_example) scored RMSE 0.00065 and candidate 5 (position_weighted_top_marking) scored 0.00140. The candidate_5 repair, `last_shown_top_discounted_ranges`, takes upper_edge_discounted_range_size and replaces its constant top_bonus with the incumbent's last-shown generative credit, log1p(odds*last/exp(log_in)). At MAP: sharpness 0.925, LL -3091.0 vs the incumbent's -3090.5, RMSE 0.0073 vs the incumbent and 0.0099 vs ued.
Check OK after a batched_dot block rewrite: smoke LL -3094.8, 177-184 s per run, gradient 0.049 s (v1 took 0.076 s).
Quick novelty proxy: run find_MAP on both models, then compare p_yes (script in candidate_5_repair_1/scratch/nov/cmp.py). Use replace_rvs_by_values, otherwise p_yes is evaluated from prior draws.
**Why:** Tweaks to the top edge or order alone barely move p_yes. **How to apply:** Run the MAP RMSE proxy before submitting a refinement. Aim for a proxy RMSE of at least 0.005. Related: [[novelty-gate-tiny-tweaks]], [[position-weighted-top-marking]], [[near-top-closing-example]]
