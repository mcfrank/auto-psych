---
name: exception-tolerant-parity-doubt
description: exp3 iter_3 c4 repair: single-exception tolerance grafted on parity_bounded incumbent, smoke -4662.2 (+2.6); WM graft was rejected as duplicate of parity_doubt_three_example_memory
metadata:
  type: project
---
exp3 iter_3 candidate_4_repair_1 (2026-11-02): n-1-of-n property tolerance (from single_exception_tolerance) grafted on parity_bounded_ambiguity_doubt. Check OK, smoke -4662.2 vs incumbent -4664.8.
The first attempt (WM 3-example limit graft) was rejected: p_yes RMSE 0.0014 vs the live parity_doubt_three_example_memory (same graft already admitted).

**Why:** grafts already in the live set get rejected by the novelty gate (RMSE < 0.002).
**How to apply:** before grafting onto the incumbent, check existing_hypotheses.md for a live model with the same graft. Related: [[single-exception-tolerance]] [[parity-doubt-three-example-memory]]
