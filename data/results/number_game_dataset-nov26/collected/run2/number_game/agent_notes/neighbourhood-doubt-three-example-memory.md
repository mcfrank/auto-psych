---
name: neighbourhood-doubt-three-example-memory
description: exp3 iter_4 c4 repair — magnitude-faded multi-set doubt fallback grafted on WM incumbent; the first tolerance graft was rejected as a duplicate (RMSE .0005)
metadata:
  type: project
---
exp3 iter_4 candidate_4_repair_1 (2026-11-02): the first attempt grafted exception tolerance onto parity_doubt_three_example_memory. Admission rejected it as a duplicate of exception_tolerant_three_example_memory, with p_yes RMSE 0.00054.
The repair grafts the Gaussian doubt-fallback fade outside the range (from [[magnitude-bounded-ambiguity-doubt]]) onto the incumbent. It is named neighbourhood_doubt_three_example_memory.
A residual check on the base model's predictions showed "multiple of an example" and "divides an example" are already captured (|z|<1).
**Why:** before retrying a graft, check whether a sibling slot already submitted the same graft.
**How to apply:** look at the existing_hypotheses "no comparison row" entries before picking a graft. Related [[exception-tolerant-three-example-memory]].
