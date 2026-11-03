---
name: exp2-iter2-open-ended-digit-sum
description: Exp2 iter-2 cand_4 — open-ended range discount grafted onto digit_sum_concepts, +2.2 nats MAP (a≈-1.09), submitted
metadata:
  type: project
---
Exp2 iter 2, candidate_4 (2026-10-30), a refinement slot on `digit_sum_concepts`. With the refit MAP, the base NLL is 1925.26 (lapse logit -4.87, guess_yes→1, gamma≈-0.04). Adding the open-ended range prior weight a, with ranges grouped by (width, open flag), gives NLL 1923.10 at a=-1.09. Submitted `open_ended_digit_sum_concepts`. The screen script is /tmp/c4/screen.py (not persistent).
**How to apply:** grafting the endpoint discount stacks roughly additively with digit sums. See [[exp2-iter1-open-ended-range]] and [[exp2-iter1-digit-sum-concepts]].
