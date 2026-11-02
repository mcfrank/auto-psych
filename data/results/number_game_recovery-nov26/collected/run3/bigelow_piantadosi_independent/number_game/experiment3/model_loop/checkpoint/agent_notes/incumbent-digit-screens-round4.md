---
name: incumbent-digit-screens-round4
description: MAP screens on multiplicative_not_exponential_concepts; ends-in-d bonus +26 nats, contains-digit cost +16, both together +27
metadata:
  type: project
---
Round 4 (2026-10-30), cell_1/experiment1, MAP screens on the incumbent `multiplicative_not_exponential_concepts` (base loglik -825.3):
- "ends in d" log-prior bonus: -799.5. The bonus fits at about 1.8 and the exponent cost at about 4.2. Submitted as `final_digit_multiplicative_concepts` (iter_4 candidate_3).
- Extra cost on contains-digit rules alone: -809.2 (cost about 1.4).
- Both together: -798.7, so the contains cost adds only 0.9 once the ends bonus is in.
- Lapse fits at about 0 at MAP.

**Why:** the critique says people use "contains digit d" less than the model predicts (0.43 vs 0.61), and the ends-in bonus accounts for most of that gap.
**How to apply:** don't re-screen these. See [[final-digit-salience-finding]] and [[exponent-concept-cost-finding]].
