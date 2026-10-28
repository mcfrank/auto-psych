---
name: final-digit-pair-concepts
description: exp3 iter_3 c3: "ending in d or e" rule kind (45 pairs) added to sequence_opens_bottom_ends_top; smoke -3093.3 (same as incumbent); novelty risk
metadata:
  type: project
---
2026-10-22, exp3 iter_3 candidate_3 (`final_digit_pair_concepts`) targets the shared_ones_digit_outside_gap critique (q=0.18, not FDR-significant). It adds the 45 two-digit final-digit unions as a 6th rule kind (Dirichlet a=1). They are plain rules only, not conjunctions: N_BASE_RULES=42 and conj slots use only the base rules. Raw data for 2-distinct-digit sets, targets 1-20 outside the span: yes 0.49 when the target shares a digit (n=95) vs 0.42 when it doesn't (n=240). Check OK. Smoke LL -3093.3 equals the incumbent's, so predictions may barely move. About 250 s per run.
**Why:** It is a direct concept-space answer to the ones-digit critique. **How to apply:** Check the novelty and LOO outcome. If it is rejected, digit-pair concepts are too weak to matter. Related: [[sequence-opens-bottom-ends-top]], [[exp3-iter3-critique-stats]], [[novelty-gate-tiny-tweaks]]
