---
name: pedagogical-teacher-sampling
description: exp2 iter_2 cand1 — teacher sampling P(x|h)∝[P(x|h)/P(x)]^beta on beta-share base: beta≈1.02, LOO −6.5 vs incumbent; prefix-sum rectangle trick gives 6 ms grads
metadata:
  type: project
---
experiment2 iter_2/candidate_1 (2026-10-30), pedagogical_teacher_sampling_concepts, in response to the nested_multiple_specificity critique (obs 0.49 vs null 0.39, q=0.28).
Likelihood: the teacher picks x ∝ P(h|x)^beta; P(h) cancels, so it is P(x|h)^beta·P(x)^-beta / Z_h, with P(x) = the learner's prior-marginal predictive.
A 2×(500+500) NUTS fit: beta 1.018±0.019, 0 divergences, trial-wise elpd_loo −2279.65 vs the incumbent's −2273.11 under the same settings (6.5 nats worse). Submitted anyway (exploratory slot).
Speed: a dense 119 sets × 5050 intervals layout costs 164 ms per logp+grad. Exact fix: an interval's example count k is constant on (lo-segment × hi-segment) rectangles, so put per-(n,k) interval weights on a 100×100 grid, take 2-D prefix sums, and gather 4 signed corners per rectangle. That gives 5.8 ms, matching to 1e-13. Usable for any model whose interval weights depend on position (not just size).
**Why:** down-weighting examples that are common under the marginal hurts overall fit more than it helps nested-multiple sets.
**How to apply:** don't retry teacher/pedagogical sampling variants on exp2; reuse the rectangle trick, see [[example-recall-subset-mixture]], [[experiment2-round0-readout-and-sampling]].
