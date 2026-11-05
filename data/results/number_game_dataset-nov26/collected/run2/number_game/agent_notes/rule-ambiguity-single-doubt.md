---
name: rule-ambiguity-single-doubt
description: exp3 iter_1 c3 — n=1 doubt share rising with lone example's property-category count: +8 MAP, smoke -4675.4 vs -4682.4; n=1 own slope +3, n=1 parity boost +3
metadata:
  type: project
---
Refinement of single_example_random_program_doubt (2026-11-02): doubt = sigmoid(d0 + b*(#property categories the lone example is in - 6)), flat yes at random_yes. MAP +8.2 (all-families count; arithmetic-only count +7.5). Doubt RISES with richness (b ~ +0.3): rule-ambiguous examples get flatter answers. Converged 2x500, R-hat <= 1.01, 0 divergences with separate doubt/random_yes params.
Other probes on the same incumbent: an own slope for n=1 gave +3 MAP, and a larger n=1 parity salience gave +3.

**Why:** targets the n1_set_mean_variance critique (lone examples differ in breadth).
**How to apply:** the low-mean n=1 sets are odd multiples of 5 (15, 55, 85, 95) asked about even multiples of 10. That is the one place with opposite-parity targets, so parity and mult-5 are confounded there. Related: [[single-example-random-doubt]] [[exp3-n1-data-facts]]
