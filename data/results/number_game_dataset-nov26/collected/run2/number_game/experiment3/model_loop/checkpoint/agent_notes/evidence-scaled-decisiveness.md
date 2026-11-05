---
name: evidence-scaled-decisiveness
description: decision temperature as power of n examples on exp3 span_interval_similarity: +2.3 MAP only, power ~0.2
metadata:
  type: project
---
Tested 2026-11-02 (experiment3 iter_0/candidate_0, evidence_scaled_decisiveness). Took the exp3 incumbent span_interval_similarity and multiplied its logit by n^g (n = number of examples, centred at 2.5). MAP went from -4709.4 to -4707.0, with g about 0.2 (in the predicted direction). The smoke log-likelihood was -4716.5 and the check took about 1 minute.
**Why:** this was a decision-rule slot aimed at the single_vs_multi_yes_gap critique. In exp3 the yes rate is 0.58 for n=1 and about 0.40 for n>=2, but flattening toward 0.5 explains little of that gap.
**How to apply:** don't expect slope/temperature changes by set size to close the n=1 gap. An additive shift worked better in exp2 ([[single-example-leniency]]).
