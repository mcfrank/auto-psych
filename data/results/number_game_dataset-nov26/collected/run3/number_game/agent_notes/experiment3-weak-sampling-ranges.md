---
name: experiment3-weak-sampling-ranges
description: iter_4 cand_2 eliminative_range_checking — weak sampling (fixed reference width) for ranges on magnitude_threshold_concepts; tiny MAP gain, does NOT move critique stats
metadata:
  type: project
---
iter_4 cand_2 (2026-11-03): on magnitude_threshold_concepts (MAP -3958.2), interval likelihood replaced by n_in*log(.9/W+.001)+n_out*log(.001) (W inferred, logW~N(log30,1)). With range_evidence_shift kept: -3952.5 (W~129); with the shift removed (same param count as the incumbent, submitted): -3956.9 (W~84, lambda_0 .84). The gain is diffuse (~2 nats on PPC-relevant sets). It does NOT move the iter_4 critique stats: wide-minus-mid span in-range .062 (obs .125), mixed-parity in-minus-out .03 (obs .119), 2-ex out .262 (obs .222), set SD .188 (obs .231). Rules (lambda_0 ~.85-.9) dominate the posterior, so changing range likelihoods hardly matters. Check OK, 180 s per 100/100 chain.
**Why:** range-likelihood tweaks look saturated; the span and mixed-parity gaps need something that shifts belief from rules-with-strays to ranges.
**How to apply:** don't retry range-evidence variants; target the rule/range prior split per set instead. Scratch: experiment3/model_loop/iter_4/candidate_2/scratch (weak.py, weak2.py, map.py).
Related: [[experiment3-threshold-concepts]], [[experiment3-data-patterns]]
