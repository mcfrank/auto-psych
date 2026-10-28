---
name: decision-rule-mechanisms-null
description: Experiment2 data: incumbent posterior is already calibrated; yes-bias, lapse, log-odds warping, Luce sharpening, per-block bias all gain <2 nats
metadata:
  type: project
---
Checked 2026-10-19 (experiment2, iter_1 candidate_1). Posterior-mean p_yes of prime_concept_excludes_one is calibrated in every bin, by set size, and for in-set targets (obs .860 vs pred .861). MAP prototypes on a rules+intervals base (no conjunctions, prime fix): plain 2053.9 NLL; log-odds slope+bias, lapse, posterior exponent (Luce) all converge to the null (~2053.9). Thresholding generative probability instead of membership is far worse (2337.9). The 10 trial_index blocks need no separate bias (total gain 1.9 nats); previous-answer dependence is tiny (slight alternation).

Also: the no-conjunction base with the prime fix matches the incumbent's fit (MAP ~ -2054) yet samples about 50x faster (13s vs 636s per smoke fit), so the conjunction share (~1%) contributes almost nothing.

**Why:** Saves rerunning these checks; brief slots asking for decision-rule mechanisms will tie at best.
**How to apply:** Look for gains in the inference/hypothesis space, not the response stage. find_MAP on the incumbent gets stuck (NLL 2471); use Powell over a numpy reimplementation, or NUTS. Related: [[prime-rule-includes-one]].
