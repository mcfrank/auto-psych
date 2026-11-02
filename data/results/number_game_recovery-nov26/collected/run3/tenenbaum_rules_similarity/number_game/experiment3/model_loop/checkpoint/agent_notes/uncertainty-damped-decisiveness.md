---
name: uncertainty-damped-decisiveness
description: exp3 iter_0 c4: entropy scales logit gain instead of shifting criterion; damping≈0.03±0.13, LOO −0.7 vs incumbent (null)
metadata:
  type: project
---
exp3 iter_0 candidate_4 (2026-10-31), refining `uncertainty_shifted_criterion`: logit_yes = exp(-k·(H−0.5))·logit(p_in), with k ~ N(0,2) and H the normalised posterior entropy. 2×300 fit: k = 0.03±0.13, mean ll −3653.8 vs incumbent −3653.1, quick LOO −3658.4 vs −3657.6. The incumbent's own shift was −0.19±0.14 in the same fit, so neither entropy readout does much on exp3.
Exp3 blocks (5 participants each) see different set lists (75/71/109 sets), so the block yes rates of 0.40/0.58/0.49 come from the stimuli. Within a block, people differ by only ±0.01, so there are no individual differences to model.
**Why:** so later rounds don't retry entropy-modulated gain or slope forms.
**How to apply:** k≈0 collapses to the base beta-share learner, so novelty is at risk. Look for gains outside the entropy readouts. See [[experiment2-decision-rule-nulls]], [[experiment3-iter0-critique-stats]].
