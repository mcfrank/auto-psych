---
name: gist-memory-min-max-range
description: exp3 iter_4 c5: min-max hull boost grafted on gist_memory_examples, MAP −3318.27 (+1.4 over gist, −0.4 vs incumbent), pool RMSE .0052
metadata:
  type: project
---
exp3 iter_4/candidate_5 (2026-11-02) is gist_memory_min_max_range: the hull boost from min_max_bounded_range on the gist_memory_examples base.
- How it is built: the hull is an extra interval [min, max] with prior boost·(1−λ)/N_INT, and gist traces support it through the overlap of their window with the hull. hull_coef is (S,P,T), and cell_range_coef gets a third T-block.
- MAP ll: −3318.27, vs −3319.67 for gist and −3317.89 for the incumbent variable_min_max (the incumbent also has a lambda_sd parameter). boost = 6.6, verbatim = .975.
- Pool RMSE vs the incumbent: .0052.
- Smoke sampling: 136 s; the full check took ~10.5 min.
- Not tried: adding variable lambda on top, which would make the gist tables ~9× costlier.
See [[gist-memory-examples]], [[variable-min-max-bounded-range]].
