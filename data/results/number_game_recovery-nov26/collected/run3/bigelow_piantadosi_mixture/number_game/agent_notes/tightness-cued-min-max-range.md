---
name: tightness-cued-min-max-range
description: exp3 iter_4 c3: tightness cue on variable_min_max incumbent +1.05 MAP, pool RMSE .0073; decade-rounded hull is a dup (RMSE .001)
metadata:
  type: project
---
exp3 iter_4/candidate_3 (2026-11-02) is tightness_cued_min_max_range. It takes variable_min_max_bounded_range and gives each set its own lambda mean, mu - beta*(tight - .34).
- MAP ll: −3316.84, vs −3317.89 for the incumbent. beta = .37 and boost drops to 5.2.
- Pool RMSE vs the incumbent: .0073.
- Smoke check: OK, 87 s of sampling.
- Adding a decade-rounded hull concept (e.g. 37,57 → 31–60) gives boost2 2.5 and ll −3317.94. It is a dup (RMSE .001).
- Scripts: iter_4/candidate_3/scratch/w (mapeval.py, tight.py, dec.py).
See [[tightness-cued-rule-range-expectation]], [[variable-min-max-bounded-range]].
