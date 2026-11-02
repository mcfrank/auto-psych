---
name: typical-width-magnitude-bands
description: exp3 iter_1 c0: Erlang width prior on ranges (rule_count base) loses 5 MAP ll (sigma≈19); exponential loses 9; uniform per-range prior already favours narrow
metadata:
  type: project
---
exp3 iter_1/candidate_0 (2026-11-01), typical_width_magnitude_bands. It targets the critique narrow_rule_inhull_nonmember_span_contrast. For in-span rule non-members, the incumbent's yes-rate is too low on narrow spans (obs .52 vs pred .41) and right on wide spans (.22 vs .21). The narrow gap comes mostly from 31,3,1,15 (2, 4, 16 said yes) and from 66,78.
- Erlang(size; sigma) / (101 - size) per range, on rule_count_familiar_strays: MAP ll -3326.7 against -3321.8, sigma 19. The narrow-span prediction only moves .41 → .43.
- An exponential width density does worse: -3330.6, sigma 37.
- Uniform per-range priors already imply a size mass ∝ (101 - s), so width priors cannot close the gap. Submitted anyway; smoke fit 31 s, check OK.
- Implementation: run_count tables (sets, runs, 100) as data, with pt.dot against the weighted size^-j inside the model.
See [[compressed-number-line-ranges]], [[range-limited-rule-familiar-strays]].
