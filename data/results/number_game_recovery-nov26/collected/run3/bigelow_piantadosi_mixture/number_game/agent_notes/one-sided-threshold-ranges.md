---
name: one-sided-threshold-ranges
description: exp3 iter_1 c3: interval-prior variants on range_limited incumbent; width power γ→0, Erlang −4 ll, one-sided [1,b]/[a,100] share tau≈.03 +0.4 MAP (submitted), round-decade ranges tie
metadata:
  type: project
---
exp3 iter_1/candidate_3 (2026-11-01). The incumbent was range_limited_rule_familiar_strays (MAP ll −3320.79). The critique was narrow_rule_inhull_nonmember_span_contrast. The gap is mostly on narrow-span sets: in-span non-members get yes at 0.58 observed vs 0.41 predicted. The biggest residual is 31,3,1,15 → 2, 4, 16, which got 5/5 yes.
MAP screens of the interval prior:
- Width power w^γ: γ = −0.03, a tie, RMSE .002 (near-dup).
- Erlang width prior: σ≈39, −3324.9.
- Decade-aligned ranges: tau .08, −3320.79, a tie.
- One-sided "below b / above a" ranges with an extra share tau: tau .03, −3320.39, RMSE .009 vs the incumbent on the training stimuli. Submitted as one_sided_threshold_ranges. Check OK in 27 s.
None of these closes the narrow-span gap: the model's yes rate for those non-members only moves .41 → .42.
**How to apply:** interval-prior reshaping is exhausted on exp3 too ([[normative-distortions-collapse]]). Look elsewhere for the 31,3,1,15 fill-in.
