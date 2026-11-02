---
name: powers-of-2-37-misfit
description: Incumbent family-prior models over-predict "yes" to 37 after powers-of-2 sets; splitting exception rules into own family gains ~8 nats MAP
metadata:
  type: project
---
In experiment1 (round 2, 2026-10-30), the incumbent erlang_interval_family_prior_concepts gave target 37 p≈0.35 after powers-of-2 sets; observed 0/25. Putting "powers of 2, not 32"/"or 37" in their own family raised MAP ll from -1075.5 to -1067.0 (candidate iter_2/candidate_5, arbitrary_exception_rule_prior).
Things that did NOT help at MAP: rule∩range conjunction concepts (-0.8), linear-in-log-odds response (+2 for 2 params), changing fixed noise ALPHA from 0.9.
Remaining misfit: predictions in the 0.2–0.4 band run ~0.07 too high (wide-range concepts extending beyond the examples).
**Why:** saves re-running diagnostics next round.
**How to apply:** start refinements from these residuals; see [[incumbent-residual-diagnostics]].
Round 3 (2026-10-30): gamma-width prior + exception family (iter_3/candidate_5, gamma_width_exception_family_concepts) MAP ll -1064.6 vs incumbent -1067.0 (+1 param; shape≈4.6, mean width≈53). Largest remaining per-set deviance: 4,66,78,96 (~-30, flat ~0.37 predicted, noisy data), 31,3,1,15, 23,35,95,83.

Round 3 (2026-10-30): generative "next-output probability relative to max, ^sensitivity" judgment on the incumbent learner gave MAP ll -1123 (vs -1067) — membership-probability reporting is clearly better; don't retry size-principle-on-target readouts.

Round 4 (2026-10-30): iter_4/candidate_3 = gamma_width_parity_range_concepts (incumbent parity_within_range_concepts + gamma width prior from gamma_width_exception_family_concepts). Smoke-fit ll -1066.8 (100 draws); passes check in ~75 s.
