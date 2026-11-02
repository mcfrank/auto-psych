---
name: exp3-iter3-critique-stats
description: exp3 iter_3 PPC stats on round_landmark_veto; endpoints 1/100 -0.13, single-digit after two-digit sets -0.10; 15 participants in 3 design groups
metadata:
  type: project
---
2026-10-31, exp3 iter_3 critique of round_landmark_veto. Observed values: targets 1/100 minus other targets in the same sets is -0.131. Single-digit targets after all-two-digit sets is -0.103. Target nearer the first example minus nearer the last is +0.042. Round targets when a round example is present is +0.031. Unsupported round targets inside the range minus outside is 0.003. Largest-gap fill is +0.021. Adjacent (distance 1) minus distance 2-5 is +0.02. Within-design participant yes-rate SD is 0.013.
Data quirk: trial_index restarts give 15 participants × 450 trials, in 3 groups of 5 that saw identical stimuli. The PPC keeps row order, so participants can be recovered from the trial_index restarts.
**Why:** these are candidate discrepancies; iter_3/critique/ppc_results.json says which ones are significant.
**How to apply:** if the endpoint or single-digit stats are significant, try a scale-endpoint or digit-count category effect. Related: [[exp3-iter2-critique-stats]], [[round-landmark-veto-exp3]].
