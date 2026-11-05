---
name: parity-bounded-ambiguity-doubt
description: exp3 iter_2 c5 repair — parity kept in doubt mode at all sizes on rule_ambiguity_doubt_all_sizes: smoke -4664.8 (~+1 vs parent)
metadata:
  type: project
---
exp3 iter_2 candidate_5_repair_1 (2026-11-02): `parity_bounded_ambiguity_doubt`. It is rule_ambiguity_doubt_all_sizes, except that the doubt mode's flat yes is discounted by parity_keep for targets that break the parity all the examples share (at any n). Smoke log-likelihood -4664.8, against the parent's -4665.6.
The first attempt in this slot was rejected because it was an exact copy of rule_ambiguity_parity_kept_doubt (RMSE 0).
Data: targets breaking the shared parity get .40/.22/.13/.27 yes for n=1/2/3/4, against .59/.43/.43/.40 for the other targets.

**Why:** before writing a refinement, check that its mechanism differs from every live model, not only the pruned ones.
**How to apply:** a parity graft on the all-sizes doubt is done. Related: [[rule-ambiguity-all-sizes]] [[parity-preserving-random-doubt]]
