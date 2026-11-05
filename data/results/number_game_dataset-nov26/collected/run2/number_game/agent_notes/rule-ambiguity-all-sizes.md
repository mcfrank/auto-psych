---
name: rule-ambiguity-all-sizes
description: exp3 iter_2 c3 — rule-ambiguity doubt extended to multi sets (shared-category count): +10 MAP, smoke -4665.6 vs -4675.4; needs HalfNormal slopes
metadata:
  type: project
---
exp3 iter_2 candidate_3 (2026-11-02): `rule_ambiguity_doubt_all_sizes`. It is rule_ambiguity_single_doubt with flat-rate doubt also on multi-example sets: doubt = sigmoid(dm + dbm*(#categories ALL examples share - 6)), with its own base and slope. MAP went from -4664.6 to -4654.3. The gain comes almost entirely from 4 power-pair sets (64,4; 8,64; 81,9; 96,48), endorsed about .6-.75 where the incumbent predicted about .45-.55.
With Normal slopes the chains switched modes (random_yes near 0 vs .8, R-hat 1.8). HalfNormal slopes converged: 4x400 draws, R-hat 1.01, 0 divergences, random_yes .75.
Variants of the n=1 doubt on the same incumbent all gave about +1 MAP: round-example gate, parity-keep, a richness-dependent flat rate, and a round-dependent flat rate. The n=1 direction is used up.

**Why:** the multi-set mean variance is under-predicted (obs .036 vs pred .019).
**How to apply:** ambiguity-driven flat answering is a lever for multi-example sets. Related: [[rule-ambiguity-single-doubt]] [[exp3-n1-data-facts]]
