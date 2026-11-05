---
name: accumulating-regularity-exemplar
description: exp2 iter_0 c5 — shared_regularity_exemplar with rule evidence × (n-1); MLE 2710 vs incumbent 2730; smoke -2724.7
metadata:
  type: project
---
experiment2 iter_0 candidate_5 (2026-11-02), `accumulating_regularity_exemplar`. MLE NLL on exp2 data (5040 rows): shared_factor_parity base 2814, incumbent coincidence_sharpened 2730, pruned shared_regularity 2731, SR with reg×(n-1) 2710 (MAP 2721), reg×sqrt(n-1) 2721. Incumbent+reg graft 2681 (but slot forbade refining the incumbent); SR+coincidence graft 2680. Gcd-rule logit term: only +15 on base, +15 on incumbent.
**Why:** set-level SD and gcd-contrast critiques; rule evidence is the big lever on exp2 data. MLE often goes degenerate (beta→0, c→large: a nearest-neighbour ridge), so check MAP under priors too.
**How to apply:** next round, try incumbent + shared-regularity term if it is allowed. See [[shared-regularity-screen]], [[exp2-iter0-critique-stats]].
