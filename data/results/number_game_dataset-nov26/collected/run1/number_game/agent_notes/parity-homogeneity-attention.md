---
name: parity-homogeneity-attention
description: iter_2 cand_1 extends variability-tuned attention with parity attention scaled by example parity homogeneity (targets critique parity_contrast_far_nodigit)
metadata:
  type: project
---
Round 2 critique of best model `variability_tuned_exemplar_attention`: only FDR-surviving miss is parity_contrast_far_nodigit (obs 0.176 vs null 0.037). Candidate `variability_tuned_parity_attention` adds c_par = c_par_max * parity_homog; the smoke fit log-lik was -1565.9 (in-sample, not LOO). Related: [[exemplar-variant-mle-screen]].
