---
name: noisy-criterion-decision-rule
description: exp2 iter_1 (2026-11-01): a Beta-distributed yes/no criterion, P(yes)=I_p(a,b), collapses to probability matching
metadata:
  type: project
---
Tested in exp2 iter_1/candidate_1 (noisy_criterion_concept_judgment). The inference was simple: family-only rule weights and a single inferred stray rate eps≈0.09. On top of that, P(yes)=betainc(mu*kappa,(1-mu)*kappa, belief).
- MAP fit: mu=0.50, kappa≈1.96, so a≈b≈1. That is the identity, i.e. probability matching. The ll was -2035.54, against -2035.59 for the identity readout.
- The incumbent family_pooled_uncertain_reliability is already almost perfectly calibrated (binned p_yes matches the observed rate from 0.007 to 0.99), so no decision-stage distortion has anything to fix.
- pt.betainc has working gradients for a, b and x, at about 3.5 ms for 2000 values.
**How to apply:** do not spend decision-rule slots (lapse, bias, threshold, matching) on this data; they collapse. See [[exp2-incumbent-refinements]].
Speed: the first version, which gathered lik_table[set_n] and rule_post[cell_set], took 0.22 s/iter, about 25 min for a 7000-iter chain. Flattening the (k, size) counts and using pt.dot with one-hot matrices for n and for cell→set brought it to 0.07 s/iter, with about 15 NUTS steps and no divergences. See [[fast-incumbent-sampling]].
