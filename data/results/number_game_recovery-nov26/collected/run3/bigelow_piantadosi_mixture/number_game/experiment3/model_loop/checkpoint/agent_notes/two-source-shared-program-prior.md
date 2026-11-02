---
name: two-source-shared-program-prior
description: exp3 iter_0 c5: second source of two_source may be a range (prior = learner's own lambda_0 split); 0 params, ties MAP (-0.28), RMSE .004; no binding signal in exp3
metadata:
  type: project
---
exp3 iter_0/candidate_5 (2026-11-01), two_source_shared_program_prior. It refines two_source_program_concepts: the second program is a rule with weight mix*lambda_0/45 or a range with weight mix*(1-lambda_0)/100^2, and glitches take 1-mix. On exp3 data the MAP ll is -3322.96, against -3322.68 for the original two_source (mix .035 there, .06 here). p_yes RMSE against the original is .004 and the max difference .022. Smoke run: 7 s, check OK.
The exp3 critique (rule members inside the hull, q=.29, not FDR-significant) was being handled by siblings: c2 with a concept-gated exemplar distance, and c1/c3 apparently with range-restricted rules.
Final-digit binding check on the exp3 data: tens-digit targets after clean ends-in-d sets have obs .160 and pred .164, so there is no signal.
Harness: scratch/work/map.py and resid.py. Use per-obs logp, not compile_fn on p_yes. See [[two-source-program-concepts]], [[check-sibling-candidates-first]].
