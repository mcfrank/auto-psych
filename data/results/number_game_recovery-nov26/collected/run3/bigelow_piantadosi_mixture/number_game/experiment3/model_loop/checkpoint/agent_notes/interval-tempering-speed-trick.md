---
name: interval-tempering-speed-trick
description: How to make a parameter-dependent interval likelihood (e.g. tempered) fast in the number-game Bayesian models
metadata:
  type: project
---
In the number game Bayesian concept models, an interval's log-lik depends only on (n examples, size m, k inside), i.e. 2,000 values. Precompute per-set interval counts per value and per-(set,target) counts containing the target in prepare_observed; in the graph, exp(beta*values) and two dot products. ~1.4 ms/grad vs 37-230 ms for full 100x100 grids or per-row (rows x groups) gathers (machine has 1 CPU, load ~80).
**Why:** partial_size_principle_concepts was rejected as too slow; the naive grid version of conservative_evidence_weighting was 0.23 s/grad.
**How to apply:** any refinement that puts a parameter inside the interval likelihood (round 1, 2026-10-30: iter_1/candidate_1).
Also: to evaluate p_yes at a point, use model.replace_rvs_by_values; compile_fn on p_yes silently draws random RVs.
Outcome (2026-10-30): the size-principle tempering refinement (candidate_4) was rejected as a near-duplicate of the incumbent (p_yes RMSE 0.0015 < 0.002); the fitted exponent stayed near 1. Its repair split the intervals into decade-aligned vs other groups, each with exact precomputed evidence, and fit in about 30 s per 200 draws.
Outcome (2026-10-30): evidence tempering (conservative_evidence_weighting) was rejected as a near-duplicate of the incumbent (p_yes RMSE 0.00116) — the data pin the exponent near 1. Repair swapped it for a linear-in-log-odds report distortion of the posterior (log_odds_compressed_concept_posterior); its smoke fit was fast (20 s) with loglik -832.
Round 2 (2026-10-30, iter_2/candidate_2): response_bias_concept_learner = tenenbaum_digit_rule_inventory + logit-shift yes/no bias; check OK, 32 s per 200 draws, smoke loglik -841.1.
Round 2 (2026-10-30, iter_2/candidate_3): Beta-marginalised noise rate (uncertain_reliability_noise_concepts) also only needs an (n,k,s) table — sum_j C(k,j) s^-j 100^-(n-j) B(u+j,v+n-j)/B(u,v). The incumbent's inferred alpha sits at ~0.91, which is why fixed-rate tweaks are near-duplicates; this one differed (RMSE 0.013 on data trials), in-sample loglik -819.1 vs -820.7, and the check passed (40 s smoke fit). kappa stayed near its prior mean of 10.
