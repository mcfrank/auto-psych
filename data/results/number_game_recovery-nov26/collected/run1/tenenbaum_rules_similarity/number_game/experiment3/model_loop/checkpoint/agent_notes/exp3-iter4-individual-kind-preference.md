---
name: exp3-iter4-individual-kind-preference
description: Exp3 iter_4 cand_5 — participant-level (hierarchical) kind prior on equal_prior_rules_round_edge_ranges gives +13.7 nats MAP (-3500.1 vs -3513.7), by far the biggest gain since the noise ceiling; 34 ms/grad
metadata:
  type: project
---
2026-11-02, exp3 iter_4 candidate_5 (refinement slot). Participant = cumsum(diff(trial_index)<0) (15 people), see [[exp3-iter3-critique-participant-structure]].
- Each participant gets own logits for (digit sums, ranges) vs familiar rules: kind_mu + kind_sd*z, non-centred. MAP kind_sd ≈ [1.87 dsum, 0.99 ranges], kind_mu [-2.2, 0.14].
- Trick: within-kind evidence and predictions are per set (global params only); per-row mix via kind posterior softmax(log_w[part] + log_z[set]). 34 ms/grad.
- Population-level stimulus-only models were saturated (~1 nat gains); individual differences in rule-vs-range preference are where the remaining signal is.
**How to apply:** next probes should be participant-level variants (per-person interval_scale, per-person noise) rather than more concept families. Scripts: iter_4/candidate_5/scratch/p (resid.py MAP + residuals, tg.py timing).

**Repair (2026-11-02, candidate_5_repair_1):** the original was rejected as too slow (34 ms/grad at 4 parallel chains ≈ 0.8 s/iter; production is 7000 iter/chain at ta 0.99, 30-min cap). Fix: per-person fixed-effect range logits Normal(0,2) (no hierarchy), and evaluate only at the 1227 unique (set,target) pairs, as mat-vec products with precomputed exp-evidence (pair×300 range groups, pair×18 dsum) times exp(prior weights). Familiar rules become constants. 3.9 ms/grad single-thread; 4-chain 500/500 ta0.99 in 159 s, R-hat 1.00, 0 div, ~30 leapfrog steps. MAP -3506.0. Gathering param-dependent tensors (AdvancedIncSubtensor1 grad) is the slow op in pytensor, so avoid it.
