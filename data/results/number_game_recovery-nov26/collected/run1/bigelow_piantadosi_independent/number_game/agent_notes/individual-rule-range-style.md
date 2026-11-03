---
name: individual-rule-range-style
description: exp3 iter_4 MAP screen on incumbent salient_contained_digit_rules — nested tweaks ~+1.5 nats; per-participant rule-vs-range prior share +17 nats
metadata:
  type: project
---
2026-10-31, iter_4/candidate_4. MAP log-lik on salient_contained_digit_rules: base -2536.2; special-number family kappa -2534.7; decade "in the d0s" rules + kappa_t -2535.0; odds power -2534.6; per-participant lambda_0 (non-centred logit, lam_sd fit 1.3, participants from trial_index resets, see [[participant-segmentation]]) -2519.1. Submitted as individual_rule_vs_range_style. Note alpha (rule+exception prior share) fits ~0.001 on the incumbent: exception concepts are nearly unused now.
**Why:** individual differences are a larger untapped source than more concept-space tweaks.
**How to apply:** lambda factors out of rule vs range sums (drop log lambda from log_rules/log_cls/log_size_prior, mix per row), so it costs nothing extra. Scripts: iter_4/candidate_4/scratch/map (mkP.py). Next ideas: per-participant beta or eps. Related: [[contained-digit-salience-gain]].
Speed gotcha: the hierarchical non-centred version (lam_mu, lam_sd, 64 lam_z) sampled at 300 s per 200 iterations (deep trees, same 10 ms gradient); independent per-participant logits Normal(-0.5, 1.5), 32 slots, with unknown rows using the mean of the slots that hold data took 68 s. Check OK.
