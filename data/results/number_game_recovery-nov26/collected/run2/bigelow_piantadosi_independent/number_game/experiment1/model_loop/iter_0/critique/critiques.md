# Critique of `bigelow_piantadosi_mixture`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **parity_match_gap** — Among sets whose examples share one parity (all even or all odd, >=2 examples), yes rate for targets matching that parity minus non-matching; t_observed above null_mean means people weight parity more than the model. (observed 0.419 vs null mean 0.167, z=14.61, p=0.002, q=0.004) [survives FDR]
- **last_digit_share_yes_rate** — Yes rate for non-example targets sharing the last digit with at least one but not all examples (multi-example sets); t_observed above null_mean means people use partial surface-digit similarity the model's all-or-none rules ignore. (observed 0.512 vs null mean 0.339, z=8.52, p=0.002, q=0.004) [survives FDR]
- **far_target_yes_rate** — Yes rate for targets more than 25 away from every example in the set; t_observed above null_mean means people give more distant/rule-based (or noise) yeses than the model predicts. (observed 0.178 vs null mean 0.118, z=5.04, p=0.002, q=0.004) [survives FDR]
- **near_target_yes_rate** — Yes rate for non-example targets within 1-5 of the nearest example; t_observed above null_mean means people show stronger similarity (numerical proximity) generalization than the model. (observed 0.312 vs null mean 0.397, z=-4.77, p=0.002, q=0.004) [survives FDR]
