# Critique of `rule_built_tally_switch_triplet`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **participant_majority_agreement_sd** — SD across participants of each participant's rate of agreeing with the pair's majority choice; observed above null means people differ in consistency/noise more than the model allows (it under-produces heterogeneity in decision noise), below means less. (observed 0.15 vs null mean 0.132, z=2.68, p=0.00599, q=0.048) [survives FDR]
