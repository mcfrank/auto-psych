# Critique of `person_pattern_sensitivity_rule_built`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **agreement_sd_structurally_neutral** — SD across participants of the rate of agreeing with the pair majority, restricted to pairs where both sequences have equal three-flip chunk variety and neither/both are rule-built (so pattern sensitivity cannot act); observed above null means person heterogeneity in consistency persists outside pattern structure (model under-produces it there), below means it over-produces it. (observed 0.224 vs null mean 0.194, z=1.93, p=0.04, q=0.32)
