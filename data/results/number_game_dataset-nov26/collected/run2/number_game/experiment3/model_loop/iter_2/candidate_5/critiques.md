# Critique of `rule_ambiguity_single_doubt`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **n1_set_mean_variance** — Variance across one-example sets of each set's mean yes rate; observed > null means people's breadth of generalization still differs more between lone examples than the richness-gated doubt allows (observed < null: model over-differentiates lone examples). (observed 0.0261 vs null mean 0.0154, z=3.11, p=0.016, q=0.128)
