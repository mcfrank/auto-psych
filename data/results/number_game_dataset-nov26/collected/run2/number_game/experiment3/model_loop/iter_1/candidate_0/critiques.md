# Critique of `single_example_random_program_doubt`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **n1_set_mean_variance** — Variance across one-example sets of each set's mean yes rate; observed > null means people's breadth of generalization differs more between lone examples than the model's single global doubt/flat-rate mixture allows (observed < null: model over-differentiates lone examples). (observed 0.0261 vs null mean 0.0143, z=3.49, p=0.004, q=0.032) [survives FDR]
- **n1_parity_match_gap** — For one-example sets, yes rate on targets with the example's parity minus yes rate on opposite-parity targets; observed > null means the model's flat-rate doubt dilutes property-based (parity) generalization from a lone example too much (observed < null: model over-weights parity). (observed 0.188 vs null mean 0.0861, z=2.10, p=0.032, q=0.128)
