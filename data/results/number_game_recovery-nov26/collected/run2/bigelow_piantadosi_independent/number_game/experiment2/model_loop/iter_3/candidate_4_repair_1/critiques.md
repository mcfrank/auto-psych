# Critique of `base_ten_multiples_capacity_limited`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **two_example_dip** — Yes rate (non-example targets) on 2-example sets minus the mean of the yes rates on 3- and 4-example sets; observed below null means people are more conservative after two examples, relative to larger sets, than the model's monotone capacity limit predicts. (observed -0.0481 vs null mean -0.00795, z=-2.37, p=0.02, q=0.0879)
- **pure_parity_opposite_parity_yes** — In sets of >=2 examples that are all odd or all even, yes rate for targets of the opposite parity; observed below null means people exclude parity-violating targets more strictly than the model predicts (model over-generalizes across parity). (observed 0.111 vs null mean 0.136, z=-2.22, p=0.022, q=0.0879)
