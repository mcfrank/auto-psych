# Critique of `rule_plus_exception_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **parity_outlier_neighbour_yes** — In sets of >= 3 examples where exactly one example differs in parity from all others, yes rate for non-example targets within 3 of that off-parity example; observed > null means people generalise around the exception (treat it as an interval seed) more than the rule-plus-exception model, observed < null means they reject its neighbours more firmly. (observed 0.683 vs null mean 0.539, z=2.63, p=0.02, q=0.16)
