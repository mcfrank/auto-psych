# Critique of `monotone_order_range_prior_shift`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **rule_member_out_minus_in_range** — For sets of >=3 examples consistent with a parity or multiples-of-m (3-12) rule, among non-example targets that belong to the smallest such consistent rule, yes rate outside the examples' min-max range minus inside; the model's rule/interval mixture predicts a negative gap, so observed above null means people extend broad arithmetic rules beyond the example range as readily as within it (e.g. 63,27,99,33 -> 3; 69,9,39,21 -> 72-96). (observed -0.187 vs null mean -0.122, z=-2.08, p=0.022, q=0.176)
