# Critique of `range_limited_rule_familiar_strays`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **narrow_rule_inhull_nonmember_span_contrast** — For multi-example sets sharing a multiple-of-k (k>=2) or odd rule, yes-rate for rule NON-members inside the examples' span, wide-span (>40) minus narrow-span (<=40) sets; observed > null means people fill in wide spans with non-members more than the model predicts. (observed -0.305 vs null mean -0.196, z=-2.53, p=0.00999, q=0.0799)
