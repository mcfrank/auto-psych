# Critique of `span_restricted_rule_concepts`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **nonrule_set_below_minus_above_extension** — In sets of 2+ examples with no small common rule, yes rate for non-example targets 1-15 below the smallest example minus 1-15 above the largest; observed above the null means people extend number ranges downward more than the model's symmetric intervals predict (below = upward more). (observed 0.192 vs null mean 0.0672, z=2.76, p=0.00799, q=0.0639)
- **decade_crossing_near_contrast** — For non-example targets 1-5 from the nearest example, yes rate when the target is in the same tens decade (floor((n-1)/10)) as that example minus when it crosses a decade boundary; observed above the null means people treat decades as category boundaries the model ignores (below = decade boundaries matter less than the model predicts). (observed 0.00542 vs null mean 0.0616, z=-2.29, p=0.026, q=0.104)
