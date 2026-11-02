# Critique of `no_digit_containment_span_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **upward_minus_downward_extrapolation** — Yes rate on targets 1-15 above the largest example minus yes rate on targets 1-15 below the smallest example (sets of 2+ examples); observed above null means people extend ranges upward more (or downward less) than the model's symmetric interval/rule account predicts. (observed -0.129 vs null mean -0.0884, z=-2.25, p=0.024, q=0.192)
