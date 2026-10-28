# Critique of `sequence_opens_bottom_ends_top`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **shared_ones_digit_outside_gap** — In sets of >=2 examples, among non-example targets 1-20 outside the span, yes-rate for targets sharing a ones digit with some example minus those sharing none; observed above null means people use digit-level similarity to examples beyond what the model's ends-in-d rules (which need all examples to share the digit) capture. (observed 0.314 vs null mean 0.262, z=2.54, p=0.022, q=0.176)
