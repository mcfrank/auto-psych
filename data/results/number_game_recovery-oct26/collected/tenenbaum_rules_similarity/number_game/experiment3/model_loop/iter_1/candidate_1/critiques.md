# Critique of `last_example_marks_range_top`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **parity_match_outside_span_gap** — In sets of >=2 whose examples all share a parity, yes-rate for targets outside the span with the same parity minus opposite parity; observed above null means people use the parity rule beyond the range more than the model (which favors ranges), below null means the model over-weights parity rules. (observed 0.248 vs null mean 0.184, z=2.62, p=0.012, q=0.0959)
