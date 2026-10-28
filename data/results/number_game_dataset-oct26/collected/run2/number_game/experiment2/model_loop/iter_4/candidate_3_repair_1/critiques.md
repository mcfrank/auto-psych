# Critique of `sequence_momentum_span_exemplar`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **between_set_yes_rate_sd** — Standard deviation across number sets (with >= 18 responses) of each set's overall yes-rate; observed above the null means sets differ in how generously they are generalized more than the model predicts (its set-level breadth is too compressed), below means the model exaggerates set-to-set differences. (observed 0.18 vs null mean 0.158, z=2.73, p=0.00799, q=0.0639)
