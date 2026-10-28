# Critique of `session_growing_acquiescence_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **single_example_nonmember_yes_rate** — Yes-rate for non-example targets on single-example sets; observed > model means people generalize more broadly from one example than the model's size-principle posterior (with n**gamma evidence weighting) predicts, observed < model means more conservative. (observed 0.452 vs null mean 0.385, z=3.15, p=0.004, q=0.032) [survives FDR]
