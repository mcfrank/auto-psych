# Critique of `first_example_anchored_concepts`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **lag1_response_autocorrelation** — Pearson correlation between a response and the same participant's immediately preceding response (consecutive trial_index); observed above the null means people show sequential yes/no carryover (perseveration) the trial-independent model fails to produce, below means alternation it fails to produce. (observed 0.159 vs null mean 0.127, z=2.09, p=0.036, q=0.288)
