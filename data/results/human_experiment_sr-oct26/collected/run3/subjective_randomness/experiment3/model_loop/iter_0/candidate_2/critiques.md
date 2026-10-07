# Critique of `gamblers_run_second_order_side_habit`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **participant_agreement_variance** — Variance across participants of each person's rate of agreeing with the pair majority choice (heterogeneity of individual judgment sharpness); observed above null means people differ in consistency more than the model's person-level sensitivity produces (e.g. a subgroup of random responders), below means less. (observed 0.02 vs null mean 0.0158, z=3.17, p=0.00799, q=0.0639)
- **short_pairs_more_switches** — Among pairs of length 2-3 with differing switch counts, the rate of choosing the sequence with more switches; observed above null means people prefer alternation in very short sequences more than the model's length-scaled evidence predicts, below means less. (observed 0.744 vs null mean 0.78, z=-2.18, p=0.042, q=0.168)
