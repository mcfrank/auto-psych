# Critique of `gamblers_second_order_person_lapse`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **participant_agreement_variance** — Variance across participants of each person's rate of agreeing with the pair's majority choice (pairs oriented canonically, majority from the data at hand); observed above null means people differ in consistency more than the person-lapse model produces, below means the lapse population over-disperses consistency. (observed 0.02 vs null mean 0.0162, z=2.69, p=0.00599, q=0.048) [survives FDR]
- **short_pairs_more_switches** — Among pairs of length 2-3 with differing switch counts, the rate of choosing the sequence with more switches; observed above null means people prefer alternation in very short sequences more than the model's length-scaled evidence predicts, below means less. (observed 0.744 vs null mean 0.778, z=-2.16, p=0.036, q=0.144)
