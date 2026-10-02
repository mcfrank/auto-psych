# Critique of `complement_symmetry_aversion`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **participant_majority_agreement_variance** — Variance across participants of each participant's rate of agreeing with the pair's overall majority choice (pair identified regardless of side); observed above null_mean means people differ in decisiveness/consistency more than the model's personal sensitivity and lapse terms produce, below means the model over-disperses individuals. (observed 0.0262 vs null mean 0.0211, z=3.09, p=0.00599, q=0.048) [survives FDR]
