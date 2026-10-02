# Critique of `mirror_symmetry_streak_aversion_lapse`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **participant_majority_agreement_sd** — SD across participants of each person's rate of agreeing with the pair-wise majority choice (majority computed from the same dataset); observed above the null means people differ in consistency/engagement more than the model's person-level weights and lapse rates produce, below means the model over-produces individual differences. (observed 0.177 vs null mean 0.162, z=2.21, p=0.028, q=0.213)
