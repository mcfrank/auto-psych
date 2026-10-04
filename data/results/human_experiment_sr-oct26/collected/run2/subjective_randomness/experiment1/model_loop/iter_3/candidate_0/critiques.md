# Critique of `length_scaled_alternation_ideal`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **shorter_longest_run_pref** — Among pairs with equal switch counts but different longest-run lengths, rate of choosing the sequence with the shorter longest run; observed above null_mean means the model under-penalises long streaks beyond what switch rate (and periodicity) explain. (observed 0.588 vs null mean 0.448, z=2.55, p=0.018, q=0.144)
