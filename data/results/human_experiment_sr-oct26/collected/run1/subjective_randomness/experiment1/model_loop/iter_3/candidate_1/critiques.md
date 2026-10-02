# Critique of `personal_lapse_ideal_alternation`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **maxrun_pref_equal_alternation** — Among trials with alternation rates within 0.15 of each other but different longest runs, the proportion choosing the sequence with the shorter longest run; observed above null means people penalise long streaks beyond what alternation rate captures (model under-produces streak aversion). (observed 0.688 vs null mean 0.615, z=3.25, p=0.002, q=0.016) [survives FDR]
