# Critique of `linear_accumulated_typicality`

1 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **max_run_penalty** — Proportion of times sequence A is chosen when it has a longer max run than B, but similar alternation rates (abs(p_alts_a - p_alts_b) <= 0.1). (observed 0.378 vs null mean 0.524, z=-2.43, p=0.00995, q=0.0796)
