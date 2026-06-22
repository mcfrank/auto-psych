# Critique of `inner_loop_model`

1 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **max_run_aversion** — Proportion of trials where the sequence with the shorter normalized maximum run is chosen, conditioned on similar alternation rates (|p_alts_a - p_alts_b| <= 0.1). (observed 0.533 vs null mean 0.506, z=2.27, p=0.0299, q=0.149)
