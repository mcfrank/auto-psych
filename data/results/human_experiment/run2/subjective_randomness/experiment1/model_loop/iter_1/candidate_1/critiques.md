# Critique of `iter0_candidate0`

6 of 23 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **fallback_corr_periodicity_b** — Pearson correlation between the response and the `periodicity_b` feature. (observed -0.183 vs null mean -0.095, z=-3.20, p=0.00995, q=0.114)
- **fallback_corr_p_a** — Pearson correlation between the response and the `p_a` feature. (observed -0.0604 vs null mean 0.00166, z=-2.59, p=0.00995, q=0.114)
- **fallback_corr_alts_a** — Pearson correlation between the response and the `alts_a` feature. (observed 0.254 vs null mean 0.191, z=2.31, p=0.0398, q=0.183)
- **fallback_corr_n_a** — Pearson correlation between the response and the `n_a` feature. (observed 0.0167 vs null mean -0.0444, z=2.16, p=0.0299, q=0.183)
- **fallback_corr_alts_b** — Pearson correlation between the response and the `alts_b` feature. (observed -0.241 vs null mean -0.181, z=-2.09, p=0.0398, q=0.183)
- **fallback_corr_h_b** — Pearson correlation between the response and the `h_b` feature. (observed -0.0413 vs null mean 0.00868, z=-1.88, p=0.0498, q=0.191)
