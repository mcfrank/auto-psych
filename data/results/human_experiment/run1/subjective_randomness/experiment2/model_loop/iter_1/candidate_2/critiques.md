# Critique of `inner_loop_model`

2 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **len_pref_under_alt** — Rate of choosing the longer sequence when both sequences have very low alternation rates (p_alts < 0.3) and unequal lengths. (observed 0.656 vs null mean 0.539, z=2.89, p=0.00995, q=0.0796)
- **pref_over_alt** — Mean rate of choosing sequence A when A is highly over-alternating (p_alts > 0.7) and B is normally alternating (0.4 <= p_alts <= 0.6). (observed 0.583 vs null mean 0.51, z=2.11, p=0.0398, q=0.159)
