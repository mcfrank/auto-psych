# Critique of `inner_loop_model`

3 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **alt_motifs_preference** — Correlation between the preference for sequence A and the difference in alternating motifs (A - B), for equal length sequences. (observed 0.0921 vs null mean 0.255, z=-3.78, p=0.00995, q=0.0265) [survives FDR]
- **rep_motifs_aversion** — Correlation between the preference for sequence A and the difference in repeating motifs (B - A), for equal length sequences. (observed -0.141 vs null mean -0.02, z=-3.55, p=0.00995, q=0.0265) [survives FDR]
- **max_run_preference** — Correlation between the preference for sequence A and the difference in maximum run lengths (B - A), for equal length sequences. (observed 0.354 vs null mean 0.287, z=2.33, p=0.00995, q=0.0265) [survives FDR]
