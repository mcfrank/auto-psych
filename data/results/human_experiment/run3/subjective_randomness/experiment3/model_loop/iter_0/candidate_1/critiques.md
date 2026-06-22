# Critique of `bayesian_diagnosticity`

4 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **pref_longer_max_run_matched_rep_motifs** — Rate of choosing the sequence with a longer maximum run, conditioned on identical number of repetition motifs. (observed 0.451 vs null mean 0.498, z=-4.11, p=0.00995, q=0.0796)
- **pref_longer_max_run** — Overall rate of choosing the sequence with a longer maximum run length. (observed 0.504 vs null mean 0.524, z=-2.79, p=0.0199, q=0.0796)
- **pref_more_alternations** — Overall rate of choosing the sequence with more overall alternations (alts). (observed 0.654 vs null mean 0.637, z=2.26, p=0.0398, q=0.0796)
- **pref_more_alts_matched_alt_motifs** — Rate of choosing the sequence with more alternations, conditioned on identical number of alternating motifs. (observed 0.623 vs null mean 0.603, z=1.97, p=0.0398, q=0.0796)
