# Critique of `iter0_candidate0`

3 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **length_penalty** — Choice rate of sequence A when it is shorter than B, controlling for alternation rate and proportion of heads. (observed 0.383 vs null mean 0.526, z=-3.87, p=0.00995, q=0.0398) [survives FDR]
- **rep_motifs_preference** — Choice rate of sequence A when it has more repeated motifs than B, controlling for alternation rate and proportion of heads. (observed 0.707 vs null mean 0.52, z=3.09, p=0.00995, q=0.0398) [survives FDR]
- **imbalance_preference** — Choice rate of sequence A when it has an intermediate imbalance (0.5) and B is more balanced, to check if the model's quadratic penalty on p captures human preferences. (observed 0.74 vs null mean 0.649, z=2.22, p=0.0199, q=0.0531)
