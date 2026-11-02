# Critique of `rule_family_doubt_criterion`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **late_vs_early_discrimination** — Correlation between each answer and its (set,target) cell's mean yes rate in the second half of each session (trial_index>=225) minus the same in the first half; positive vs the null means people get sharper/more consistent with the consensus as the session goes on (negative: noisier), which a static model cannot produce. (observed 0.0354 vs null mean -0.00745, z=3.24, p=0.002, q=0.016) [survives FDR]
