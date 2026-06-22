# Critique of `bayesian_diagnosticity`

2 of 8 test statistics show a significant discrepancy (p ≤ 0.05), over 200 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **max_run_penalty** — The mean choice rate for sequence A when A has a larger normalized maximum run length than sequence B. (observed 0.427 vs null mean 0.464, z=-2.77, p=0.00995, q=0.0796)
- **exact_symmetry** — The mean choice rate for sequence A when A is exactly symmetric (imbalance 0) and B is slightly imbalanced (>0 to 0.3). (observed 0.367 vs null mean 0.432, z=-2.27, p=0.0398, q=0.159)
