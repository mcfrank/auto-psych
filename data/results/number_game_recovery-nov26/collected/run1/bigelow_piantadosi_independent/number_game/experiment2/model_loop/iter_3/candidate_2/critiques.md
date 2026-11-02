# Critique of `rule_plus_exception_neighbourhood`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **outlier_vs_majority_neighbour_contrast** — In sets of >= 3 examples with exactly one minority-parity example, yes rate for non-example targets within 3 of the outlier minus yes rate for minority-parity (rule-violating) non-example targets within 3 of a majority example; observed > null means the outlier's neighbourhood is privileged over mere proximity to other examples more than modelled, observed < null means people treat proximity to any example alike (interval-like) more than modelled. (observed 0.392 vs null mean 0.179, z=3.14, p=0.002, q=0.016) [survives FDR]
- **lag1_response_autocorr** — Correlation between consecutive responses within a session (sessions split where trial_index resets, rows ordered by trial_index); observed > null means responses are serially perseverative beyond what the trial-independent model produces. (observed 0.137 vs null mean 0.0985, z=2.74, p=0.014, q=0.0559)
