# Critique of `recency_weighted_exemplar_similarity`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **common_factor_rule_gap** — Among multi-example sets whose examples share a greatest common divisor g>=3, the yes rate for targets divisible by g minus the yes rate for targets not divisible by g; observed > model means people generalize a shared "multiples of g" rule more than pairwise exemplar divisibility/digit similarity predicts. (observed 0.197 vs null mean 0.0731, z=2.77, p=0.004, q=0.032) [survives FDR]
- **trial_index_drift** — Pearson correlation between trial_index and the yes response across all trials; a nonzero observed value vs a near-zero model value means the yes tendency drifts over the session (fatigue/learning), which the stationary model cannot produce. (observed 0.0417 vs null mean -0.00116, z=2.25, p=0.028, q=0.112)
