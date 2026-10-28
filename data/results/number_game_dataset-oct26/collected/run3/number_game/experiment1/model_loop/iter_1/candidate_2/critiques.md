# Critique of `conservative_evidence_interval_concepts`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **single_example_yes_rate** — Yes-rate on trials whose set has exactly one example (target different from it); observed above null means the model generalizes too narrowly from a single example, below means too broadly. (observed 0.44 vs null mean 0.339, z=3.18, p=0.004, q=0.032) [survives FDR]
- **trial_index_drift** — Pearson correlation between trial_index and yes (0/1) across all trials; the model has no session dynamics, so observed away from null means people drift toward more yes (positive) or more no (negative) as the session proceeds. (observed 0.0495 vs null mean 0.0071, z=2.26, p=0.036, q=0.144)
