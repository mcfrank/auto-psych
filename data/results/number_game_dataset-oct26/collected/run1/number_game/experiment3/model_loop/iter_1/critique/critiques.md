# Critique of `shortfall_from_mutual_similarity`

2 of 7 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **unanimous_cell_fraction** — Fraction of (set, target) cells with >=5 responses where all responses agree (all yes or all no); observed > null means human judgments are more extreme/deterministic per item than the model's graded probabilities predict (model too lukewarm), observed < null means the reverse. (observed 0.0714 vs null mean 0.054, z=2.19, p=0.042, q=0.147)
- **trial_index_drift** — Pearson correlation between trial_index and the yes response across all trials; observed above (below) null means people drift toward saying yes (no) later in the session, a drift the stationary model cannot produce. (observed 0.0333 vs null mean 0.0111, z=2.08, p=0.03, q=0.147)
