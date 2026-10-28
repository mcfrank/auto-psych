# Critique of `rule_size_prior_anchored_evidence`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **prev_yes_carryover** — Over consecutive within-session trial pairs (trial_index increments by 1), P(yes | previous yes) minus P(yes | previous no); observed above the null means sequential response perseveration that the trial-independent model under-produces, below means alternation. (observed 0.159 vs null mean 0.128, z=2.07, p=0.038, q=0.304)
