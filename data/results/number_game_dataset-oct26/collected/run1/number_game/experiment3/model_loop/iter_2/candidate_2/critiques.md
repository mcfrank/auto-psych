# Critique of `round_counting_privileged_factor`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **set_level_yes_rate_dispersion** — Standard deviation across sets (with >=30 responses) of each set's overall yes rate; observed > null means sets differ in how broadly people generalise more than the model's features capture (model too homogeneous across sets), observed < null the reverse. (observed 0.15 vs null mean 0.137, z=1.98, p=0.038, q=0.304)
