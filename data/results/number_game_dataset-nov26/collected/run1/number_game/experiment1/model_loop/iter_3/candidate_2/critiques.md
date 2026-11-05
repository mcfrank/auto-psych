# Critique of `graded_parity_variability_exemplar`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **set_level_yes_rate_sd** — Standard deviation across sets (with >=3 distinct targets) of each set's mean yes-rate; observed > null means people's overall willingness to say yes varies by set more than the model's single bias/slope mapping from summed similarity produces (set-specific rule breadth). (observed 0.193 vs null mean 0.15, z=3.02, p=0.002, q=0.016) [survives FDR]
- **digit_homog_near_nomatch_rate** — In sets whose examples (>=2) all share one final digit, yes-rate for targets within 10 of some example that do NOT share that digit; observed < null means people apply the final-digit rule more strictly (rejecting magnitude-near violators) than the model's homogeneity-tuned digit attention predicts, observed > null means magnitude similarity still wins more than predicted. (observed 0.142 vs null mean 0.236, z=-2.61, p=0.016, q=0.0639)
