# Critique of `inflated_size_familiar_rule_prior`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **pow2_vs_other_even_contrast** — For sets whose examples are all powers of two (e.g. 16,32,2,8; 2,8; 8,64; 64,4), yes rate on non-example powers of two minus yes rate on other even targets; observed above null_mean means people restrict to the small nested rule more sharply than the model (model over-generalizes to the larger "even"/multiples rules), below means the model is too sharp. (observed -0.281 vs null mean -0.127, z=-2.23, p=0.024, q=0.192)
