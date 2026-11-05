# Critique of `variability_tuned_exemplar_attention`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **parity_contrast_far_nodigit** — For sets whose examples all share parity (>=2 examples), yes-rate for same-parity targets minus opposite-parity targets, restricted to targets >10 from every example and sharing no final digit; observed > null means people use parity (even/odd) as a generalization dimension the model lacks. (observed 0.176 vs null mean 0.0367, z=3.79, p=0.002, q=0.016) [survives FDR]
