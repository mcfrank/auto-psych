# Critique of `bigelow_piantadosi_mixture`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **numeric_neighbor_yes_rate** — Yes-rate for targets 1-5 away from the nearest example that share no last digit with any example, among sets whose examples span more than 20; observed > null means people use raw numerical similarity beyond what rules and a single interval predict, observed < null means less. (observed 0.22 vs null mean 0.299, z=-4.64, p=0.002, q=0.00799) [survives FDR]
- **cell_response_consistency** — Mean over repeated (set,target) cells (2+ responses) of the within-cell Bernoulli variance p(1-p) of yes responses; observed < null means people agree with each other more (more deterministic answers) than the model's intermediate probabilities imply, observed > null means more disagreement. (observed 0.086 vs null mean 0.0726, z=3.66, p=0.002, q=0.00799) [survives FDR]
- **near_outside_range_yes_rate** — Yes-rate for targets outside [min,max] of the examples but within 10 of that range (excluding targets sharing a last digit with an example), for sets of 2+ examples; observed > null means people extend numerical intervals beyond the examples more than the model, observed < null means less. (observed 0.332 vs null mean 0.399, z=-2.83, p=0.00799, q=0.016) [survives FDR]
- **early_minus_late_yes_rate** — Yes-rate on trials with trial_index < 90 minus yes-rate on later trials; observed > null means people say yes more early in the session than the static model predicts (a time-varying response bias the model lacks). (observed 0.175 vs null mean 0.227, z=-2.73, p=0.00599, q=0.016) [survives FDR]
