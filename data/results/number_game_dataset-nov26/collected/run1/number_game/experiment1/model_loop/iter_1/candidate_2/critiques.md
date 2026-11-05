# Critique of `place_value_exemplar_similarity`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **between_set_yes_rate_spread** — Standard deviation across distinct number_sets of the per-set yes rate; observed above null means sets differ more in overall endorsement (rule vs similarity sets) than the model predicts. (observed 0.23 vs null mean 0.158, z=4.54, p=0.002, q=0.016) [survives FDR]
- **common_divisor_rule_contrast** — In size>=2 sets whose GCD is >=3 and not 10, yes rate for targets divisible by the GCD minus not divisible (distance>5 from examples); observed above null means the model under-produces multiples-rule generalisation beyond final-digit similarity. (observed -0.0204 vs null mean -0.0937, z=2.51, p=0.016, q=0.0639)
- **parity_rule_contrast** — In all-even sets of size>=2, yes rate for even minus odd targets at distance>10 from every example; observed above null means the model under-produces parity-rule generalisation. (observed 0.291 vs null mean 0.182, z=2.07, p=0.026, q=0.0693)
