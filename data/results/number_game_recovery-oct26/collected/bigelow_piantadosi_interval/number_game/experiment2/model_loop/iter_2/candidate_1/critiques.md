# Critique of `first_example_evidence_weighting`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **power_of_two_specificity** — Among sets whose examples are all powers of 2 (2..64), yes rate to non-example powers of 2 minus yes rate to other even non-example targets; observed above the null means people commit to the specific "powers of 2" rule more sharply than the model (which leaks to "even"/ranges), below means the model over-commits to the specific rule. (observed -0.34 vs null mean -0.16, z=-2.21, p=0.014, q=0.112)
- **switch_trial_carryover** — Among consecutive within-participant trial pairs where the example set CHANGES, yes rate after a yes minus yes rate after a no; observed above the null means response-level perseveration that carries across different sets (not explained by repeated-set stimulus similarity), which the trial-independent model under-produces. (observed 0.124 vs null mean 0.0888, z=2.08, p=0.03, q=0.12)
