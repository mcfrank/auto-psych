# Critique of `simplicity_cost_concept_search`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **same_decade_yes_rate** — Yes rate for non-example targets in the same decade (same tens digit, 1-100 with 100 in its own decade) as at least one example minus yes rate for targets in a decade containing no example; observed > null means people use decade/tens-digit similarity more than the model, observed < null means less. (observed 0.045 vs null mean 0.0948, z=-2.65, p=0.016, q=0.104)
- **unanimity_rate** — Fraction of distinct (set, target, trial_index) items whose repeated responses are all identical (all yes or all no), among items answered 2+ times; observed > null means people agree more (model's predicted probabilities are too intermediate/uncertain), observed < null means the model is too deterministic. (observed 0.556 vs null mean 0.496, z=2.32, p=0.026, q=0.104)
- **even_set_odd_target_yes_rate** — Yes rate for odd targets after sets whose examples are all even (2+ examples); observed > null means the model over-commits to parity/even rules and people accept parity-violating numbers more, observed < null means people respect parity more strictly than the model. (observed 0.0923 vs null mean 0.126, z=-2.14, p=0.046, q=0.123)
