# Critique of `rule_family_prior_tempered_mixture`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **parity_even_minus_odd_asymmetry** — Parity-match effect (yes rate for same-parity minus opposite-parity non-example targets) in all-even sets minus the same effect in all-odd sets (>=2 examples); t_observed differing from null_mean means people treat "even" and "odd" rules asymmetrically, which the model's single shared parity prior cannot capture (above = even rule more favoured than the model allows). (observed 0.0983 vs null mean 0.00961, z=2.32, p=0.026, q=0.112)
- **cell_mixed_response_fraction** — Fraction of distinct (number_set, target) cells whose repeated answers are not unanimous (some yes, some no); t_observed below null_mean means people agree more (more all-or-none) than the model's graded per-cell probabilities predict, above means they disagree more. (observed 0.467 vs null mean 0.513, z=-2.25, p=0.028, q=0.112)
