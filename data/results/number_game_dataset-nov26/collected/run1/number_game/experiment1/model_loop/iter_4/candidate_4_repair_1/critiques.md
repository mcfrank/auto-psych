# Critique of `shared_factor_parity_exemplar`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **digit_run_nonmatch_rate** — Yes-rate, over all trials of sets with >=3 examples that all share one final digit, for targets that do NOT have that final digit; observed < null means people treat a run of same-final-digit examples as a strict rule and reject non-matching targets more than the model predicts, observed > null means more lenient than predicted. (observed 0.135 vs null mean 0.236, z=-3.04, p=0.002, q=0.016) [survives FDR]
- **multiple_of_five_set_contrast** — In sets with >=2 examples that are all multiples of 5, mean yes-rate for targets that are multiples of 5 minus mean yes-rate for targets that are not; observed > null means people extend a multiples-of-5 rule more sharply than the model's final-digit/magnitude similarity predicts, observed < null means the model over-generalizes along it. (observed 0.57 vs null mean 0.36, z=2.91, p=0.00599, q=0.0213) [survives FDR]
- **set_level_yes_rate_sd** — SD across sets (with >=3 distinct targets) of each set's mean yes-rate; observed > null means people's overall willingness to say yes varies by set more than the model's single similarity-to-response mapping produces (set-specific rule breadth), observed < null means less. (observed 0.193 vs null mean 0.153, z=2.84, p=0.00799, q=0.0213) [survives FDR]
