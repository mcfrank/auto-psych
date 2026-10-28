# Critique of `outlier_tolerant_rule_concepts`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **in_range_nonrule_same_digit_sets** — For 3-4 example sets whose examples all share one last digit, yes-rate for targets strictly inside the [min,max] range that do NOT share that last digit; observed > null means people still accept in-range numbers (interval-like generalization) when a sharp digit rule fits, more than the model's rule-dominated posterior allows (observed < null: humans stricter). (observed 0.0796 vs null mean 0.141, z=-3.72, p=0.002, q=0.016) [survives FDR]
- **shared_last_digit_far_target** — For 2-4 example sets whose examples all share one last digit, yes-rate for targets with that same last digit lying more than 10 outside the set's range; observed < null means people extend a last-digit rule to far targets less than the model's rule posterior predicts (observed > null: more). (observed 0.708 vs null mean 0.853, z=-3.29, p=0.00599, q=0.024) [survives FDR]
- **shared_tens_digit_outside** — Among targets outside the set's [min,max] range and not examples, yes-rate for targets sharing a tens digit (same decade, 100 counted as 10) with some example minus targets sharing none, restricted to targets within 10 of the nearest example; observed > null means people generalize by decade membership more than the model's decade-aligned interval bonus captures. (observed 0.113 vs null mean 0.0375, z=2.57, p=0.00999, q=0.0266) [survives FDR]
- **unanimous_no_cell_fraction** — Fraction of distinct (set, target) cells in which every response is "no"; observed > null means people reject clearly-out-of-concept targets more uniformly than the model's guessing/stray-example noise allows (observed < null: humans noisier than the model). (observed 0.23 vs null mean 0.199, z=2.31, p=0.028, q=0.0559)
