# Critique of `common_divisor_sharpened_exemplar`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **background_set_rate_sd** — Standard deviation across sets (with >=9 'background' trials: targets sharing no example's final digit and >=10 from every example) of each set's background yes-rate; observed > null means sets differ in how broadly people generalize to dissimilar numbers (set-specific rule breadth, e.g. "4,66,78,96" vs "80,70,90,100") more than the model's similarity features produce, observed < null means less. (observed 0.205 vs null mean 0.147, z=4.55, p=0.002, q=0.00799) [survives FDR]
- **single_example_shared_factor_contrast** — In single-example sets, yes-rate for targets sharing a common factor >=3 with the example (gcd(target, example) >= 3, e.g. 14/49/98 for "7", 30/100 for "5") minus yes-rate for other targets; observed > null means people infer a 'multiples/factors of the example' rule from one example, which the model (no factor term for single examples) under-produces, observed < null means less. (observed -0.0966 vs null mean -0.355, z=3.30, p=0.002, q=0.00799) [survives FDR]
