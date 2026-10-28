# Critique of `leading_digit_string_similarity`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **set_level_yes_rate_dispersion** — Standard deviation across example sets with 40+ rows of each set's overall yes-rate; observed > null means people treat some sets as broad and others as narrow more strongly than the model does (model compresses set-level differences), observed < null the reverse. (observed 0.147 vs null mean 0.128, z=2.53, p=0.012, q=0.0959)
- **even_vs_odd_set_parity_asymmetry** — (Yes-rate on even minus odd non-example targets in all-even sets of 2+ examples) minus (yes-rate on odd minus even non-example targets in all-odd sets of 2+ examples); observed > null means people abstract an "even numbers" rule more than an "odd numbers" rule, an asymmetry the model's symmetric parity similarity cannot produce. (observed 0.139 vs null mean 0.0598, z=2.09, p=0.032, q=0.107)
- **log_magnitude_region_gap** — Among out-of-span targets 5-40 (linear) from the nearest span edge, yes-rate when that nearest edge is <=20 minus when it is >=60; observed < null means equal linear gaps hurt more among small numbers (log-like magnitude compression), which the model's linear size distance under-produces. (observed -0.0584 vs null mean 0.00106, z=-2.06, p=0.04, q=0.107)
