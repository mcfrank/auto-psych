# Critique of `saturating_criterion_relaxation`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **set_level_yes_rate_dispersion** — Standard deviation across example sets (with 40+ rows) of each set's overall yes-rate; observed > null means people differentiate sets (some sets read as broad, others as narrow) more than the model does, observed < null means the model over-differentiates. (observed 0.147 vs null mean 0.123, z=3.35, p=0.00599, q=0.048) [survives FDR]
- **ratio_vs_magnitude_fixed_gap** — Among out-of-span targets at a linear gap of 6-20 from the nearest span edge, yes-rate when |log(target/edge)| < 0.25 minus when > 0.5 (at fixed linear gap this contrasts large vs small magnitudes); observed > null means people judge size by ratio (log-compressed), which the model's linear size distance under-produces. (observed 0.0689 vs null mean -0.00925, z=2.23, p=0.02, q=0.0799)
