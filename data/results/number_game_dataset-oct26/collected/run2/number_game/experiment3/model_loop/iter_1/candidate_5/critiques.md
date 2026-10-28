# Critique of `enumerated_list_concept_exemplar`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **session_midsession_plateau** — Yes-rate on trials 100-274 minus the average of the yes-rates on trials 0-99 and 275-449; observed > null means the relaxation of the yes criterion happens early then plateaus (concave), which the model linear-in-trial criterion drift cannot produce. (observed 0.0323 vs null mean -0.00272, z=3.12, p=0.004, q=0.024) [survives FDR]
- **log_magnitude_region_gap** — Among out-of-span targets with gap 5-40 from the span, yes-rate when the nearest bounding example is <=20 minus when it is >=60; observed < null means equal linear gaps hurt more among small numbers (log-like compression), which the model under-produces. (observed -0.0584 vs null mean 0.028, z=-2.96, p=0.00599, q=0.024) [survives FDR]
- **below_vs_above_span_asymmetry** — Yes-rate for targets 5-40 below the smallest example minus targets 5-40 above the largest example; observed < null means people generalize less downward than upward (log-like magnitude: equal linear gaps below are larger ratios), which the symmetric linear-distance model fails to produce. (observed -0.0283 vs null mean 0.0263, z=-2.45, p=0.014, q=0.0373) [survives FDR]
