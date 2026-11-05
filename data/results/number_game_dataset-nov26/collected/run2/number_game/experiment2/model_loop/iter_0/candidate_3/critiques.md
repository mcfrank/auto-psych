# Critique of `final_digit_overshadowing_overlap`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **last_vs_first_example_digit** — In multi-example sets whose first and last shown examples have different final digits, yes-rate for targets sharing the LAST example's final digit minus yes-rate for targets sharing the FIRST example's final digit; nonzero observed vs null means presentation order (recency/primacy) matters, which the order-blind model cannot produce (positive = recency). (observed 0.0985 vs null mean 0.0299, z=1.90, p=0.05, q=0.4)
