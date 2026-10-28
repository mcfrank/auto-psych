# Critique of `serial_position_ends_inflated_size`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **decade_boundary_effect** — Among non-example targets 1-5 away from their nearest example, yes rate when the target shares that example's tens decade (same value of (x-1)//10) minus when it crosses a decade boundary; observed above null_mean means people treat decades as categories beyond what the model's uniform intervals and rules predict, below means less. (observed 0.088 vs null mean 0.0346, z=2.29, p=0.026, q=0.208)
