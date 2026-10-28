# Critique of `erlang_interval_size_prior`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **span_boundary_gap** — For sets of 2+ examples spanning at least 10, yes rate for non-example targets strictly inside the range minus targets 1-10 outside it; the interval mechanism makes the range edge a sharp boundary, so observed below null_mean means people's generalization crosses the span edge more smoothly (similarity-like) than the model, above means a sharper edge. (observed 0.145 vs null mean 0.0676, z=1.94, p=0.042, q=0.336)
