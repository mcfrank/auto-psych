# Critique of `span_filling_size_exemplar`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **recency_last_vs_first_example** — In 2+-example sets, for non-example targets strictly closer in size to the last-shown example than to the first (vs the reverse), yes-rate closer-to-last minus closer-to-first; observed above the null means people's size generalization favours the most recent example, which the span-based size mismatch (identical for every example) cannot produce (below means primacy). (observed 0.0301 vs null mean -0.0194, z=3.32, p=0.002, q=0.016) [survives FDR]
- **upward_vs_downward_outside_span** — Over all sets, yes-rate for targets 1-20 above the largest example minus targets 1-20 below the smallest example; observed above the null means people generalize upward (to larger numbers) more than the model's symmetric outside-span size mismatch predicts (below means a stronger downward bias than modelled). (observed 0.0869 vs null mean 0.00602, z=2.84, p=0.014, q=0.0559)
- **drift_step_curvature** — (yes-rate in trials 150-299 minus trials 0-149) minus (yes-rate in trials 300+ minus trials 150-299); observed above the null means the session drift toward "yes" is a front-loaded / saturating jump rather than the linear criterion relaxation the model assumes (below means the drift accelerates late). (observed 0.0681 vs null mean 0.0125, z=2.01, p=0.038, q=0.101)
