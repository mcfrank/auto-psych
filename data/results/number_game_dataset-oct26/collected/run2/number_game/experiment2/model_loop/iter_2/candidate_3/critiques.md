# Critique of `drifting_criterion_dual_route_exemplar`

4 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **between_set_yes_rate_sd** — Standard deviation across number sets (with >=18 responses) of each set's overall yes-rate; observed above the null means sets differ in how generously they are generalized more than the model predicts (model's set-level breadth is too compressed), below means the model exaggerates set-to-set differences. (observed 0.18 vs null mean 0.16, z=2.38, p=0.022, q=0.0639)
- **upward_vs_downward_generalization** — Over all sets, among targets 1-20 above the largest example vs 1-20 below the smallest example, yes-rate above minus below; observed above the null means people generalize upward (to larger numbers) more than downward, an asymmetry the model's symmetric size-distance similarity cannot produce (below means a downward bias). (observed 0.0869 vs null mean 0.0217, z=2.33, p=0.024, q=0.0639)
- **recency_last_vs_first_example** — In 2+-example sets, for non-example targets strictly closer in size to the last-shown example than the first (or vice versa), yes-rate for closer-to-last minus closer-to-first; observed above the null means people weight the most recent example more than the model's fitted recency (below means primacy / less recency than modelled). (observed 0.0301 vs null mean -0.00596, z=2.33, p=0.022, q=0.0639)
- **drift_step_curvature** — (yes-rate in trials 150-299 minus trials 0-149) minus (yes-rate in trials 300+ minus trials 150-299); positive observed above the null means the session drift toward "yes" is a front-loaded step/saturating jump rather than the steady linear relaxation the model assumes (below means the drift accelerates late). (observed 0.0681 vs null mean 0.0103, z=2.16, p=0.034, q=0.0679)
