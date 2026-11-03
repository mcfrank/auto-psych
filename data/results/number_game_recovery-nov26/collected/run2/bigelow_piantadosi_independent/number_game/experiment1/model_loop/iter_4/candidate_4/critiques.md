# Critique of `recency_weighted_even_odd_family_prior`

1 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **first_exposure_minus_repeat_yes_rate** — Per participant (trial_index restart), yes rate on the first trial on which each set appears minus the yes rate on that set's later trials; the model ignores exposure history, so t_observed far from null_mean means first encounters of a set are judged more (above) or less (below) generously than later ones. (observed -0.0382 vs null mean -0.00632, z=-2.03, p=0.042, q=0.336)
