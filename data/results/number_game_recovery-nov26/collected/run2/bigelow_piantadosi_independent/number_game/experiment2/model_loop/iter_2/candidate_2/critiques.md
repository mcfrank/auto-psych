# Critique of `capacity_limited_evidence_integration`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **rule_violating_near_target_yes_rate** — Among sets of >=2 examples that all fit some small mathematical rule (<=20 members), the yes rate for targets NOT in that rule but within 5 of some example; observed above null means people let numerical proximity override the rule more than the model does (similarity leak), below means people respect the rule more strictly than the model. (observed 0.301 vs null mean 0.255, z=2.45, p=0.03, q=0.12)
- **cell_response_agreement** — Mean within-cell pairwise agreement (fraction of response pairs that match) over (number_set, target) cells answered >=2 times, weighted by pairs; observed above null means people's answers to the same item are more deterministic/consistent than the model's middling p_yes predicts (model under-sharp), below means people are noisier. (observed 0.707 vs null mean 0.688, z=2.31, p=0.012, q=0.0959)
