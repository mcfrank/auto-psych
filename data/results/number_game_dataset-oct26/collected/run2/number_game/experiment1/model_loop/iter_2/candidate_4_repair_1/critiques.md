# Critique of `coherence_tuned_attention_exemplar`

3 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **between_set_yes_rate_sd** — Standard deviation across number sets of each set's mean yes rate; observed > model means people treat some sets as far more (or less) generous than the model, i.e. the model under-produces set-level extremity (near-all-yes / near-all-no sets). (observed 0.229 vs null mean 0.195, z=2.59, p=0.012, q=0.0693)
- **common_factor_rule_gap** — Among multi-example sets whose examples all share a greatest common divisor g>=3, yes rate for targets divisible by g minus yes rate for targets not divisible by g; observed > model means people generalize the shared "multiples of g" rule more sharply than coherence-tuned pairwise factor similarity predicts. (observed 0.197 vs null mean 0.0915, z=2.33, p=0.022, q=0.0693)
- **trial_index_drift** — Pearson correlation between trial_index and the yes response; observed > model (model ~0, being stationary) means yes responses become more frequent later in the session. (observed 0.0417 vs null mean 0.00191, z=2.19, p=0.026, q=0.0693)
