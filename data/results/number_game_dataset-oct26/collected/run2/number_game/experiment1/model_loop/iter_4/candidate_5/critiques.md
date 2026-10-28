# Critique of `coherence_sharpened_decision_exemplar`

2 of 8 evaluated test statistics show a significant discrepancy (p ≤ 0.05), over 1000 posterior-predictive replicates.

## Significant discrepancies (a better model should address these)

Raw two-sided p shown with a Benjamini-Hochberg FDR-adjusted q across this round's statistics. Prioritise discrepancies that survive the FDR (`q ≤ alpha`); a raw-only hit may be one of several screened at once.
- **common_factor_rule_gap** — Among multi-example sets whose examples all share a greatest common divisor g>=3, yes rate for targets divisible by g minus yes rate for targets not divisible by g; observed > model means people generalize the shared "multiples of g" rule more sharply than graded-factor coherence-tuned similarity predicts (tracks whether the iter-3 refinement closed this gap). (observed 0.197 vs null mean 0.0819, z=2.65, p=0.00999, q=0.0799)
- **trial_index_drift** — Pearson correlation between trial_index and the yes response; observed > model (model ~0, being stationary) means yes responses become more frequent later in the session. (observed 0.0417 vs null mean 0.005, z=1.98, p=0.046, q=0.184)
