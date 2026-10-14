We refine empty_baseline_salience_l2 by generalizing the baseline ground anchor from an absolute zero-feature object to the contextual minimal-complexity object in the visual display. In any scene containing objects with varying feature counts, the simplest object acts as a distinct perceptual ground and baseline reference anchor against which more complex objects are contrasted, attracting baseline attention regardless of whether it has zero or positive features. This contextual baseline salience elevates the prior probability of minimal objects relative to intermediate-feature items across all multi-level displays, directly resolving the severe overprediction of intermediate feature-count objects on uninformative trials.

Differences from empty_baseline_salience_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: w_baseline (prior: dist.Normal(0.0, 1.0)).
- Parameters removed: w_empty.
- Object prior: Computes contextual minimal-complexity baseline salience (w_baseline * is_baseline, where is_baseline indicates having the minimum feature count in the display when feature counts vary) instead of requiring strictly zero features (is_empty).
- All other terms: Discrete singleton salience, contextual distinctiveness, visual feature complexity, visual color contrast, unbiased literal L0 semantics, depth-2 RSA recursion (L1, L2), and uniform lapse mixing remain identical to the source model.
