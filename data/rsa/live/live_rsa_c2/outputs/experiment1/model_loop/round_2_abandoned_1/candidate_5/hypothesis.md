Refining feature_redundancy_discounted_l2, we incorporate an aspiration threshold from satisficing_speaker_rsa into the speaker's communicative utility: speakers are boundedly rational satisficers who evaluate informativeness against a satisficing threshold, experiencing diminishing marginal returns for additional uniqueness once an expression is sufficiently clear. This mechanism prevents the artificial collapse of competitor probability in scalar reference games, allowing speakers to produce shared referring expressions for overspecified competitors even when distinctive alternative features exist. Pragmatic listeners invert this satisficing, redundancy-discounted speaker at depth 2 while maintaining common knowledge of visual object salience.

Differences from feature_redundancy_discounted_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: theta (prior: dist.Beta(2.0, 2.0)).
- Parameters removed: None.
- Speaker utility: Replaces standard linear log-informativeness exp(alpha * log(L + EPS)) with satisficing utility exp(alpha * log((L / (L + theta)) + EPS)) at both speaker levels (L1 and L2).
- All other terms: Utterance redundancy discounting on speaker alternatives (w_redundancy via pairwise Jaccard overlap), referent prior components (w_singleton, w_distinct, w_features, w_color), literal L0 semantics, and uniform lapse mixing remain identical to the source model.
