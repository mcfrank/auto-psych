We refine feature_redundancy_discounted_l2 by incorporating an aspiration threshold for communicative informativeness from satisficing_speaker_rsa into the speaker's utility function. Rather than optimizing informativeness linearly to perfection, boundedly rational speakers satisfice: once an expression achieves a sufficient threshold of informativeness, further uniqueness provides diminishing marginal returns, allowing speakers to produce shared or redundant words even when describing complex competitor objects. Depth-2 listeners invert this satisficing, redundancy-discounted speaker, naturally maintaining substantial credence for overspecified competitor objects across scalar reference games and resolving the source model's systematic underprediction of competitor choices.

Differences from feature_redundancy_discounted_l2:
- Recursion depth: Unchanged; choice_probs calls L2.
- Parameters added: theta (prior: dist.Beta(2.0, 2.0)).
- Parameters removed: None.
- Speaker utility: Replaces linear communicative informativeness log(L + EPS) with bounded satisficing utility log((L / (L + theta)) + EPS) in speaker utterance selection at depth 1 (L1) and depth 2 (L2).
- All other terms: Contextual feature redundancy discounting on speaker alternatives (w_redundancy), pairwise Jaccard overlap calculation, visual referent prior (w_singleton, w_distinct, w_features, w_color), unbiased literal semantics at L0, depth-2 RSA recursion (L1, L2), and uniform lapse mixing remain identical to the source model.
