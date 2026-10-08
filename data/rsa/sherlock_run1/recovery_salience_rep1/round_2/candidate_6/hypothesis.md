Listeners reason at depth 2 with a feature-simplicity prior, but rather than simply probability matching their posterior beliefs when clicking an object, they apply softmax decision rationality. This allows listeners who infer communicative intentions through recursive mentalizing and complexity priors to decisively select the referent with the highest posterior support.

Model refinement details:
- Base model refined: feature_salience_l2
- Recursion depth: Depth 2 (choice_probs calls L2, unchanged from feature_salience_l2)
- Parameters added: beta (LogNormal(0.0, 1.0)), representing listener decision rationality
- Parameters removed: None (alpha, w_feature, and lapse retained unchanged)
- Functional terms changed: In L2, listener choice weights are modified from probability matching the posterior to softmax decision rationality, exponentiating posterior probabilities by beta before normalization.
