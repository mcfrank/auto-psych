This model refines fewest_features_listener by extending the feature economy heuristic to prior expectations over referents: listeners possess an intrinsic cognitive preference for minimal, less complex objects that operates both when interpreting utterances and when forming a priori expectations. Rather than reverting to a uniform distribution when no informative description is given, listeners expect speakers to favor simpler objects with fewer total features a priori according to the same feature economy penalty. This improves model fit because human referent selection consistently reflects a general simplicity bias across both communicative and uninformative contexts.

Differences from fewest_features_listener:
- Refined model: fewest_features_listener
- Recursion depth: depth 0 (choice_probs calls L_heuristic, unchanged).
- Parameters added: none.
- Parameters removed: none.
- Prior calculation in choice_probs: on prior trials (ctx.is_prior > 0), the listener chooses according to a feature economy simplicity prior softmax_prior(-params["beta"] * ctx.feature_count) rather than an uninformative uniform distribution.
- Heuristic listener L_heuristic: unchanged.
- All other memo agents, distributions, and terms are unchanged.
