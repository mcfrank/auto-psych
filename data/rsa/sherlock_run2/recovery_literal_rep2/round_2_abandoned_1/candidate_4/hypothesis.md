Listeners interpret referring expressions through direct feature contrast, evaluating matching referents by penalizing unmentioned features against a neutral object prior that tracks familiarity base rates without an intrinsic feature-count bias. When no communicative utterance is given, people exhibit no baseline preference for feature-rich objects, so communicative reference resolution operates against an unskewed object distribution rather than competing with an opposing feature salience prior.

Differences from feature_contrast_listener:
- Recursion depth: Depth 0 (calls L_contrast, unchanged).
- Parameters added: None.
- Parameters removed: w_features.
- Other terms: In choice_probs, the object prior is computed purely from familiarization base rates (softmax_prior(params["w_familiar"] * ctx.familiarization)) rather than including a feature-count term (w_features * ctx.feature_count).
