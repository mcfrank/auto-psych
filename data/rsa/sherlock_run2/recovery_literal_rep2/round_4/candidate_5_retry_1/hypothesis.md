Listeners interpret referring expressions using a perceptual distinctiveness heuristic over matching referents, while evaluating visual displays under an unskewed neutral prior that reflects familiarity base rates without an intrinsic distinctiveness bias (taken from feature_contrast_neutral_prior). In communicative contexts, hearing an informative description directs attention to matching candidates and favors objects whose distinguishing features stand out contextually; on unprompted prior trials without an informative utterance, people exhibit no baseline preference against distinctive or feature-rich objects. Decoupling communicative distinctiveness evaluation from unprompted object preferences allows listeners to resolve reference sensitively without distorting prior expectations.

Differences from distinctiveness_heuristic_listener:
- Refined model: distinctiveness_heuristic_listener
- Recursion depth: Depth 0 (calls L_heuristic, unchanged).
- Parameters added: None.
- Parameters removed: None.
- Other terms: In choice_probs, the object prior on prior trials is computed purely from familiarization base rates (softmax_prior(params["w_familiar"] * ctx.familiarization)) rather than adding the distinctiveness heuristic term (params["beta"] * distinctiveness).
