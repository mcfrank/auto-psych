The population of listeners is cognitively heterogeneous, consisting of a mixture of literal and pragmatic reasoning types who share a visual salience prior that incorporates perceptual color distinctiveness alongside feature count and familiarization. When candidate referents are visually grounded, chromatic coloration heightens an object's visual pop-out and prior communicative salience over grayscale competitors for both literal and pragmatic listeners. This fits better because interlocutors share visual attention to salient chromatic cues before word interpretation, aligning prior expectations on colored displays without disrupting scalar implicature reasoning across the mixture of reasoning types.

Refinement of literal_pragmatic_mixture_listener:
- Base model refined: literal_pragmatic_mixture_listener
- Recursion depth: Depth 1 (choice_probs calls L0 and L1, unchanged from literal_pragmatic_mixture_listener).
- Parameters added: w_color (prior: Normal(0.0, 1.0)).
- Parameters removed: None (retains alpha, gamma, p_pragmatic, w_features, w_familiar, and lapse).
- Other terms: choice_probs incorporates perceptual color distinctiveness (w_color * (1.0 - ctx.grayscale)) into the common-knowledge object salience prior before the softmax (component taken from rsa_l2_color_salience); the literal listener L0, pragmatic listener L1, the mixture proportion p_pragmatic, and uninformative prior choices are unchanged.
