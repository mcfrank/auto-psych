Listeners interpret referring expressions by combining an expectation that speakers avoid ambiguous utterances with a common-knowledge object salience prior that incorporates perceptual color distinctiveness alongside diagnostic feature contrast. When candidate referents are visually grounded, distinctive chromatic coloration elevates an object's visual pop-out and prior communicative salience over grayscale competitors, complementing the salience of unique semantic features and the speaker's utterance-level ambiguity penalty. This fits better because interlocutors share visual attention to salient chromatic cues before word interpretation, aligning prior expectations on colored displays without disrupting scalar implicature reasoning.

Refinement of costly_speaker_contrast_prior:
- Base model refined: costly_speaker_contrast_prior
- Recursion depth: Depth 1 (choice_probs calls L1, unchanged from costly_speaker_contrast_prior).
- Parameters added: w_color (prior: Normal(0.0, 1.0)).
- Parameters removed: None (retains alpha, w_contrast, w_familiar, cost_ambiguity, and lapse).
- Other terms: choice_probs incorporates perceptual color distinctiveness (w_color * (1.0 - ctx.grayscale)) into the common-knowledge object salience prior before the softmax (component taken from rsa_l2_costly_color_salience); the diagnostic contrast computation and cost-sensitive speaker utility are unchanged.
