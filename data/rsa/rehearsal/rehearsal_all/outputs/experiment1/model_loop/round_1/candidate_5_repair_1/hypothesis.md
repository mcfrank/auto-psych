Pragmatic listeners reason at depth two about a speaker who actively avoids referring expressions shared with visually confusable competitors. People expect speakers to avoid ambiguous labels that apply to alternative referents with high visual feature similarity, choosing more distinctive descriptions instead. Pragmatic listeners invert this confusion-averse speaker while integrating a shared perceptual prior favoring visual singletons, distinctive objects, feature complexity under positive framing, and high color contrast.

Refining rsa_l2_singleton_feat_color_valence_l0 by incorporating competitor-confusion aversion from competitor_confusion_speaker into the speaker utility. This should fit better by explaining participants' reluctance to choose confusable competitors in complex visual contexts (such as the Mayn & Demberg displays where confusion models excel) while maintaining depth-two recursive reasoning and rich perceptual salience priors.

Differences from source (rsa_l2_singleton_feat_color_valence_l0):
- Recursion depth: Unchanged; choice_probs calls depth-2 listener L2.
- Parameters added: w_confusion ~ Normal(0.0, 2.0).
- Parameters removed: None.
- Terms changed: S1 and S2 speaker choice utilities subtract w_confusion * at(confusion, u, r), where confusion is the matrix product of the non-sink lexicon and pairwise exponential-distance visual similarity between objects. The prior computation, literal L0 semantics, and lapse process remain identical to the source.
