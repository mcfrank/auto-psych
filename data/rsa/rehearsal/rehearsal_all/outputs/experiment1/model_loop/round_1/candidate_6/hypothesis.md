Refining competitor_confusion_speaker by incorporating discrete perceptual singleton salience from oddity_heuristic_listener into the shared referent prior. When evaluating candidate referents, speakers and listeners not only avoid expressions shared with visually confusable competitors, but also share a perceptual expectation that unique objects lacking identical duplicates in the display are more salient and more likely to be referred to a priori. This shared singleton preference biases spontaneous object choices on uninformative trials and guides pragmatic listener interpretation when resolving referring expressions.

Differences from competitor_confusion_speaker:
- Recursion depth: unchanged, choice_probs calls L1.
- Parameters added: w_singleton (prior Normal(0, 1)).
- Parameters removed: none.
- Terms changed: compute_confusion_and_prior computes an is_singleton indicator and adds w_singleton * is_singleton to the object prior logits; L1 receives the object prior as an additional parameter and sets speaker: given(r in OBJ, wpp=vec(prior, r)) instead of wpp=1; choice_probs passes prior to L1.