Pragmatic listeners reason at depth 2 about speakers who both avoid expressions creating competitor confusion and penalize expressions that omit informative alternative features of the target referent, guided by visual singleton salience. When multiple descriptions apply, speakers avoid words that overlap with visually similar competitors and penalize words leaving distinctive alternative features unsaid. On uninformative trials, listeners spontaneously favor unique visual singletons over duplicated items in the display, defaulting to uniform choice when all items are distinct.

Refining rsa_l2_confusion_exhaustification_l0 by simplifying the object prior to discrete visual singleton salience:
We simplify the common-ground object prior from multi-attribute continuous salience to discrete visual singleton salience, removing feature distinctiveness, feature complexity, valence, and color biases that distort uninformative prior expectations. This aligns the uninformative prior directly with empirical behavior while retaining depth-2 recursive reasoning, competitor confusion avoidance, and omitted-alternative exhaustification in the speaker utility.

Differences from rsa_l2_confusion_exhaustification_l0:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L2).
- Parameters added: None.
- Parameters removed: w_distinct, w_features, w_valence, w_color.
- Other terms: In choice_probs, the object prior is simplified from the multi-attribute salience combination to pure discrete visual singleton salience (softmax_prior(w_singleton * is_singleton)), eliminating the continuous distinctiveness, feature count, valence, and grayscale terms.
