Listeners interpret referring expressions by inverting a speaker who balances competitor confusion avoidance with visual contrast resolution against non-matching distractors. When selecting referring expressions, speakers penalize candidate words that are shared with visually similar competitor objects in the context, but also gain communicative utility when an expression actively resolves minimal visual contrast against neighboring objects that lack the named feature. Pragmatic listeners reason recursively at depth 2 about this contrast-sensitive, confusion-averse speaker alongside common knowledge of visual pop-out and distinctiveness.

Refining rsa_l2_salience_confusion_l0 by incorporating contrast-maximizing referring expression utility from contrastive_speaker:
Speakers evaluate candidate words not only by their informativeness to simulated listeners and the penalty for overlapping with confusable competitors, but also by adding a communicative utility bonus when an expression differentiates the intended referent from minimally different visual alternatives in the display that lack the named property. This improves predictions on displays where a target contrasts with a minimal foil, capturing the communicative value of contrastive focus that pure distractor confusion overlooks.

Differences from rsa_l2_salience_confusion_l0:
- Recursion depth: unchanged at depth 2 (choice_probs calls L2).
- Parameters added: w_contrast (Normal(0.0, 1.0)), weighting the communicative utility bonus for expressions that establish visual contrast against context objects lacking the named feature.
- Parameters removed: none.
- Other terms: added compute_contrast to calculate pairwise visual feature contrast between referents and competitors lacking each utterance; incorporated + w_contrast * at(contrast, u, r) into the simulated speaker utility at both depth-1 (S1 in L1) and depth-2 (S2 in L2).
