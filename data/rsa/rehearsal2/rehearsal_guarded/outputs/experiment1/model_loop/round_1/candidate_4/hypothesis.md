Listeners interpret referring expressions by inverting a speaker who actively avoids words that create confusion with visually similar competitors in the display. When multiple objects satisfy a description, speakers avoid referring to a target if its competitors share many visual features, leading listeners to expect words to target objects whose competitors are visually distinct. This confusion-averse speaker reasoning operates alongside visual salience and feature contrast at depth-2 pragmatic recursion.

Refining rsa_l2_singleton_feat_color_valence_l0 by incorporating competitor confusion avoidance from competitor_confusion_speaker:
Speakers evaluate candidate words not only by their informativeness to simulated listeners, but also by penalizing words that overlap with visually similar competitor objects in the context. This improves fit on reference games where distractors share visual features with the intended referent, preventing severe confusion penalties that purely informative speakers overlook.

Differences from rsa_l2_singleton_feat_color_valence_l0:
- Recursion depth: unchanged at depth 2 (choice_probs calls L2).
- Parameters added: w_confusion (Normal(0.0, 1.0)), penalizing words proportional to their visual similarity overlap with alternative referents satisfying the word.
- Parameters removed: none.
- Other terms: added compute_confusion to calculate pairwise visual similarity between referents and competitors sharing each utterance; incorporated - w_confusion * at(confusion, u, r) into the speaker utility at both depth-1 (S1 in L1) and depth-2 (S2 in L2).
