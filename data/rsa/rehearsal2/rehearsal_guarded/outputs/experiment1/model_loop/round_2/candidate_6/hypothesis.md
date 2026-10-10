Pragmatic listeners reason at depth 2 about speakers who avoid competitor confusion while accounting for visual singleton salience. When multiple objects satisfy a description, speakers avoid referring to a target if its competitors share many visual features, leading listeners to expect words to target objects whose competitors are visually distinct. On uninformative trials, listeners spontaneously favor unique visual singletons over duplicated items in the display.

Refining rsa_l2_singleton by incorporating competitor confusion avoidance from competitor_confusion_speaker:
Speakers evaluate candidate words not only by their informativeness to simulated listeners, but also by penalizing words that overlap with visually similar competitor objects in the context. This improves fit on displays where distractors share visual features with the intended referent, guiding listeners away from targets whose competitors create visual confusion.

Differences from rsa_l2_singleton:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L2).
- Parameters added: w_confusion (Normal(0.0, 1.0), penalizing candidate words proportional to their visual similarity overlap with alternative referents satisfying the word).
- Parameters removed: None.
- Other terms: Added compute_confusion to calculate pairwise visual similarity between referents and competitors sharing each utterance; incorporated - w_confusion * at(confusion, u, r) into the simulated speaker utility at both depth-1 (S1 in L1) and depth-2 (S2 in L2).
