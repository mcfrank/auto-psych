Pragmatic listeners reason at depth 2 about speakers who avoid competitor confusion, penalize unmentioned informative features of the target, penalize broadly shared descriptors with utterance extension costs, and account for visual singleton salience. When multiple descriptions apply to an object, speakers prefer specific, less widely shared words over descriptors that extend to many scene objects, while avoiding ambiguous words when the target possesses superior unstated descriptors. On uninformative trials, listeners spontaneously favor unique visual singletons over duplicated items in the display.

Refining rsa_l2_singleton_confusion_omission by incorporating utterance extension costs from rsa_l2_singleton_confusion_cost:
Speakers evaluate candidate referring expressions not only by informativeness to simulated listeners, distractor confusion avoidance, and omitted-alternative exhaustification penalties, but also by penalizing words proportional to their visual extension (the fraction of display objects satisfying the feature). When an object possesses multiple valid descriptions, speakers avoid broadly applicable words in favor of more specific descriptors, capturing listeners' preference for specific descriptions on displays where features differ in contextual generality while preserving sensitivity to unmentioned target features.

Differences from rsa_l2_singleton_confusion_omission:
- Recursion depth: Unchanged at depth 2 (choice_probs calls L2).
- Parameters added: w_cost (Normal(0.0, 1.0), penalizing candidate words proportional to their visual extension / scene frequency).
- Parameters removed: None.
- Other terms: Added compute_extension_cost to calculate the scene-wide object fraction sharing each word; incorporated - w_cost * vec(cost, u) into simulated speaker utility at both depth-1 (S1 in L1) and depth-2 (S2 in L2).
