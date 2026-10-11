Listeners interpret referring expressions by modeling a trembling-hand speaker who occasionally produces unintended utterances due to speech production noise. When a heard word is ambiguous, the intentional utility of producing it is divided and low, increasing the listener's posterior inference that the utterance was an unintentional slip referring to a non-matching distractor, whereas unambiguous expressions provide strong intentional signal that sharply suppresses foil inferences. On uninformative prior trials, choices are governed by visual singleton pop-out, contextual distinctiveness, and feature complexity, with background motor lapses capturing random clicking.

Model refined: `r1_c4`.
Differences from source:
- Recursion depth: Unchanged (depth 2, evaluating L2 in choice_probs).
- Parameters added: `tremble` (prior Beta(1.0, 19.0)) capturing baseline speech production error rate.
- Parameters removed: None.
- Speaker policy and listener inversion: The simulated second-order speaker policy S2 is mixed with uniform speech production noise over real context words (`(1.0 - tremble) * s2 + tremble * tremble_dist`), and pragmatic listener L2 inverts the resulting trembling-hand joint distribution over referents and utterances.
- Referent prior: Unchanged (combining discrete singleton salience, contextual distinctiveness, feature count modulated by framing valence, and color contrast).
