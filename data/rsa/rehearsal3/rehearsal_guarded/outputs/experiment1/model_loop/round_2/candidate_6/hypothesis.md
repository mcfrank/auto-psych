Listeners interpret referring expressions by modeling a trembling-hand speaker who occasionally produces unintended utterances due to speech production noise, while actively avoiding expressions that are shared with visually confusable competitors. When a heard word is ambiguous, the rational utility of speaking it intentionally is diluted, increasing the posterior probability that the utterance was an unintended speech slip referring to a non-matching foil, while competitor confusion aversion penalizes words shared with visually similar distractors. On uninformative prior trials, choices are governed directly by perceptual singleton pop-out, distinctiveness, and visual feature complexity, with background motor lapses capturing random clicking.

Model refined: `trembling_hand_speaker_2`.
Differences from source:
- Recursion depth: Unchanged (depth 2, calling L2 in choice_probs).
- Parameters added: `w_confusion` (prior Normal(0.0, 2.0)) weighting competitor confusion aversion in simulated speaker utilities.
- Parameters removed: None.
- Speaker utility: Extended at both simulated speaker tiers (S1 inside L1, and S2) to penalize expressions shared with visually confusable competitors in the visual scene (`- w_confusion * at(confusion, u, r)`), where confusion is proportional to the feature similarity of alternative referents satisfying each word (`real_lex @ sim`).
- Referent prior: Unchanged (combining discrete singleton salience, continuous distinctiveness, and visual feature complexity).
- Speech noise and lapse: Unchanged (trembling-hand production error mixed into S2 policy, background motor lapse mixed into final choices).
