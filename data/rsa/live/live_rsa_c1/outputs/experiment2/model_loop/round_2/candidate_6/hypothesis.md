Perceptual pop-out from uniqueness in a visual scene requires distinguishing features; an unadorned, empty object does not capture pre-attentive attention merely because it lacks identical copies. Pragmatic listeners invert a similarity-contrast speaker with stochastic transmission noise under a referent prior where visual singleton salience applies only to unique objects possessing at least one visual feature. This feature-gated singleton salience directly corrects the overprediction of empty singleton choices when competing against duplicated full objects on uninformative trials, while contrastive similarity reasoning and transmission noise cushion pragmatic inferences on utterance trials.

Refining contrastive_noisy_channel_speaker by incorporating feature-gated singleton salience:
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from source).
- Parameters added: none.
- Parameters removed: none.
- Other terms: In choice_probs, discrete visual singleton salience is gated by the presence of visual features (`featured_singleton = is_singleton * (ctx.feature_count > 0.0)`), replacing unadorned singleton indicator in the referent prior so that empty objects lacking visual features receive no singleton pop-out boost.
