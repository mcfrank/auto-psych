Visual singleton salience requires the presence of visual features that can pop out pre-attentively; an unadorned referent lacking visual features does not capture attention merely by virtue of having no duplicates. Pragmatic listeners reason at depth 2 about a noisy-channel speaker under common knowledge of a referent prior where visual singleton salience specifically favors feature-bearing unique objects (featured singletons) rather than empty blanks. This feature-contingent salience prevents listeners from over-selecting empty singletons when competing against duplicated full objects on uninformative trials, while transmission noise cushions likelihoods on word trials.

Refining noisy_channel_speaker_l2:
- Recursion depth: Depth 2 (choice_probs evaluates L2, unchanged from noisy_channel_speaker_l2).
- Parameters added: None.
- Parameters removed: None.
- Other terms: In the referent prior, visual singleton salience is modulated by feature presence (`featured_singleton = is_singleton * (ctx.feature_count > 0)`), ensuring that only objects possessing visual features receive singleton pop-out weighting.
