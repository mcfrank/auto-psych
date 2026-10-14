Pragmatic listeners reason at depth 1 about a noisy-channel speaker who communicates with an attentive literal listener whose visual attention is drawn to distinctive and feature-rich objects. Refining noisy_channel_speaker_l1 by incorporating the attentive literal listener component from attentive_display_listener, literal interpretation is biased toward visually prominent referents rather than uniform across truth-conditional matches. Because the speaker anticipates an attentive listener, they are more willing to use shared descriptors for prominent referents, while stochastic execution noise over descriptors cushions extreme likelihood ratios to prevent excessive elimination of competitor objects when competitors have duplicates or large feature gaps.

Differences from source (noisy_channel_speaker_l1):
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from source).
- Parameters added: None.
- Parameters removed: None.
- Other terms: In the literal listener L0, referent choice weights are modulated by the referent prior (attention) via `wpp=at(lex, u, r) * vec(prior, r)` rather than uniform truth-conditional weighting `at(lex, u, r)`. In the simulated speaker at level L1, the speaker evaluates intended messages using this attentive literal listener `L0[m, r](lex, prior)`.
