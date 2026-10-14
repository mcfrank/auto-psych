Pragmatic listeners reason about an attentive speaker whose intended message selection balances literal informativeness against visual attention, but whose actual feature articulation is subject to stochastic execution noise over applicable descriptors. Refining attentive_display_listener by incorporating the stochastic transmission noise component from noisy_channel_speaker_l1: communicative speakers anticipate an attentive literal listener whose gaze favors visually salient and feature-rich objects, but occasionally slip when articulating true features. This transmission noise cushions extreme likelihood ratios, directly resolving the incumbent model's severe overprediction of minimal-feature targets when competitors have duplicate copies, large feature gaps, or broad ambiguity.

Differences from source (attentive_display_listener):
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from source).
- Parameters added: epsilon (prior: dist.Beta(1.0, 5.0)), representing the probability of stochastic execution noise over applicable features.
- Parameters removed: none.
- Other terms: In the simulated speaker at level L1, the speaker chooses an intended communicative feature m according to literal informativeness over the attentive literal listener L0, but articulates an applicable feature u with execution noise epsilon.
