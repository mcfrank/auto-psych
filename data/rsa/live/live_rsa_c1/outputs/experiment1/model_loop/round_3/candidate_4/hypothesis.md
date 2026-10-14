Pragmatic listeners reason at depth 1 about an attentive literal listener, but account for stochastic execution noise in the speaker's feature articulation. Refining attentive_display_listener by incorporating the stochastic transmission noise component from noisy_channel_speaker_l1: foundational literal interpretation remains biased toward visually salient and distinctive objects, and communicative speakers anticipate this attentive gaze, but speakers occasionally articulate an alternative applicable descriptor by mistake. This transmission noise cushions extreme likelihood ratios, directly correcting the incumbent model's severe overprediction of minimal-feature targets when competitors have duplicate copies or large feature gaps, while preserving visual attention biases on uninformative mumble trials.

Differences from source (attentive_display_listener):
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from source).
- Parameters added: epsilon (prior: dist.Beta(1.0, 5.0)), representing the probability of stochastic execution noise over true features.
- Parameters removed: none.
- Other terms: In the simulated speaker at level L1, the speaker chooses an intended communicative feature m according to informativeness over attentive L0, but articulates an applicable feature u with stochastic execution noise epsilon.
