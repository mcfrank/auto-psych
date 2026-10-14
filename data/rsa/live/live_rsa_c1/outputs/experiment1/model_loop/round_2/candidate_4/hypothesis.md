Pragmatic listeners reason at depth 1 about a speaker whose intended message selection is rational but whose actual feature articulation is subject to stochastic execution noise over applicable descriptors. Refining rsa_l1_singleton_feat_color_familiar by incorporating the stochastic transmission noise component from noisy_channel_speaker_l2: listeners perform first-order pragmatic inversion of a speaker who balances literal informativeness against a rich perceptual and experiential referent prior, but occasionally articulates any applicable descriptor by mistake. This transmission noise cushions likelihood ratios, directly mitigating the model's overprediction of minimal-feature targets when competitors have duplicate copies or large feature gaps, while preserving depth-1 calibrated reasoning and perceptual prior expectations on uninformative mumble trials.

Differences from source (rsa_l1_singleton_feat_color_familiar):
- Recursion depth: depth 1 (choice_probs calls L1, unchanged from source).
- Parameters added: epsilon (prior: dist.Beta(1.0, 5.0)), representing the probability of stochastic execution noise over true features.
- Parameters removed: none.
- Other terms: In the simulated speaker at level L1, the speaker chooses an intended communicative feature m according to literal informativeness but articulates an applicable feature u with execution noise epsilon.
