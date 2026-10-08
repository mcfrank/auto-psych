We refine rsa_l1_salience by proposing that speakers balance communicative informativeness against utterance production costs, penalizing shared non-distinctive descriptors that refer to multiple objects in the visual scene. Pragmatic listeners invert this cost-sensitive speaker to disambiguate referents, while coordinating their inferences with contextual object salience driven by feature richness and familiarization base rates.

Differences from source rsa_l1_salience:
- Model refined: rsa_l1_salience
- Recursion depth: unchanged (choice_probs calls L1 at depth 1)
- Parameters added: cost with prior dist.Normal(0.0, 1.0)
- Parameters removed: none
- Other terms: In L1, the speaker's utility subtracts an utterance production cost for non-distinctive words, - cost * vec(utt_cost, u), where utt_cost is the extension size of non-sink words beyond a single referent.
