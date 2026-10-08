We refine bounded_alternatives_listener by proposing that speakers balance communicative informativeness against utterance production costs, penalizing ambiguous words that refer to multiple objects in the visual display. When a resource-limited listener simulates the speaker's referential choice, the speaker's utility discounts unmentioned alternative expressions due to bounded attention while simultaneously disfavoring ambiguous shared descriptors. On prior trials with no informative word, listener choices are guided solely by contextual object salience.

Differences from source (bounded_alternatives_listener):
- Recursion depth: Unchanged (choice_probs calls L1 at depth 1).
- Parameters added: cost (prior: Normal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In L1, the speaker's choice weight incorporates an utterance production cost penalty `- cost * vec(utt_cost, u)`, where `utt_cost` measures the excess extension of non-sink utterances across referents (`jnp.maximum(0.0, jnp.sum(ctx.lex * (1.0 - ctx.is_sink)[:, None], axis=1) - 1.0)`).
