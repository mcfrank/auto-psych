We refine listener_rationality_shared_prior by proposing that speakers trade off communicative informativeness against utterance production costs, penalizing ambiguous words that refer to multiple objects in the visual display. Pragmatic listeners invert this cost-sensitive speaker, reasoning that an utterance describing a shared feature is less likely to have been chosen if a less costly distinctive alternative was available, while making boundedly rational softmax choices over candidate referents. When no informative word is heard, listeners choose according to contextual object salience.

Differences from source (listener_rationality_shared_prior):
- Recursion depth: Unchanged (choice_probs calls L1 at depth 1).
- Parameters added: cost (prior: Normal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In L1, the speaker's choice weight incorporates an utterance production cost penalty `- cost * vec(utt_cost, u)`, where `utt_cost` measures the excess extension of non-sink utterances across referents (`jnp.maximum(0.0, jnp.sum(ctx.lex * (1.0 - ctx.is_sink)[:, None], axis=1) - 1.0)`).
