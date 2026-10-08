Pragmatic speakers balance communicative informativeness against utterance production costs, penalizing non-distinctive words that apply to multiple referents in the visual scene, while coordinating on a shared salience prior over objects with listeners. By incorporating this utterance cost into the speaker's utility, the pragmatic listener expects speakers to prefer distinctive terms over shared ones even when both are literally truthful, sharpening pragmatic referent selection beyond object salience alone.

Differences from source (rsa_l1_shared_prior):
- Recursion depth: Unchanged (choice_probs calls L1 at depth 1).
- Parameters added: cost (prior: Normal(0.0, 1.0)).
- Parameters removed: None.
- Other terms: In L1, the speaker's choice weight incorporates an utterance production cost penalty `- cost * vec(utt_cost, u)`, where `utt_cost` measures the excess extension of non-sink utterances across referents (`jnp.maximum(0.0, jnp.sum(ctx.lex * (1.0 - ctx.is_sink)[:, None], axis=1) - 1.0)`).
