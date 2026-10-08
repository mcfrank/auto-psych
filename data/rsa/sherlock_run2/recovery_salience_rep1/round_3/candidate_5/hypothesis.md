We refine base_rate_simplicity_listener by extending it with the selective visual attention component from selective_attention_listener, positing that communicators combine prior expectations over referents with selective visual attention focused on candidate objects compatible with the heard utterance. When evaluating a speaker's communicative alternatives, listeners discount unattended distractors in the visual background, attenuating the influence of non-matching objects on pragmatic inference while preserving empirical base-rate and simplicity integration. On uninformative prior trials without an informative message, listeners allocate attention uniformly and rely on their integrated prior expectations.

Differences from base_rate_simplicity_listener:
- Recursion depth: unchanged (choice_probs calls L1, depth 1).
- Parameters added: distractor_attention (Beta(1.0, 1.0)).
- Parameters removed: none.
- Other terms: choice_probs computes an attention vector matching the heard utterance via att = jnp.where(ctx.lex[ctx.utterance] > 0, 1.0, params["distractor_attention"]) and passes it to L0 and L1; in L0, referent choice weights are multiplied by vec(att, r).
