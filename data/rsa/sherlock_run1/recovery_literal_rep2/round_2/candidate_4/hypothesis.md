This model refines softmax_listener_shared_prior by incorporating the communicative alternative of silence into the speaker's decision process: speakers have the option to remain silent (the sink utterance) rather than produce an ambiguous or misleading word, incurring a baseline cost of silence. The pragmatic listener reasons about this communicative choice using their softmax decision policy, inferring that a speaker who chose to speak an informative word rather than remaining silent had a strong communicative motivation to refer to that object. This improves model fit because it grounds the informativeness of true words against the baseline utility of silence, preventing over-penalization of ambiguous utterances while preserving the shared salience prior and bounded listener decision rationality.

Differences from softmax_listener_shared_prior:
- Recursion depth: depth 1 (choice_probs calls L1, unchanged).
- Parameters added: cost_silence ~ Normal(0.0, 2.0) governing the speaker's baseline cost of silence.
- Parameters removed: none.
- Speaker choice in L1: speaker chooses u in UTT with probability proportional to (1.0 - vec(is_sink, u)) * at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + EPS)) + vec(is_sink, u) * exp(-cost_silence), allowing the speaker for any referent to choose silence against speaking.
- L1 arguments: L1 takes cost_silence and is_sink: ... from params["cost_silence"] and ctx.is_sink.
- All other memo agents, distributions, and prior terms are unchanged.
