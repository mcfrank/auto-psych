Listeners interpret referring expressions by inverting a speaker who actively avoids ambiguity, penalizing candidate words that apply to multiple referents in the visual context. When evaluating potential descriptions, speakers balance communicative informativeness against an effort and ambiguity cost for shared features, making listeners infer that an ambiguous word was used only when no uniquely distinguishing feature was available. Prior expectations about candidate referents continue to integrate visual feature simplicity and familiarization base rates in log-odds space, guiding both pragmatic interpretation and uninformative guessing.

Refinement of bayesian_salience_base_rate_listener:
- Base model: bayesian_salience_base_rate_listener
- Recursion depth: depth 1 (calls L1 in choice_probs, unchanged from bayesian_salience_base_rate_listener).
- Parameters added: cost (HalfNormal(1.0), ambiguity cost penalizing utterances that apply to multiple referents, taken from costly_feature_speaker).
- Parameters modified: none.
- Parameters removed: none (retains alpha, salience_weight, base_rate_weight, and lapse).
- Mathematical / structural changes: In choice_probs and L1, each utterance incurs an ambiguity cost proportional to its context extension beyond a unique referent (costs = params["cost"] * jnp.maximum(0.0, extension - 1.0)); speaker S1 subtracts this cost from the communicative log-utility of each word (exp(alpha * (log(L0[u, r](lex) + {EPS}) - vec(costs, u)))), disincentivizing ambiguous referring expressions and sharpening the listener's pragmatic inference when hearing shared words. On uninformative trials without speech (is_prior > 0), choices reflect the Bayesian referent prior directly.
