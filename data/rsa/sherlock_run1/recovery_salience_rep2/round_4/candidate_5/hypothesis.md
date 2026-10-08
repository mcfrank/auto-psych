Listeners engage in first-order pragmatic reference resolution under a shared Bayesian prior that is common knowledge across communicative partners down to literal semantic interpretation. Rather than assuming literal listeners resolve ambiguous expressions equiprobably among all matching referents, speakers anticipate that even literal interpretations are grounded in visual feature complexity and familiarization base rates. This mutual prior grounding guides both literal semantic plausibility and pragmatic informativeness throughout recursive communication, aligning listener inferences with shared expectations when resolving ambiguous utterances or guessing without speech.

Refinement of bayesian_salience_base_rate_listener:
- Base model: bayesian_salience_base_rate_listener (best).
- Recursion depth: depth 1 (calls L1 in choice_probs, unchanged from bayesian_salience_base_rate_listener).
- Parameters added: none.
- Parameters modified: none.
- Parameters removed: none (retains alpha, salience_weight, base_rate_weight, and lapse).
- Mathematical / structural changes: In literal listener L0, referent choices among matching objects are grounded in the Bayesian referent prior (listener: chooses(r in OBJ, wpp=at(lex, u, r) * (vec(prior, r) + {EPS}))) rather than assuming an unconditional uniform distribution across matching referents (wpp=at(lex, u, r)), and L0 takes the referent prior as an array parameter. In pragmatic listener L1, the simulated speaker evaluates informativeness relative to this prior-grounded literal listener (calling L0[u, r](lex, prior) + {EPS}), establishing a shared prior that shapes reference resolution at every communicative level.
