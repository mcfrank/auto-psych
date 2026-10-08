Listeners combine a shared Bayesian prior across communicative partners with selective visual attention that suppresses distractor objects mismatching the spoken word. When evaluating alternative descriptions the speaker could have produced, resource-limited visual attention discounts non-matching distractors rather than inspecting them with full fidelity, while communicative partners share the expectation that even literal interpretations are grounded in visual complexity and familiarization base rates. On uninformative trials without speech, visual attention remains uniform across all objects and choices reflect this integrated Bayesian prior directly.

Refinement of shared_prior_bayesian_listener:
- Base model: shared_prior_bayesian_listener
- Recursion depth: depth 1 (calls L1 in choice_probs, unchanged from shared_prior_bayesian_listener).
- Parameters added: distractor_attention (Beta(2.0, 2.0), attentional weight discounting mismatching distractor objects when evaluating speaker informativeness, taken from distractor_suppression_listener).
- Parameters modified: none.
- Parameters removed: none (retains alpha, salience_weight, base_rate_weight, and lapse).
- Mathematical / structural changes: In choice_probs, when an informative utterance is heard, candidate objects that mismatch the spoken word receive attenuated visual attention (distractor_attention); this attention vector is passed to literal listener L0 (weighting choice probabilities by at(lex, u, r) * (vec(prior, r) + {EPS}) * (vec(obj_attention, r) + {EPS})) and through to pragmatic listener L1, discounting non-matching distractors when evaluating alternative descriptions the speaker could have produced. On uninformative trials without speech (is_prior > 0), attention is uniform across all objects.
