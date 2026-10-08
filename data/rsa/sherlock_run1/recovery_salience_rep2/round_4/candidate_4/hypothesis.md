Listeners combine pragmatic reference resolution under an integrated Bayesian salience and base-rate prior with selective visual attention that suppresses distractor objects mismatching the spoken word. When evaluating alternative descriptions the speaker could have produced, resource-limited visual attention discounts non-matching distractors rather than inspecting them with full fidelity, sharpening pragmatic reference resolution. On uninformative trials without speech, visual attention remains uniform across all objects and choices are guided directly by the integrated Bayesian prior.

Refinement of bayesian_salience_base_rate_listener:
- Base model: bayesian_salience_base_rate_listener (best).
- Recursion depth: depth 1 (calls L1 in choice_probs, unchanged from bayesian_salience_base_rate_listener).
- Parameters added: distractor_attention (Beta(2.0, 2.0), attentional weight discounting mismatching distractor objects when evaluating speaker informativeness, taken from distractor_suppression_listener).
- Parameters modified: none.
- Parameters removed: none (retains alpha, salience_weight, base_rate_weight, and lapse).
- Mathematical / structural changes: In choice_probs, when an informative utterance is heard, candidate objects that mismatch the spoken word receive attenuated visual attention (distractor_attention); this attention vector is passed to literal listener L0 (weighting choice probabilities by at(lex, u, r) * (vec(obj_attention, r) + EPS)) and through to pragmatic listener L1, discounting non-matching distractors when evaluating alternative descriptions the speaker could have produced. On uninformative trials without speech (is_prior > 0), attention is uniform across all objects.
