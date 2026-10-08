Listeners combine pragmatic reference resolution under an integrated Bayesian salience and base-rate prior with selective distractor suppression and prior-guided lapses. When inattentive lapses occur during reference resolution, participants default to their baseline prior referent expectations—grounded in visual simplicity and familiarization base rates—rather than clicking uniformly at random across all display objects. On uninformative trials without speech, visual attention remains uniform across objects and choices reflect the integrated Bayesian prior directly.

Refinement of distractor_bayesian_salience_listener:
- Base model: distractor_bayesian_salience_listener (best).
- Recursion depth: depth 1 (calls L1 in choice_probs, unchanged from distractor_bayesian_salience_listener).
- Parameters added: none.
- Parameters modified: none.
- Parameters removed: none (retains alpha, salience_weight, base_rate_weight, distractor_attention, and lapse).
- Mathematical / structural changes: In choice_probs, the lapse error process replaces uniform guessing across all display objects (with_lapse mixing with 1.0 / N_OBJ) with prior-guided guessing ((1.0 - lapse) * heard + lapse * prior), such that inattentive choices default to baseline Bayesian prior expectations.
