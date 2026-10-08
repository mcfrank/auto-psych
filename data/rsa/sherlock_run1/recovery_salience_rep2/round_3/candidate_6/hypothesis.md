Listeners interpret referring expressions through first-order pragmatic reasoning, inverting a speaker who is informative to a literal listener rather than simulating higher recursive mental states, while integrating visual feature complexity with empirical familiarization base rates via Bayesian evidence combination. In the absence of prior exposure, listeners favor simpler objects with fewer visual features as their baseline expectation; when familiarization occurs, empirical exposure frequencies update this perceptual prior multiplicatively in log-probability space to guide communicative reference and uninformative guessing.

Refinement of bayesian_base_rate_l2:
- Base model: bayesian_base_rate_l2
- Recursion depth: depth 1 (calls L1 in choice_probs, whereas bayesian_base_rate_l2 calls L2 at depth 2).
- Parameters added: none.
- Parameters removed: none (retains alpha, salience_weight, base_rate_weight, and lapse).
- Mathematical / structural changes: In choice_probs, listeners perform first-order pragmatic reference resolution (calling L1) rather than second-order reasoning (L2), inverting a softmax-rational speaker who reasons about literal listener L0 under the Bayesian log-linear referent prior. The depth-2 recursive listener memo L2 is removed from the model.
