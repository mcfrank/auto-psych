Listeners combine pragmatic reference resolution with an integrated object prior that unites visual feature complexity and empirical familiarization base rates via normative Bayesian evidence combination rather than linear probability mixing. In the absence of prior exposure, listeners favor simpler objects with fewer visual features as their baseline expectation; when familiarization occurs, empirical exposure frequencies update this perceptual prior multiplicatively in log-probability space to guide communicative reference and uninformative guessing.

Refinement of salience_base_rate_listener:
- Base model: salience_base_rate_listener (best).
- Recursion depth: depth 1 (calls L1 in choice_probs, unchanged from salience_base_rate_listener).
- Parameters added: none.
- Parameters modified: base_rate_weight (prior changed from Beta(2.0, 2.0) to LogNormal(0.0, 1.0) to represent the multiplicative log-odds sensitivity to exposure base rates).
- Parameters removed: none (retains alpha, salience_weight, and lapse).
- Mathematical / structural changes: In choice_probs, referent prior expectations are formed by Bayesian log-linear evidence combination (softmax_prior(salience_weight * feature_count + base_rate_weight * log(familiarization))) rather than linear probability mixing ((1.0 - base_rate_weight) * salience_prior + base_rate_weight * familiarization). When familiarization is absent, the prior smoothly reduces to the intrinsic feature-salience prior; when familiarization is present, exposure frequencies update the perceptual prior in log-odds space rather than through a probability mixture.
