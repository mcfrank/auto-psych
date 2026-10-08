Listeners interpret referring expressions at depth 2 by integrating familiarization base rates with an intrinsic perceptual prior via normative Bayesian evidence combination rather than linear probability mixing. In the absence of prior exposure, listeners favor simpler objects with fewer visual features as their baseline expectation; when familiarization occurs, empirical exposure frequencies update this perceptual prior multiplicatively in log-probability space to guide communicative reference and uninformative guessing.

Refinement of base_rate_l2:
- Base model: base_rate_l2 (best).
- Recursion depth: depth 2 (calls L2 in choice_probs, unchanged from base_rate_l2).
- Parameters added: salience_weight (Normal(0.0, 1.0), sensitivity of the referent prior to visual feature count, taken from feature_salience_l2).
- Parameters modified: base_rate_weight (prior changed from Beta(2.0, 2.0) to LogNormal(0.0, 1.0) to represent the multiplicative log-odds sensitivity to exposure base rates).
- Parameters removed: none (retains alpha and lapse).
- Mathematical / structural changes: In choice_probs, referent prior expectations are formed by Bayesian log-linear evidence combination (softmax_prior(salience_weight * feature_count + base_rate_weight * log(familiarization))) rather than linear probability mixing with a uniform baseline. When familiarization is absent, the prior smoothly reduces to the intrinsic feature-salience prior; when familiarization is present, exposure frequencies update the prior in log-odds space.
