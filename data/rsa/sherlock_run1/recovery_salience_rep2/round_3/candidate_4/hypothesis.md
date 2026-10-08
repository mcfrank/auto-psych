Listeners combine pragmatic reference resolution with an integrated object prior that combines visual feature complexity and empirical familiarization base rates via normative Bayesian log-odds evidence integration rather than linear probability mixing. In the absence of prior exposure, listeners rely on visual feature complexity, expecting speakers to refer to simpler objects; when familiarization occurs, exposure frequencies update this prior multiplicatively in log-probability space, weighting communicative reference and uninformative guessing.

Refinement of salience_base_rate_listener:
- Base model: salience_base_rate_listener
- Recursion depth: depth 1 (calls L1 in choice_probs, unchanged from salience_base_rate_listener).
- Parameters added: none.
- Parameters modified: base_rate_weight (prior changed from Beta(2.0, 2.0) to LogNormal(0.0, 1.0) to represent multiplicative log-odds sensitivity to empirical exposure base rates).
- Parameters removed: none (retains alpha, salience_weight, and lapse).
- Mathematical / structural changes: In choice_probs, the referent prior is formed by Bayesian log-linear evidence combination (softmax_prior(salience_weight * feature_count + base_rate_weight * log(familiarization))) rather than linear probability mixing ((1.0 - base_rate_weight) * salience_prior + base_rate_weight * familiarization). When familiarization is absent, the prior reduces directly to the intrinsic feature-salience baseline; when familiarization is present, empirical exposure updates the prior in log-odds space.
