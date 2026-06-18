# Theory Report — Experiment 2

## subjective_markov_evidence
**Hypothesis:** Observers evaluate randomness by computing the log Bayes factor of the sequence's transitions under a subjective ideal Markov process versus a purely independent fair coin.
**Motivation:** In the previous experiment, the `bayesian_fair_coin` model performed best (posterior mass > 0.7) because it scaled deviations by sequence length (a property of log-likelihoods), but it wrongly assumed people focus on the proportion of heads rather than alternations. A previous candidate (`iter1_candidate1`) used a Markov process but inverted the evidence (treating the Markov process as the alternative *against* the fair coin). This model fixes both by computing the Bayes factor of the transitions, favoring the subjective Markov process.
**Mechanism:** Computes `log P(transitions | subjective_theta) - log P(transitions | 0.5)` and uses a softmax over this difference. This inherently makes the judgment more sensitive to longer sequences, capturing the length-sensitivity lacking in the simple `alternation_rate` model.

## absolute_alternation_deviation
**Hypothesis:** People judge a sequence as less random the further its number of alternations deviates from the expected number under their subjective ideal rate, computing distance in absolute counts rather than proportions.
**Motivation:** The `alternation_rate` model from Experiment 1 performed poorly compared to length-sensitive models, suggesting that the standard metric (proportion) fails because it doesn't penalize deviations in longer sequences enough. Scaling the error by length might capture the necessary length-sensitivity without a full Bayesian calculation.
**Mechanism:** Instead of taking the absolute difference of proportions `abs(p_alts - ideal_rate)`, it computes the absolute difference in expected counts `abs(alts - (n - 1) * ideal_rate)`, increasing the penalty for sequences of greater length.
