# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## equally_likely  — posterior 0.004, ELPD-LOO -215.69

People judge a sequence as more random the closer its proportion of heads is to 50%.

## alternation_rate  — posterior 0.018, ELPD-LOO -214.02

People judge a sequence as more random if its proportion of alternations is closer to their subjective ideal alternation rate.

## bayesian_fair_coin  — posterior 0.484, ELPD-LOO -210.40

Observers compare two binary sequences via the log Bayes factor between a fair-coin null and a biased-coin alternative.

## inner_loop_model  — posterior 0.484, ELPD-LOO -210.40

Best PyMC model found by the inner model-improvement loop.

## subjective_markov_evidence  — posterior 0.007, ELPD-LOO -214.68

Observers evaluate randomness by computing the log Bayes factor of the sequence's transitions under a subjective ideal Markov process versus a purely independent fair coin.

## absolute_alternation_deviation  — posterior 0.004, ELPD-LOO -215.40

People judge a sequence as less random the further its number of alternations deviates from the expected number under their subjective ideal rate, computing distance in absolute counts rather than proportions.

## iter0_candidate0

Observers evaluate the randomness of a sequence by computing the true log Bayes factor between a fair-coin null and a biased-coin alternative, marginalizing over possible alternative biases using a subjective Beta prior instead of comparing to a single fixed bias.
