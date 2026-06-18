# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## equally_likely  — posterior 0.039, ELPD-LOO -111.65

People judge a sequence as more random the closer its proportion of heads is to 50%.

## alternation_rate  — posterior 0.047, ELPD-LOO -111.43

People judge a sequence as more random if its proportion of alternations is closer to their subjective ideal alternation rate.

## bayesian_fair_coin  — posterior 0.769, ELPD-LOO -108.28

Observers compare two binary sequences via the log Bayes factor between a fair-coin null and a biased-coin alternative.

## iter0_candidate0  — posterior 0.054, ELPD-LOO -111.48

People judge a sequence as more random the higher its proportion of alternations, evaluating randomness via a linear monotonic preference rather than calculating distance to a subjective ideal alternation rate.

## iter0_candidate1  — posterior 0.044, ELPD-LOO -111.39

People judge a sequence as less random the longer its longest continuous run of identical outcomes. When comparing two sequences, they prefer the one with the shorter maximum run length as being more random.

## iter0_candidate2  — posterior 0.047, ELPD-LOO -111.34

People judge a sequence as less random if it contains periodic, repeating patterns. When comparing two sequences, they prefer the one with a lower periodicity score as being more randomly generated.

## iter1_candidate0

People evaluate a sequence's randomness by comparing its likelihood under a fair coin against its probability under a biased coin, mathematically marginalizing over all possible alternative biases rather than assuming a single fixed bias. They hold a symmetric prior belief about the alternative coin's bias, with the concentration of this prior acting as a subjective parameter, and they prefer the sequence that provides stronger Bayesian evidence for the fair coin.

## iter1_candidate1

Observers evaluate a sequence's randomness by computing the log Bayes factor between an independent fair coin and an alternative first-order Markov process with a fixed, subjective transition probability. Rather than evaluating the overall proportion of heads, they assess the sequential dependence, preferring sequences that provide stronger evidence against the Markov alternative.
