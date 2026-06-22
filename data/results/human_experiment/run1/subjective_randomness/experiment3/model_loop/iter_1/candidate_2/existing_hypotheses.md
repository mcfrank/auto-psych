# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how well it currently explains the data. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## prototype_similarity  — posterior 0.000, ELPD-LOO -2330.85

Random-looking sequences are close to a prototype with balanced H/T counts and an ideal alternation rate.

## encoding_compressibility  — posterior 0.000, ELPD-LOO -2411.18

Random-looking sequences are those with low simple-description penalties: long runs, periodic templates, and imbalance.

## bayesian_diagnosticity  — posterior 0.000, ELPD-LOO -2335.55

Randomness is the log-likelihood ratio of a fair coin versus a regular process: a mixture of a complexity-penalized motif process (Griffiths et al. 2018) and a biased coin. Merges the former diagnosticity and statistical-inference accounts.

## window_typicality  — posterior 0.000, ELPD-LOO -2460.92

Random-looking sequences have a longest run typical of a fair coin seen through a limited memory window; over-long streaks look non-random (Hahn & Warren 2009).

## accumulated_alternation_typicality  — posterior 0.000, ELPD-LOO -2317.52

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but this typicality is based entirely on how closely the sequence's alternation rate matches a mental prototype, completely ignoring the overall proportion of heads and tails.

## linear_accumulated_typicality  — posterior 0.620, ELPD-LOO -2290.63

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but their penalty for deviating from the ideal head proportion and alternation rate is linear (absolute difference) rather than quadratic, treating extreme deviations proportionally to small ones.

## inner_loop_model  — posterior 0.074, ELPD-LOO -2292.76

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, evaluating how well a sequence conforms to a mental prototype for heads and alternations.

## power_law_accumulated_typicality  — posterior 0.036, ELPD-LOO -2293.12

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but this evidence integration exhibits diminishing returns scaling according to a power-law function of sequence length, which prevents extremely long sequences from disproportionately dominating judgments.

## leaky_accumulated_typicality  — posterior 0.028, ELPD-LOO -2293.22

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its events, but working memory for past events is leaky with exponential decay, meaning the total accumulated randomness score saturates for longer sequences.

## asymmetric_alternation_typicality  — posterior 0.242, ELPD-LOO -2291.12

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but their penalty for deviating from the ideal alternation rate is asymmetric: sequences that under-alternate (cluster) are heavily penalized, while those that over-alternate are penalized much less severely.

## iter0_candidate0  — posterior 0.000, ELPD-LOO -2321.04

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but we refine how alternation typicality is evaluated. Rather than penalizing the sequence's global average alternation rate, people evaluate typicality at the level of individual runs (streaks): they accumulate a penalty for each run that grows quadratically with how much its length deviates from an ideal run length. This non-linear run-level penalty explains why sequences with atypically long streaks are judged as significantly less random even when their overall alternation rates match.

## iter0_candidate1  — posterior 0.000, ELPD-LOO -2410.39

People evaluate the randomness of a sequence based solely on the proportion of the sequence occupied by its longest contiguous streak of identical outcomes, penalizing sequences where a single streak dominates the sequence length.

## iter0_candidate2  — posterior 0.000, ELPD-LOO -2533.62

People evaluate the randomness of a sequence solely by assessing whether its longest streak of identical outcomes is typical for a random sequence of that length. They linearly penalize sequences based on the absolute deviation of their observed maximum run length from the mathematically expected maximum run length of a fair coin (approximately log2 of the sequence length).

## iter1_candidate0

People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, computing each event's typicality as a penalty based on its distance from a mental prototype. Rather than assuming a strictly linear (City Block) or quadratic (Euclidean) penalty, this distance is computed using a freely inferred exponent (a Minkowski-like metric), allowing the model to naturally capture how human observers disproportionately scale the punishment for extreme feature deviations.

## iter1_candidate1

People evaluate the randomness of a sequence by accumulating the Kullback-Leibler (KL) divergence (relative entropy) between its empirical features—specifically its proportion of heads and alternation rate—and a mental prototype. By acting as intuitive statisticians measuring information-theoretic divergence rather than geometric distance, they inherently apply a progressively steeper penalty for extreme deviations (such as highly imbalanced proportions), while naturally accumulating this evidence over the length of the sequence.
