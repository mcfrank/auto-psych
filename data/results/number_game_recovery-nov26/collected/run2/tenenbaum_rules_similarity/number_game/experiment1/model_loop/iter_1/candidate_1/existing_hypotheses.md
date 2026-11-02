# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## bigelow_piantadosi_independent  — rank 0, the best model on this data, ELPD-LOO -1214.2

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## bayesian_concepts_with_guessing_lapse  — rank 1, 1.4 ± 0.8 nats behind the best (1.8× dse: statistically tied with the best), ELPD-LOO -1215.5

A refinement of bigelow_piantadosi_independent: people still judge by Bayesian concept learning over the same 45 rules and all intervals of 1-100 (independently weighted rule prior, size principle), but on a fraction of trials they do not consult that posterior at all and instead give a guess, saying "yes" with their own fixed guessing rate. The one change is this response-lapse stage, which explains why even targets that are among the examples are often rejected and why clearly excluded targets are sometimes accepted.

## non_accumulating_size_principle  — rank 2, 80.6 ± 15.1 nats behind the best (5.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1294.7

People do Bayesian concept learning over the same rules and intervals, but they do not accumulate the size-principle evidence across examples: they judge how well a concept fits the examples by the average fit of one example, sharpened by a fixed amount, so their generalisation is no tighter after four examples than after one. This disagrees most with the current best model on sets with one or two examples, where it predicts sharper, more rule-like generalisation, and on four-example sets, where it predicts broader, more graded generalisation that keeps giving partial "yes" to targets that fit only a looser concept (for example, odd numbers that are not in a narrow rule the examples happen to share).

## arithmetic_vs_digit_rule_grammar  — rank 3, 100.8 ± 17.4 nats behind the best (5.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1314.9

Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, 10% of examples random), but their prior over rules is not flat. It first chooses between two kinds of rule, arithmetic properties of the number (odd, even, primes, squares, cubes, powers, multiples) or patterns in how it is written in decimal (ends in a digit, contains a digit), and then picks uniformly within that kind. The one change is this two-kind grammar step, with the share of prior given to written-digit rules inferred from the data rather than fixed by how many such rules there are.

## bigelow_piantadosi_mixture  — rank 4, 112.8 ± 16.5 nats behind the best (6.8× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1327.0

Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

## exemplar_feature_similarity  — rank 5, 317.5 ± 27.5 nats behind the best (11.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1531.6

People do not infer a rule at all; they judge a target by its summed similarity to the remembered examples (exemplar generalization), where similarity between two numbers falls off exponentially with how many simple arithmetic properties they disagree on (parity, divisibility by 3-10, last digit, tens digit, being prime, square or a power of two) plus how far apart they are in magnitude. A target that closely resembles even one example is accepted, with no size-principle penalty and no requirement that it fit all examples jointly.

## exemplar_attention_similarity  — rank 6, 333.9 ± 28.0 nats behind the best (11.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1548.1

People do not infer a rule; they judge the target by its summed similarity to the stored examples (exemplar-based generalization, as in the Generalized Context Model). Each number is represented on a few psychological dimensions — magnitude, parity, final digit and tens digit — each with its own attention weight, similarity to an example decays exponentially with attention-weighted mismatch, and the probability of "yes" rises with the target's average similarity to the examples.
