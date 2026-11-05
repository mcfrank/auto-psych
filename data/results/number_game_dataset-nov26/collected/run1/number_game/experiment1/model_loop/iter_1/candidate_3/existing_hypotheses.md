# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## place_value_exemplar_similarity  — rank 0, the best model on this data, ELPD-LOO -1624.2

People do not induce a rule; they judge a target by its summed similarity to the remembered examples (exemplar generalization), where similarity between two numbers is computed in a two-dimensional psychological space of overall magnitude and final digit. Similarity decays exponentially with the numeric distance between the numbers and drops by a fixed factor when their final digits differ, so targets near the examples or sharing an example's last digit feel like they "belong", and the more total similarity a target accrues the more likely the answer is yes.

## exemplar_magnitude_digit_similarity  — rank 1, 4.7 ± 5.9 nats behind the best (0.8× dse: statistically tied with the best), ELPD-LOO -1628.9

People do not reason about hidden rules at all; they judge a target by its summed similarity to the remembered example numbers (an exemplar / GCM-style process). Each number is represented psychologically by its magnitude and by its final digit, similarity to an example decays exponentially with attention-weighted city-block distance on those two dimensions, and the probability of saying "yes" is the target's summed similarity relative to a fixed response criterion (Luce choice rule).

## digit_numeral_exemplar_similarity  — rank 2, 31.4 ± 9.3 nats behind the best (3.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1655.5

People do not induce a rule or interval at all; they judge the target by its summed similarity to the examples treated as two-digit numerals, where similarity to an example falls off with the distance between tens digits and drops by a fixed factor when the final digits differ. This predicts partial generalization by shared final digit even in mixed sets (e.g. 72 or 12 after "92,82,27,42") and graded interpolation through neighbouring decades, and disagrees most sharply with the rule-based Bayesian model on sets that form a mathematical rule (e.g. powers of 2 or multiples of 7), where it predicts no special endorsement of rule members that share neither tens nor final digits with any example.

## tempered_size_principle_mixture  — rank 3, 81.1 ± 21.8 nats behind the best (3.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1705.3

Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 equally likely rules and all intervals of 1-100, but they apply the size principle only partially — the evidence each example provides is tempered by a single inferred exponent, so smaller concepts are favoured less steeply than strong sampling implies and people hedge between broad and narrow concepts. The one change is this free size-principle strength (an exponent on the example likelihood, applied identically to rules and intervals); "yes" is still the posterior probability that the concept contains the target.

## tempered_size_principle_concepts  — rank 4, 90.3 ± 28.8 nats behind the best (3.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1714.4

Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept learning over 45 rules and all intervals of 1-100 with independently weighted rules, but people apply a weakened (or strengthened) size principle — the likelihood of an example inside a concept falls with the concept's size raised to a fitted exponent rather than exactly one over its size. The single change is that exponent; a value below one lets broad intervals spanning scattered examples compete with small coincidental rules, producing more generalization within the examples' range.

## bigelow_piantadosi_independent  — rank 5, 154.1 ± 39.5 nats behind the best (3.9× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1778.3

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## interval_width_prior_concepts  — rank 6, 166.0 ± 39.4 nats behind the best (4.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1790.1

Refinement of `bigelow_piantadosi_independent`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size-principle sampling with 10% noise, "yes" = posterior probability the concept contains the target), but the one change is that intervals are not equally likely a priori — an interval's prior weight decays exponentially with its width at a rate inferred from the data, so people favour tight ranges hugging the examples and generalize more sharply to numbers inside the example span than to those outside it.

## bigelow_piantadosi_mixture  — rank 7, 218.1 ± 41.1 nats behind the best (5.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1842.3

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100. A rule is chosen with probability lambda_0, and the
rules are equally likely; each interval has prior (1 - lambda_0) / 100^2.
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.

## tenenbaum_rules_similarity  — rank 8, 431.7 ± 56.1 nats behind the best (7.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -2055.9

Bayesian concept learning over 33 mathematical rules (odd, even, squares,
cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every
interval of 1-100. The rules share a prior fraction lambda equally; the
intervals share the rest by an Erlang density on their size (scale sigma).
Examples are sampled from the concept (the size principle), 10% of them
uniformly from 1-100, and "yes" is the posterior probability that the
concept contains the target.
