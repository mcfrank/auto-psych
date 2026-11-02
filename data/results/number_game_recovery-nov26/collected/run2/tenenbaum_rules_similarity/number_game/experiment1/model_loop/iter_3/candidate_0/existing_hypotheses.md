# Existing hypotheses

Each model below is ONE cognitive hypothesis, with how it stands on the current data by ELPD-LOO (best first). `elpd_diff ± dse` is a model's deficit against the best and the standard error of that difference (with the responses to each stimulus pair counted together, as they are correlated): within about 2·dse the two are statistically tied on this data; beyond it the model has lost. "PSIS-LOO unreliable" means the estimate itself is untrustworthy (too many high-Pareto-k trials), not that the model is bad. Propose a hypothesis that is genuinely different from these, or a refinement of a single one of them — never a combination of several.

## noisy_examples_erlang_intervals  — rank 0, the best model on this data, ELPD-LOO -1210.3

A refinement of inferred_example_noise_concepts: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100, with the size principle and an inferred rate at which they believe examples are random noise, but they do not treat every interval as equally likely a priori. The one change is grafting the Erlang interval-size prior from erlang_interval_size_prior: people expect a "range" concept to have a typical width (inferred from the data), so they neither readily entertain tiny ranges hugging the examples nor ranges spanning most of 1-100, which governs how far they extend "yes" around examples that fit no rule.

## inferred_example_noise_concepts  — rank 1, 1.8 ± 1.5 nats behind the best (1.2× dse: statistically tied with the best), ELPD-LOO -1212.1

A refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but they do not assume a fixed 10% of examples are random noise; how often they believe the program emits an example from outside its concept is a free quantity inferred from the data. The one change is this inferred example-noise rate, which governs both how sharply the size principle accumulates over examples and how readily a rule is kept despite an example it does not cover.

## log_scale_interval_width_prior  — rank 2, 1.8 ± 1.8 nats behind the best (1.0× dse: statistically tied with the best), ELPD-LOO -1212.1

A refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100 (size principle, 10% of examples random) with a prior favouring ranges of a typical width, but they judge a range's width on a compressed, logarithmic mental number line rather than in raw counts. The one change is that the Erlang width prior applies to the ratio of a range's upper to lower end (its log-width), so a "typical" range is narrow among small numbers (e.g. 3-6) and wide among large ones (e.g. 60-100), making people extend "yes" further around large examples than around small ones that fit no rule.

## erlang_interval_size_prior  — rank 3, 2.3 ± 1.7 nats behind the best (1.4× dse: statistically tied with the best), ELPD-LOO -1212.5

A refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100 (size principle, 10% of examples random), but they do not treat every interval as equally likely a priori. Their prior over intervals favours ranges of a typical width (an Erlang prior on interval size, as in Tenenbaum's number game), with that preferred width inferred from the data, so they neither readily entertain tiny ranges hugging the examples nor ranges spanning most of 1-100; the one change is this size-dependent interval prior in place of the uniform one.

## log_number_line_interval_prior  — rank 4, 2.7 ± 1.5 nats behind the best (1.7× dse: statistically tied with the best), ELPD-LOO -1213.0

A refinement of inferred_example_noise_concepts: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100, with the size principle and an inferred rate of random examples, but they picture ranges of numbers on a compressed, logarithmic-like mental number line, so a priori they find ranges whose endpoints are small numbers more plausible than equally wide ranges among large numbers. The one change is this magnitude-dependent prior over intervals (the strength of the compression inferred from the data, with zero recovering the incumbent's uniform prior), which predicts that "yes" extends further below the examples than above them.

## bigelow_piantadosi_independent  — rank 5, 3.9 ± 1.9 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1214.2

Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every
interval of 1-100, with a prior weight inferred for each rule separately
and a share lambda_0 for the rules as a group; each interval has prior
(1 - lambda_0) / 100^2. Examples are sampled from the concept (the size
principle), 10% of them uniformly from 1-100, and "yes" is the posterior
probability that the concept contains the target.

## erlang_interval_width_prior  — rank 6, 4.6 ± 1.4 nats behind the best (3.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1214.9

Refinement of `bayesian_concepts_with_guessing_lapse`: people judge by Bayesian concept learning over the same 45 independently weighted rules and the intervals of 1-100 (size principle, with a guessing lapse), but they do not consider every range of numbers equally plausible a priori — they expect a "range" concept to have a typical width, favouring intervals of moderate size over both tiny and nearly-all-numbers ranges (an Erlang-shaped prior over interval size, as in Tenenbaum's number game). The one change is this size prior on intervals, with its preferred width inferred from the data, which governs how far people extend "yes" around examples that fit no rule.

## recency_decay_concept_learning  — rank 7, 4.9 ± 2.4 nats behind the best (2.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1215.2

People do Bayesian concept learning over the same rules and intervals (size principle, independently weighted rule prior), but their memory of the examples fades as they read the set: the most recently shown example counts in full, and each earlier example's evidence is discounted by a constant factor per later example. Because early examples are partly forgotten, concepts that fit the last examples but not the first ones keep substantial posterior weight, so generalisation is driven by the end of the list more than its beginning.

## bayesian_concepts_with_guessing_lapse  — rank 8, 5.3 ± 1.6 nats behind the best (3.2× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1215.5

A refinement of bigelow_piantadosi_independent: people still judge by Bayesian concept learning over the same 45 rules and all intervals of 1-100 (independently weighted rule prior, size principle), but on a fraction of trials they do not consult that posterior at all and instead give a guess, saying "yes" with their own fixed guessing rate. The one change is this response-lapse stage, which explains why even targets that are among the examples are often rejected and why clearly excluded targets are sometimes accepted.

## posterior_confidence_miscalibration  — rank 9, 7.7 ± 3.1 nats behind the best (2.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1217.9

People do the normative Bayesian concept learning of the incumbent model (45 rules with individually weighted priors and all intervals of 1-100, size principle, 10% random examples), but they mis-weight their own posterior confidence when turning it into a yes/no answer: the log-odds of saying "yes" are the posterior log-odds that the concept contains the target, multiplied by a single confidence-scaling factor. A factor below one means people under-trust their inference (strongly supported generalisations are answered less consistently and strongly excluded targets get more yeses, in a graded way rather than a flat lapse), and a factor above one means they over-commit to the more probable answer.

## noisy_posterior_criterion  — rank 10, 33.8 ± 6.3 nats behind the best (5.4× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1244.0

People do the same Bayesian concept learning as bigelow_piantadosi_independent (45 individually weighted rules and all intervals of 1-100, size principle, 10% random examples), but they answer "yes" by comparing their posterior belief that the target belongs to the concept against an internal criterion of how sure they must be, with noise in that comparison. The criterion (which need not be 50%) and the comparison noise are inferred; because the comparison is made on the probability itself, confident beliefs at either extreme give similar, non-certain answers, and beliefs below the criterion are mostly refused.

## win_stay_lose_shift_concepts  — rank 11, 61.1 ± 9.4 nats behind the best (6.5× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1271.4

People do not hold a full posterior over concepts; they keep a single running guess and revise it example by example (win-stay, lose-shift). After the first example they pick a concept (rule or range) in proportion to its Bayesian posterior given that example; each later example that fits the current guess leaves it unchanged, and only an example that does not fit makes them abandon it and re-pick a concept from the posterior given all examples seen so far. "Yes" is the probability that the guess they end with contains the target, so the size principle tightens generalisation only when a later example forces a switch, and the order of the examples changes which guess they end up holding.

## non_accumulating_size_principle  — rank 12, 84.5 ± 14.9 nats behind the best (5.7× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1294.7

People do Bayesian concept learning over the same rules and intervals, but they do not accumulate the size-principle evidence across examples: they judge how well a concept fits the examples by the average fit of one example, sharpened by a fixed amount, so their generalisation is no tighter after four examples than after one. This disagrees most with the current best model on sets with one or two examples, where it predicts sharper, more rule-like generalisation, and on four-example sets, where it predicts broader, more graded generalisation that keeps giving partial "yes" to targets that fit only a looser concept (for example, odd numbers that are not in a narrow rule the examples happen to share).

## arithmetic_vs_digit_rule_grammar  — rank 13, 104.7 ± 17.5 nats behind the best (6.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1314.9

Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, 10% of examples random), but their prior over rules is not flat. It first chooses between two kinds of rule, arithmetic properties of the number (odd, even, primes, squares, cubes, powers, multiples) or patterns in how it is written in decimal (ends in a digit, contains a digit), and then picks uniformly within that kind. The one change is this two-kind grammar step, with the share of prior given to written-digit rules inferred from the data rather than fixed by how many such rules there are.

## bigelow_piantadosi_mixture  — rank 14, 116.7 ± 16.6 nats behind the best (7.1× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1327.0

Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

## weak_sampling_concept_learning  — rank 15, 175.9 ± 19.5 nats behind the best (9.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1386.2

People infer which concept (one of the 45 arithmetic or digit rules, or an interval of 1-100) produced the examples, but they use weak sampling rather than the size principle: a concept is judged only by whether it is consistent with the examples, not by how small it is, so a broad concept like "even numbers" that contains the examples is exactly as well supported as a narrow one like "powers of 2". Their answer is the posterior probability that the concept contains the target, which predicts generalisation that stays broad and does not tighten as more consistent examples are seen.

## exemplar_feature_similarity  — rank 16, 321.4 ± 26.9 nats behind the best (12.0× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1531.6

People do not infer a rule at all; they judge a target by its summed similarity to the remembered examples (exemplar generalization), where similarity between two numbers falls off exponentially with how many simple arithmetic properties they disagree on (parity, divisibility by 3-10, last digit, tens digit, being prime, square or a power of two) plus how far apart they are in magnitude. A target that closely resembles even one example is accepted, with no size-principle penalty and no requirement that it fit all examples jointly.

## exemplar_attention_similarity  — rank 17, 337.8 ± 27.4 nats behind the best (12.3× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1548.1

People do not infer a rule; they judge the target by its summed similarity to the stored examples (exemplar-based generalization, as in the Generalized Context Model). Each number is represented on a few psychological dimensions — magnitude, parity, final digit and tens digit — each with its own attention weight, similarity to an example decays exponentially with attention-weighted mismatch, and the probability of "yes" rises with the target's average similarity to the examples.

## spacing_scaled_magnitude_similarity  — rank 18, 344.9 ± 27.5 nats behind the best (12.6× dse: distinguishable from the best — it has lost on this data), ELPD-LOO -1555.2

People judge a target by its numerical closeness to the nearest example on the 1-100 number line, with no rule inference: the chance of "yes" falls off smoothly with the distance from the target to the closest example. The width of this generalization gradient is set by how spread out the examples are — the mean gap between neighbouring examples (plus a baseline width for single examples) — so widely spaced examples license "yes" to far-away targets, while tightly clustered examples restrict "yes" to targets right beside them.
