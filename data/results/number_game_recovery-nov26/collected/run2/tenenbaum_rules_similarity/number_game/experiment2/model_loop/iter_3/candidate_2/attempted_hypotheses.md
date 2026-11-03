# Tried before

43 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### exemplar_attention_similarity — pruned (experiment1 end of experiment)

**Outcome:** 341.6 nats behind without_replacement_example_sampling (12.4× dse)

**Hypothesis:** People do not infer a rule; they judge the target by its summed similarity to the stored examples (exemplar-based generalization, as in the Generalized Context Model). Each number is represented on a few psychological dimensions — magnitude, parity, final digit and tens digit — each with its own attention weight, similarity to an example decays exponentially with attention-weighted mismatch, and the probability of "yes" rises with the target's average similarity to the examples.

### non_accumulating_size_principle — pruned (experiment1 end of experiment)

**Outcome:** 88.2 nats behind without_replacement_example_sampling (5.7× dse)

**Hypothesis:** People do Bayesian concept learning over the same rules and intervals, but they do not accumulate the size-principle evidence across examples: they judge how well a concept fits the examples by the average fit of one example, sharpened by a fixed amount, so their generalisation is no tighter after four examples than after one. This disagrees most with the current best model on sets with one or two examples, where it predicts sharper, more rule-like generalisation, and on four-example sets, where it predicts broader, more graded generalisation that keeps giving partial "yes" to targets that fit only a looser concept (for example, odd numbers that are not in a narrow rule the examples happen to share).

### exemplar_feature_similarity — pruned (experiment1 end of experiment)

**Outcome:** 325.1 nats behind without_replacement_example_sampling (12.0× dse)

**Hypothesis:** People do not infer a rule at all; they judge a target by its summed similarity to the remembered examples (exemplar generalization), where similarity between two numbers falls off exponentially with how many simple arithmetic properties they disagree on (parity, divisibility by 3-10, last digit, tens digit, being prime, square or a power of two) plus how far apart they are in magnitude. A target that closely resembles even one example is accepted, with no size-principle penalty and no requirement that it fit all examples jointly.

### bayesian_concepts_with_guessing_lapse — pruned (experiment1 end of experiment)

**Outcome:** 9.0 nats behind without_replacement_example_sampling (3.9× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people still judge by Bayesian concept learning over the same 45 rules and all intervals of 1-100 (independently weighted rule prior, size principle), but on a fraction of trials they do not consult that posterior at all and instead give a guess, saying "yes" with their own fixed guessing rate. The one change is this response-lapse stage, which explains why even targets that are among the examples are often rejected and why clearly excluded targets are sometimes accepted.

### bayesian_concepts_with_guess_lapse — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_independent repair 1)

**Outcome:** predicts like existing model 'bayesian_concepts_with_guessing_lapse' (p_yes RMSE 0.00154 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of bayesian_concepts_with_guessing_lapse, not a new hypothesis.

**Hypothesis:** Refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules and all intervals (rule weights inferred separately, size principle, 10% noisy examples), but on a fraction of trials they do not consult that posterior at all and instead guess, answering "yes" with a fixed bias. The one change is this response-lapse process: it predicts that even targets among the examples, or clearly inside the learned concept, get "no" at a non-trivial rate (and clearly excluded targets get occasional "yes"), which the incumbent, whose answers follow the posterior exactly, cannot produce.

### arithmetic_vs_digit_rule_grammar — pruned (experiment1 end of experiment)

**Outcome:** 108.4 nats behind without_replacement_example_sampling (6.0× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 rules and all intervals of 1-100 (size principle, 10% of examples random), but their prior over rules is not flat. It first chooses between two kinds of rule, arithmetic properties of the number (odd, even, primes, squares, cubes, powers, multiples) or patterns in how it is written in decimal (ends in a digit, contains a digit), and then picks uniformly within that kind. The one change is this two-kind grammar step, with the share of prior given to written-digit rules inferred from the data rather than fixed by how many such rules there are.

### recency_decay_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 8.6 nats behind without_replacement_example_sampling (2.6× dse)

**Hypothesis:** People do Bayesian concept learning over the same rules and intervals (size principle, independently weighted rule prior), but their memory of the examples fades as they read the set: the most recently shown example counts in full, and each earlier example's evidence is discounted by a constant factor per later example. Because early examples are partly forgotten, concepts that fit the last examples but not the first ones keep substantial posterior weight, so generalisation is driven by the end of the list more than its beginning.

### posterior_confidence_miscalibration — pruned (experiment1 end of experiment)

**Outcome:** 11.4 nats behind without_replacement_example_sampling (3.3× dse)

**Hypothesis:** People do the normative Bayesian concept learning of the incumbent model (45 rules with individually weighted priors and all intervals of 1-100, size principle, 10% random examples), but they mis-weight their own posterior confidence when turning it into a yes/no answer: the log-odds of saying "yes" are the posterior log-odds that the concept contains the target, multiplied by a single confidence-scaling factor. A factor below one means people under-trust their inference (strongly supported generalisations are answered less consistently and strongly excluded targets get more yeses, in a graded way rather than a flat lapse), and a factor above one means they over-commit to the more probable answer.

### weak_sampling_concept_learning — pruned (experiment1 end of experiment)

**Outcome:** 179.7 nats behind without_replacement_example_sampling (8.8× dse)

**Hypothesis:** People infer which concept (one of the 45 arithmetic or digit rules, or an interval of 1-100) produced the examples, but they use weak sampling rather than the size principle: a concept is judged only by whether it is consistent with the examples, not by how small it is, so a broad concept like "even numbers" that contains the examples is exactly as well supported as a narrow one like "powers of 2". Their answer is the posterior probability that the concept contains the target, which predicts generalisation that stays broad and does not tighten as more consistent examples are seen.

### inferred_example_noise_concepts — pruned (experiment1 end of experiment)

**Outcome:** 5.6 nats behind without_replacement_example_sampling (2.7× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 rules (each with its own inferred prior weight) and all intervals of 1-100 with the size principle, but they do not assume a fixed 10% of examples are random noise; how often they believe the program emits an example from outside its concept is a free quantity inferred from the data. The one change is this inferred example-noise rate, which governs both how sharply the size principle accumulates over examples and how readily a rule is kept despite an example it does not cover.

### erlang_interval_size_prior — pruned (experiment1 end of experiment)

**Outcome:** 6.0 nats behind without_replacement_example_sampling (3.0× dse)

**Hypothesis:** A refinement of bigelow_piantadosi_independent: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100 (size principle, 10% of examples random), but they do not treat every interval as equally likely a priori. Their prior over intervals favours ranges of a typical width (an Erlang prior on interval size, as in Tenenbaum's number game), with that preferred width inferred from the data, so they neither readily entertain tiny ranges hugging the examples nor ranges spanning most of 1-100; the one change is this size-dependent interval prior in place of the uniform one.

### erlang_interval_width_prior — pruned (experiment1 end of experiment)

**Outcome:** 8.3 nats behind without_replacement_example_sampling (4.0× dse)

**Hypothesis:** Refinement of `bayesian_concepts_with_guessing_lapse`: people judge by Bayesian concept learning over the same 45 independently weighted rules and the intervals of 1-100 (size principle, with a guessing lapse), but they do not consider every range of numbers equally plausible a priori — they expect a "range" concept to have a typical width, favouring intervals of moderate size over both tiny and nearly-all-numbers ranges (an Erlang-shaped prior over interval size, as in Tenenbaum's number game). The one change is this size prior on intervals, with its preferred width inferred from the data, which governs how far people extend "yes" around examples that fit no rule.

### win_stay_lose_shift_concepts — pruned (experiment1 end of experiment)

**Outcome:** 64.9 nats behind without_replacement_example_sampling (6.6× dse)

**Hypothesis:** People do not hold a full posterior over concepts; they keep a single running guess and revise it example by example (win-stay, lose-shift). After the first example they pick a concept (rule or range) in proportion to its Bayesian posterior given that example; each later example that fits the current guess leaves it unchanged, and only an example that does not fit makes them abandon it and re-pick a concept from the posterior given all examples seen so far. "Yes" is the probability that the guess they end with contains the target, so the size principle tightens generalisation only when a later example forces a switch, and the order of the examples changes which guess they end up holding.

### spacing_scaled_magnitude_similarity — pruned (experiment1 end of experiment)

**Outcome:** 348.7 nats behind without_replacement_example_sampling (12.6× dse)

**Hypothesis:** People judge a target by its numerical closeness to the nearest example on the 1-100 number line, with no rule inference: the chance of "yes" falls off smoothly with the distance from the target to the closest example. The width of this generalization gradient is set by how spread out the examples are — the mean gap between neighbouring examples (plus a baseline width for single examples) — so widely spaced examples license "yes" to far-away targets, while tightly clustered examples restrict "yes" to targets right beside them.

### noisy_posterior_criterion — pruned (experiment1 end of experiment)

**Outcome:** 37.5 nats behind without_replacement_example_sampling (5.7× dse)

**Hypothesis:** People do the same Bayesian concept learning as bigelow_piantadosi_independent (45 individually weighted rules and all intervals of 1-100, size principle, 10% random examples), but they answer "yes" by comparing their posterior belief that the target belongs to the concept against an internal criterion of how sure they must be, with noise in that comparison. The criterion (which need not be 50%) and the comparison noise are inferred; because the comparison is made on the probability itself, confident beliefs at either extreme give similar, non-certain answers, and beliefs below the criterion are mostly refused.

### log_number_line_interval_prior — pruned (experiment1 end of experiment)

**Outcome:** 6.5 nats behind without_replacement_example_sampling (3.2× dse)

**Hypothesis:** A refinement of inferred_example_noise_concepts: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100, with the size principle and an inferred rate of random examples, but they picture ranges of numbers on a compressed, logarithmic-like mental number line, so a priori they find ranges whose endpoints are small numbers more plausible than equally wide ranges among large numbers. The one change is this magnitude-dependent prior over intervals (the strength of the compression inferred from the data, with zero recovering the incumbent's uniform prior), which predicts that "yes" extends further below the examples than above them.

### noisy_examples_erlang_intervals — pruned (experiment1 end of experiment)

**Outcome:** 3.8 nats behind without_replacement_example_sampling (3.0× dse)

**Hypothesis:** A refinement of inferred_example_noise_concepts: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100, with the size principle and an inferred rate at which they believe examples are random noise, but they do not treat every interval as equally likely a priori. The one change is grafting the Erlang interval-size prior from erlang_interval_size_prior: people expect a "range" concept to have a typical width (inferred from the data), so they neither readily entertain tiny ranges hugging the examples nor ranges spanning most of 1-100, which governs how far they extend "yes" around examples that fit no rule.

### log_scale_interval_width_prior — pruned (experiment1 end of experiment)

**Outcome:** 5.6 nats behind without_replacement_example_sampling (2.5× dse)

**Hypothesis:** A refinement of erlang_interval_size_prior: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100 (size principle, 10% of examples random) with a prior favouring ranges of a typical width, but they judge a range's width on a compressed, logarithmic mental number line rather than in raw counts. The one change is that the Erlang width prior applies to the ratio of a range's upper to lower end (its log-width), so a "typical" range is narrow among small numbers (e.g. 3-6) and wide among large ones (e.g. 60-100), making people extend "yes" further around large examples than around small ones that fit no rule.

### offset_arithmetic_progressions — pruned (experiment1 end of experiment)

**Outcome:** 22.6 nats behind without_replacement_example_sampling (4.0× dse)

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning (size principle, inferred share of random examples, Erlang prior on interval width), but their space of rules also includes arithmetic progressions that do not start at zero — every number that leaves a given remainder when divided by 3 to 12 (e.g. 2, 5, 8, 11, ... or 5, 11, 17, 23, ...), each such "step" rule equally likely a priori and the group as a whole given a prior share inferred from the data. The one change is this added family of offset arithmetic-progression concepts, which predicts that examples sharing a step-and-offset pattern (e.g. 23, 35, 83, 95, all 5 more than a multiple of 6) are generalised to that narrow progression rather than to a broad rule such as odd numbers.

### exemplar_consequential_region_similarity — pruned (experiment1 end of experiment)

**Outcome:** 378.3 nats behind without_replacement_example_sampling (10.9× dse)

**Hypothesis:** People do not infer one concept that covers all the examples together; they store each example as a separate exemplar and judge the target by its average similarity to the stored exemplars. The similarity of the target to one exemplar is Shepard-style generalization through consequential regions: how likely a number concept (an arithmetic or digit rule, or a range of a typical width) that contains that single exemplar also contains the target, with smaller concepts counting more. Because each exemplar is generalized from on its own and the results are summed, a target resembling only some of the examples gets partial "yes", and evidence never tightens jointly across examples as it would under concept inference.

### pedagogical_example_selection — dropped (experiment1 round 3)

**Outcome:** fold 2 too slow to fit: the fit of 'pedagogical_example_selection' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People assume the examples were chosen by a helpful teacher rather than drawn at random from the concept (pedagogical sampling): a teacher picks examples that point to the true concept, so a number counts more as evidence for a concept the fewer other plausible concepts it also belongs to. People do Bayesian concept learning over the same rules and ranges of 1-100, but they interpret each example by how diagnostic it is. Concepts made up mostly of numbers a teacher would happily show (distinctive, rarely shared ones) are penalised when such numbers were not shown, so generalisation narrows to the concepts that the chosen examples single out.

### new_set_first_query_caution — rejected (experiment1 round 3 candidate 3 refine incumbent noisy_examples_erlang_intervals)

**Outcome:** model binds ['trial_index'] — response-row bookkeeping a bare stimulus never carries — so it cannot be evaluated on a stimulus pool (the novelty gate now, the held-out evaluation later); compute everything from number_set and target.

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning over the 45 independently weighted rules and Erlang-size-weighted intervals of 1-100, with the size principle and an inferred example-noise rate, but on the first question asked right after a new set of examples appears they are cautious and, some inferred fraction of the time, answer "no" regardless of their posterior. The one change is this new-set caution, which predicts a lower yes-rate on the first query about a set than on later queries repeating the same set.

### new_set_cautious_erlang_concepts — pruned (experiment1 end of experiment)

**Outcome:** 6.1 nats behind without_replacement_example_sampling (2.7× dse)

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning (45 independently weighted rules, all intervals of 1-100 with an Erlang prior favouring a typical range width, size principle, inferred rate of random examples), but on the first question after a new set of examples appears they are initially cautious and withhold some of the "yes" answers that inference supports, only fully committing to their generalisation once they have been asked about the same set again. The one change is this new-set caution: on the first query after a set change the probability of "yes" is the posterior probability that the concept contains the target scaled down by an inferred withholding fraction, while repeated queries about the same set follow the posterior unchanged.

### noisy_examples_log_width_intervals — dropped (experiment1 round 3)

**Outcome:** fold 0 too slow to fit: the fit of 'noisy_examples_log_width_intervals' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** A refinement of log_scale_interval_width_prior: people do Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100 with the size principle, and they expect a range to have a typical width judged on a compressed, logarithmic mental number line (narrow among small numbers, wide among large ones), but they do not assume a fixed 10% of examples are random. The one change is grafting the inferred example-noise rate from inferred_example_noise_concepts: how often people believe the program emits an example from outside its concept is inferred from the data, governing how sharply the size principle accumulates over examples and how readily a rule or range survives an example it does not cover.

### flexible_shape_interval_width_prior — pruned (experiment1 end of experiment)

**Outcome:** 4.9 nats behind without_replacement_example_sampling (3.8× dse)

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100, with the size principle and an inferred rate of random examples, and a prior over how wide a "range" concept is, but that width prior is not fixed to the Erlang shape. The one change is that the shape of the width prior is inferred along with its typical width (a gamma-shaped prior whose shape can fall below the Erlang value), so people can give tight ranges hugging the examples high prior weight while still allowing occasional wide ranges. This predicts that people extend "yes" beyond tightly clustered examples less than the incumbent does, while still generalising across widely spread ones.

### two_rule_conjunction_concepts — rejected (experiment1 round 4 candidate 0 lens 1)

**Outcome:** predicts like existing model 'noisy_examples_erlang_intervals' (p_yes RMSE 0.00193 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of noisy_examples_erlang_intervals, not a new hypothesis.

**Hypothesis:** People build number concepts compositionally: besides the 45 simple rules and the ranges of 1-100 (learned by Bayesian inference with the size principle, an inferred rate of random examples and a typical-width prior on ranges), they also entertain conjunctions of two simple rules, such as "odd multiples of 3" or "even numbers containing the digit 4", each equally likely a priori with the conjunctions as a group taking an inferred share of the rule prior. Because a conjunction is much smaller than either of its parts, the size principle makes it dominate when all examples share two properties. The sharpest disagreement with the current best model is on such sets (e.g. 69, 9, 39, 21 or 63, 27, 99, 33), where this model predicts "no" to targets that have only one of the two shared properties (even multiples of 3) and the best model predicts substantial "yes".

### decade_aligned_interval_prior — pruned (experiment1 end of experiment)

**Outcome:** 6.9 nats behind without_replacement_example_sampling (2.7× dse)

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100 (size principle, an inferred share of random examples, a prior favouring a typical range width), but they think of ranges in round, decade-aligned terms ("20 to 40", "the 60s", "1 to 10"), so a range whose endpoints fall on decade boundaries is a priori more plausible than an equally wide range with arbitrary endpoints such as 23-47. The one change is this preference for decade-aligned range endpoints, with its strength (and sign) inferred from the data, which predicts that "yes" answers to examples that fit no rule extend up to, and stop at, decade lines.

### stochastic_example_forgetting — dropped (experiment1 round 4)

**Outcome:** fold 0 too slow to fit: the fit of 'stochastic_example_forgetting' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People do the same Bayesian concept learning as noisy_examples_erlang_intervals (45 independently weighted rules, intervals of 1-100 with an Erlang prior on width, size principle, inferred rate of random examples), but their working memory for the examples is unreliable: each example shown is, independently and with an inferred probability, simply not retained when they come to judge the target, and they generalise from whichever examples they still hold. Because a forgotten example removes its evidence regardless of how well it fit, this predicts broader, less rule-tight generalisation from larger sets than full-memory inference (e.g. some "yes" to even non-powers after 16, 8, 64, 2), and a target that matches only some of the examples still gets partial "yes".

### noise_coupled_yes_floor — pruned (experiment1 end of experiment)

**Outcome:** 33.4 nats behind without_replacement_example_sampling (5.4× dse)

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning (45 independently weighted rules, all intervals of 1-100 with an Erlang prior on range width, size principle, inferred share of random examples), but they answer "would the program produce this number?" with the same generative model they used to explain the examples. Since they believe the program emits a random number from 1-100 on some share of its outputs, they say "yes" whenever the target is in the inferred concept or when they think the program might be emitting a random number, so the one change is that the inferred noise rate also sets a floor of "yes" for targets outside every plausible concept (no new free parameter, unlike a free guessing lapse).

### decade_aligned_range_concepts — pruned (experiment1 end of experiment)

**Outcome:** 3.8 nats behind without_replacement_example_sampling (2.8× dse)

**Hypothesis:** A refinement of noisy_examples_erlang_intervals: people do the same Bayesian concept learning over the 45 independently weighted rules and all intervals of 1-100, with the size principle, an inferred rate of random examples and an Erlang prior favouring a typical range width, but they do not find every range equally natural: ranges whose ends fall on decade boundaries (starting at 1, 11, 21, ... or ending at 10, 20, ..., 100, such as "the 30s" or "1 to 20") are a priori more plausible than equally wide ranges with arbitrary ends. The one change is this decade-alignment boost to the interval prior (its strength inferred, zero recovering the incumbent), which predicts that "yes" around examples that fit no rule stops sharply at decade boundaries, e.g. examples 32 and 37 generalise to 30-39 more than to 28 or 41.

### overcommitted_noisy_erlang_concepts — pruned (experiment1 end of experiment)

**Outcome:** 5.1 nats behind without_replacement_example_sampling (2.3× dse)

**Hypothesis:** I chose posterior_confidence_miscalibration and refined it. People turn their posterior belief that the target belongs to the concept into a yes/no answer by scaling its log-odds with a single confidence factor. The factor is inferred; above one means they commit more firmly than their belief warrants. The one change is the concept learner underneath: the old one assumed a fixed 10% of random examples and treated all intervals as equally likely. It is replaced by one that infers how often examples are random and expects ranges of a typical width (Erlang). This claims that people's beliefs about the concept are broad, because they discount examples as possibly random, but that they answer more decisively than those broad beliefs imply, giving firmer yes and no answers than calibrated Bayesian inference would.

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Outcome:** 120.5 nats behind without_replacement_example_sampling (7.3× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_independent — pruned (experiment1 end of experiment)

**Outcome:** 7.7 nats behind without_replacement_example_sampling (3.2× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100, with a prior weight inferred for each rule separately and a share lambda_0 for the rules as a group; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### evidence_weighted_without_replacement — rejected (experiment2 round 0 candidate 3 refine incumbent without_replacement_example_sampling)

**Outcome:** predicts like existing model 'conservative_evidence_weighting' (p_yes RMSE 0.00093 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of conservative_evidence_weighting, not a new hypothesis.

**Hypothesis:** A refinement of without_replacement_example_sampling: people do the same Bayesian concept learning (45 independently weighted rules, ranges of 1-100 with a preferred typical width, an inferred share of random examples, examples taken as drawn from the concept without replacement), but they do not weigh the evidence of the examples at its normative strength relative to their prior over concepts. The one change is an inferred evidence-weight that scales how much the examples' fit counts against prior plausibility (below one is Edwards-style conservatism, staying nearer the prior and generalising more broadly; above one is over-weighting, locking onto the best-fitting small concept faster than Bayes implies; exactly one recovers the incumbent).

### decade_aligned_without_replacement — rejected (experiment2 round 0 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'without_replacement_example_sampling' (p_yes RMSE 0.00100 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of without_replacement_example_sampling, not a new hypothesis.

**Hypothesis:** I chose decade_aligned_range_concepts and refined it. People do the same Bayesian concept learning over the 45 independently weighted rules and all ranges of 1-100, with an inferred share of random examples, a prior favouring a typical range width, and a preference for ranges whose ends fall on decade boundaries ("the 30s", "1 to 20"), but they assume the program never produces the same number twice, so each further example that fits a concept is taken as one of its members not yet shown (sampling without replacement, grafted from without_replacement_example_sampling). The one change is this without-replacement size principle, which makes several examples that fall inside a small decade-aligned range (e.g. 32, 35, 37 inside the 30s) much stronger evidence for exactly that range, so "yes" stops more sharply at the decade boundary than under either parent model.

### any_of_sampled_concepts_yes_rule — rejected (experiment2 round 1 candidate 1 lens 8)

**Outcome:** predicts like existing model 'without_replacement_example_sampling' (p_yes RMSE 0.00084 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of without_replacement_example_sampling, not a new hypothesis.

**Hypothesis:** People do the same Bayesian concept learning as the current best model (45 independently weighted rules, ranges of 1-100 with a preferred typical width, an inferred share of random examples, examples taken as drawn without replacement), but they answer by consulting a small number of concepts sampled from their posterior and saying "yes" if any one of them admits the target (a lenient, disjunctive decision rule). The effective number of concepts consulted is inferred (one recovers probability matching to the posterior; more than one inflates "yes" for targets with moderate support while leaving clearly excluded targets near "no"; fewer than one makes them stricter), so the response stage distorts the posterior asymmetrically rather than sharpening it symmetrically or adding a flat guessing rate.

### asymmetric_range_exemplar_similarity — dropped (experiment2 round 2)

**Outcome:** fold 0 too slow to fit: the fit of 'asymmetric_range_exemplar_similarity' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People do not store a range of magnitudes as a sharp-edged interval; they store it as its member exemplars and judge a target outside the range by its similarity to the range's nearest stored member, which falls off smoothly with distance on the number line — and falls off more slowly below the range (toward smaller numbers) than above it. Arithmetic and digit rules are still matched by whether the target has the property, and which rule or range people retrieve is the same Bayesian concept learning as the current best model, so the one claim is this exemplar-similarity comparison at the edges of magnitude concepts, predicting graded "yes" just outside the examples' span that extends further downward than upward.

### count_up_to_n_concepts — rejected (experiment2 round 2 candidate 2 lens 1)

**Outcome:** predicts like existing model 'count_from_one_ceiling_concepts' (p_yes RMSE 0.00010 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of count_from_one_ceiling_concepts, not a new hypothesis.

**Hypothesis:** People do the same Bayesian concept learning as the current best model (45 independently weighted rules plus multiples of 13-50, ranges of 1-100 with a preferred typical width, an inferred share of random examples, examples drawn without replacement), but they also entertain counting concepts of the form "the numbers from 1 up to N" — the program counts up from 1 and stops somewhere — each stopping point equally likely a priori, the family taking an inferred share of the prior. Because such a concept always reaches down to 1 but stops at the largest example, it predicts that people extend "yes" below the smallest example far more readily than above the largest one; the sharpest disagreement with the best model's symmetric ranges lives on sets of small or clustered numbers with no shared rule (e.g. 12, 17, 20), where it predicts many "yes" answers to targets between 1 and the smallest example and few just above the largest.

### lower_anchored_range_concepts — rejected (experiment2 round 2 candidate 3 refine incumbent extended_multiples_concepts)

**Outcome:** predicts like existing model 'extended_multiples_concepts' (p_yes RMSE 0.00090 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of extended_multiples_concepts, not a new hypothesis.

**Hypothesis:** A refinement of extended_multiples_concepts: people do the same Bayesian concept learning (45 independently weighted rules plus multiples of 13-50, ranges with a preferred typical width, an inferred share of random examples, examples drawn without replacement), but they find "numbers up to N" ranges — ranges that start at 1 — a priori more (or less) natural than other ranges of the same width, because a program capped from above is a simple idea. The one change is this inferred prior boost for ranges anchored at 1, which predicts that people extend "yes" downward from the smallest example toward 1 more readily than upward beyond the largest example, the asymmetry the incumbent's symmetric ranges miss.

### up_to_n_ceiling_concepts — rejected (experiment2 round 2 candidate 4 refine incumbent extended_multiples_concepts)

**Outcome:** predicts like existing model 'count_from_one_ceiling_concepts' (p_yes RMSE 0.00112 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of count_from_one_ceiling_concepts, not a new hypothesis.

**Hypothesis:** A refinement of extended_multiples_concepts: people do the same Bayesian concept learning (45 independently weighted rules plus the multiples-of-13-to-50 family, ranges of 1-100 with a preferred typical width, an inferred share of random examples, examples drawn without replacement), but besides ordinary ranges they also entertain one-sided "numbers up to N" concepts that start at 1 (e.g. "anything up to 40"), which take an inferred share of the range prior on top of the ordinary ranges. The one change is this family of counting-from-one concepts, which predicts that people extend "yes" below the smallest example (toward 1) more readily than above the largest example, the asymmetry the incumbent's symmetric ranges and rules fail to reproduce.

### small_endpoint_range_prior — rejected (experiment2 round 2 candidate 5 refine chosen)

**Outcome:** predicts like existing model 'extended_multiples_concepts' (p_yes RMSE 0.00179 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of extended_multiples_concepts, not a new hypothesis.

**Hypothesis:** I chose log_number_line_interval_prior (pruned in experiment 1, 6.5 nats behind) and refined it. People picture ranges of numbers on a compressed mental number line, so before seeing any examples they find a range among small numbers more plausible than an equally wide range among large numbers. As a result, "yes" around examples that fit no rule reaches further below the smallest example than above the largest. The one change is the concept learner underneath. The old learner treated every range width as equally likely and drew examples with replacement. It is replaced by the current best learner (extended_multiples_concepts: 45 independently weighted rules plus a multiples-of-13-to-50 family, an inferred share of random examples, examples drawn without replacement, and a preferred typical range width). The magnitude preference, with its strength and sign inferred, now acts on top of that width prior, aimed at the downward-versus-upward extension gap the critique reports.

### magnitude_biased_guessing_lapse — rejected (experiment2 round 2 candidate 3 refine incumbent extended_multiples_concepts repair 1)

**Outcome:** predicts like existing model 'extended_multiples_concepts' (p_yes RMSE 0.00186 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of extended_multiples_concepts, not a new hypothesis.

**Hypothesis:** A refinement of extended_multiples_concepts: people do the same Bayesian concept learning (45 independently weighted rules plus the multiples-of-13-to-50 family, ranges with a preferred typical width, an inferred share of random examples, examples drawn without replacement), but on some share of trials they skip that inference and guess, and their guesses favour small numbers, which feel like more typical outputs of a number-producing program. The one change is this magnitude-biased guessing lapse (a response-stage lapse as in guessing_lapse_without_replacement, with a guess rate that depends on the target's size), which predicts that stray "yes" answers fall more on small targets below the examples than on large targets above them — the downward-over-upward extension the incumbent fails to reproduce.

### magnitude_biased_example_sampling — rejected (experiment2 round 2 candidate 5 refine chosen repair 1)

**Outcome:** too slow to fit: the fit of 'magnitude_biased_example_sampling' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** I chose the incumbent's learner, extended_multiples_concepts, and changed one thing: how people believe the program picks its examples from its concept. They do not assume every member is equally likely to be shown. They assume a bias by magnitude that favours the larger members (or the smaller ones, if the inferred bias is negative), with each example drawn without replacement in proportion to its weight. If people think the program favours large members, then the absence of small examples is unsurprising, so the size principle barely penalises concepts that reach well below the smallest example but fully penalises those that reach above the largest one; this predicts that "yes" extends further below the examples than above them, the asymmetry the critique reports.
