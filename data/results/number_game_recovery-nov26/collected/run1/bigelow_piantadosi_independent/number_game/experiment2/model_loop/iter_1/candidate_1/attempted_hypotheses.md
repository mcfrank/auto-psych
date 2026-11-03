# Tried before

26 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

- A *pruned* entry entered the set and later lost to the best model by the stated margin, on the data available then. Its mechanism may be partly right: a model that changes it substantively is welcome, but do not re-propose it unchanged or merely re-parameterised, under any name. Pruned models stay readable under `models/pruned/`.
- A *rejected* entry never entered the set. If it was a near-duplicate of a model still in the set, that region is already covered: do not re-propose it. If it failed on its code or its fit (see its outcome), the idea itself was never tested and a correct implementation may be worth trying.

### common_feature_matching — pruned (experiment1 end of experiment)

**Outcome:** 173.6 nats behind parity_restricted_range_conjunctions (5.8× dse)

**Hypothesis:** People abstract the set of features that every example has in common (parity, divisibility, being a square/cube/prime/power, final digit, decade) and say "yes" to a target in proportion to how many of those shared features it also has. This is a feature-matching (common-feature contrast) process with no size principle: a rule-like feature shared by all examples is extended to any number that has it, however far away, and adding examples only matters by removing features that are no longer shared.

### tempered_size_principle — pruned (experiment1 end of experiment)

**Outcome:** 188.7 nats behind parity_restricted_range_conjunctions (5.0× dse)

**Hypothesis:** People infer the hidden concept by Bayesian reasoning over the same mathematical rules and numerical intervals, but they apply the size principle only weakly: each example favours smaller concepts by a fractional power of their size rather than in full proportion. As a result broad rules such as "even" or "ends in 7" keep their strength against narrower rules and tight intervals, so the model disagrees most sharply with the current best model on far-away targets that fit a broad rule (e.g. distant even numbers for an all-even set), which it predicts people will endorse much more often, and on near-but-off-rule targets, which it predicts they will endorse less.

### common_feature_profile_matching — pruned (experiment1 end of experiment)

**Outcome:** 53.5 nats behind parity_restricted_range_conjunctions (3.1× dse)

**Hypothesis:** People judge membership by common-feature matching rather than by probabilistic concept induction: they notice which arithmetic and surface properties (parity, being a multiple, power, square or prime, a final digit, a digit it contains, its decade) every example shares, and say yes in proportion to how much of that shared profile the target also has, with more distinctive (rarer) shared properties counting more. Numerical proximity plays no role, and there is no size-principle tightening with more examples beyond what changes the shared profile.

### tempered_size_principle_mixture — pruned (experiment1 end of experiment)

**Outcome:** 45.5 nats behind parity_restricted_range_conjunctions (4.8× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they apply the size principle only partially — the evidence from the examples is tempered by a single fitted exponent, so a few examples that happen to sit close together do not overwhelm a broad rule such as "even" or "ends in 2" the way strict strong sampling would. The one change is this evidence-weighting exponent (sharpness of the size principle), which shifts belief toward broad rules when examples share a regularity and lets generalisation tighten less with more examples.

### tempered_size_principle_mixture_2 — rejected (experiment1 round 0 candidate 4 refine incumbent bigelow_piantadosi_mixture)

**Outcome:** predicts like existing model 'tempered_size_principle_mixture' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of tempered_size_principle_mixture, not a new hypothesis.

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do Bayesian concept learning over the same 45 arithmetic rules and all intervals of 1-100 (same prior), but they do not weigh the examples' evidence as an ideal learner would. The one change is that the size principle has a free strength: the evidence the examples give for each candidate concept is tempered (or sharpened) by a single exponent, so people may favour a broad rule such as "even numbers" over a tight interval around the examples more (or less) than strong sampling predicts — which governs how sharply they reject nearby numbers of the wrong parity and how far they extend a rule to distant numbers.

### tempered_size_principle_rules_intervals — pruned (experiment1 end of experiment)

**Outcome:** 77.8 nats behind parity_restricted_range_conjunctions (4.2× dse)

**Hypothesis:** Refinement of `tenenbaum_rules_similarity`: people do Bayesian concept learning over the same mathematical rules and numerical intervals, but they weigh the evidence of the examples less sharply than an ideal learner, so the size principle is tempered (the likelihood of the examples is raised to a fitted power below or above one). The one change is this fitted evidence-strength exponent: a weak size principle lets broad rules such as "even" or "ends in 2" keep their prior weight against tight intervals, so people extend rules to distant targets and tighten generalisation with more examples less than the ideal learner does.

### size_weighted_prior_mixture — pruned (experiment1 end of experiment)

**Outcome:** 79.9 nats behind parity_restricted_range_conjunctions (3.2× dse)

**Hypothesis:** Refinement of `bigelow_piantadosi_mixture`: people do the same Bayesian concept learning (same 45 rules, all intervals of 1-100, size-principle likelihood), but their prior is not flat within each kind of concept — it favours broader concepts in proportion to a fitted power of each concept's size. The one change is this size-weighted prior: broad rules such as "even" or "ends in 3" start out more plausible than narrow ones like "multiples of 4" or tight intervals, a fixed head start that (unlike a tempered size principle) does not grow or shrink with the number of examples, so people extend a shared regularity to distant numbers more often and endorse near numbers that break it less than the incumbent predicts.

### memory_limited_example_recall — pruned (experiment1 end of experiment)

**Outcome:** 48.5 nats behind parity_restricted_range_conjunctions (4.9× dse)

**Hypothesis:** People do Bayesian concept learning over arithmetic rules and numerical intervals, but they reason from an imperfect memory of the examples: each example is independently retained with some probability, and they generalise from only the examples they hold in mind. Because a judgement is often based on one or two of the examples rather than all of them, a target that shares a property (such as a final digit or a divisor) with just one example is often endorsed, and generalisation tightens less with more examples than an ideal learner who uses every example would predict.

### example_subset_memory_bayes — pruned (experiment1 end of experiment)

**Outcome:** 92.7 nats behind parity_restricted_range_conjunctions (6.1× dse)

**Hypothesis:** People do Bayesian concept learning over the same arithmetic rules and numerical intervals as an ideal learner (strong sampling, size principle), but under a working-memory limit: each example shown enters their inference only with some fitted probability, so they often generalise from just a subset of the examples. Reasoning from one or two remembered examples lets a rule shared by only some examples (e.g. "ends in 3" from 23 in {16, 23}) win, and weakens the size-principle tightening that more examples would normally produce.

### exemplar_digit_feature_similarity — pruned (experiment1 end of experiment)

**Outcome:** 136.6 nats behind parity_restricted_range_conjunctions (4.4× dse)

**Hypothesis:** People judge membership by surface-feature similarity to the individual examples, not by inducing a rule shared by the whole set: they compare the target with each example on a few salient digit and arithmetic features (its last digit, its tens digit, its parity, and its remainder when divided by three), and the more features the target shares with a given example the more similar it feels, with similarity falling off exponentially with the number of mismatching features. The yes probability grows with the target's average similarity to the examples, so a target that shares only its final digit with one example (in a set without a common final digit) still earns credit, and plain numerical distance plays no role beyond sharing a tens digit.

### exception_tolerant_tempered_mixture — pruned (experiment1 end of experiment)

**Outcome:** 11.9 nats behind parity_restricted_range_conjunctions (2.2× dse)

**Hypothesis:** Refinement of `tempered_size_principle_mixture`: people do the same tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they believe the program often slips and produces a number outside its concept, so they tolerate exceptions — the fraction of examples assumed to be off-concept noise is a fitted parameter instead of a fixed 10%. The one change is this fitted exception rate: when most but not all examples share a regularity (e.g. three of four end in 2), the rule that covers the majority survives the exception, so people extend a partly shared last digit or arithmetic property to distant targets, and lean less on tight numerical intervals, than a learner who treats every example as strictly from the concept.

### tempered_size_principle_digit_unions — pruned (experiment1 end of experiment)

**Outcome:** 49.9 nats behind parity_restricted_range_conjunctions (4.9× dse)

**Hypothesis:** Refinement of `tempered_size_principle_mixture`: people do the same tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but their rule vocabulary also includes disjunctive last-digit concepts such as "ends in 2 or 7" or "ends in 1, 6 or 0" (unions of two to four final digits), each made less plausible a priori by a fitted penalty per extra digit. The one change is this enlarged hypothesis space: when the examples' final digits differ, people can still infer a final-digit rule covering every example, so they say yes to distant targets that share a final digit with any one example far more than the incumbent predicts.

### graded_feature_profile_matching — pruned (experiment1 end of experiment)

**Outcome:** 13.4 nats behind parity_restricted_range_conjunctions; retired to keep the live set at 8 models: ELPD-LOO rank 11 of 11

**Hypothesis:** Refinement of `common_feature_profile_matching`: people still judge membership by matching the target against the arithmetic and surface properties of the examples (parity, multiples, powers, squares, primes, final digit, contained digits, decade), rarer properties counting more, but a property enters the profile in proportion to how many of the examples have it rather than only when every example has it. The one change is this graded, frequency-weighted profile (its steepness fitted by one exponent), so a target that shares a final digit or another property with only some of the examples still gains support — closing the critique that people endorse partial last-digit matches far more than the incumbent predicts.

### serial_position_weighted_evidence — pruned (experiment1 end of experiment)

**Outcome:** 10.1 nats behind parity_restricted_range_conjunctions; retired to keep the live set at 8 models: ELPD-LOO rank 9 of 11

**Hypothesis:** People do tempered Bayesian concept learning over arithmetic rules and numerical intervals, but they do not weigh the examples equally: how much each example counts as evidence about the concept depends on its position in the order shown, rising or falling smoothly from the first example to the last (primacy or recency, the direction and strength fitted). So the same examples shown in a different order lead to different generalisations — an example that breaks the others' pattern is discounted less when it comes in the favoured position, and people generalise around it more.

### numeral_edit_distance_similarity — pruned (experiment1 end of experiment)

**Outcome:** 227.6 nats behind parity_restricted_range_conjunctions (6.3× dse)

**Hypothesis:** People judge membership by how similar the target looks, as a written numeral, to each example: similarity falls off exponentially with the number of digit edits (substitutions, insertions or deletions of a digit) needed to turn the target's numeral into an example's, and the yes probability grows with the target's average similarity across the examples. Arithmetic rules and number-line magnitude play no separate role, so a target sharing its tens digit or its final digit with many of the examples (e.g. 62 for 92, 82, 42) is endorsed even when it is far away, while one sharing no digit position with any example is rejected.

### soft_threshold_posterior_decision — pruned (experiment1 end of experiment)

**Outcome:** 50.8 nats behind parity_restricted_range_conjunctions (4.2× dse)

**Hypothesis:** People infer the hidden concept by the same simple Bayesian concept learning over arithmetic rules and numerical intervals as the basic mixture model, but they do not answer "yes" with a probability equal to their belief: they compare their belief that the target is in the concept against an internal criterion and answer yes when it clears that criterion, with some noise around it. This soft-threshold decision rule makes answers more all-or-none than the graded posterior (near-unanimous yes for well-supported targets, near-unanimous no for weakly supported ones), with a fitted criterion setting an overall bias toward yes or no.

### decisive_exception_tolerant_mixture — pruned (experiment1 end of experiment)

**Outcome:** 12.0 nats behind parity_restricted_range_conjunctions; retired to keep the live set at 8 models: ELPD-LOO rank 10 of 11

**Hypothesis:** Refinement of `exception_tolerant_tempered_mixture`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they do not report their belief that the target is in the concept as a graded probability — they turn it into a near-categorical decision, so the odds of saying yes are the posterior odds raised to a fitted decisiveness exponent. The one change is this response-decisiveness transform: targets the learner thinks are probably in (or out of) the concept are endorsed (or rejected) more consistently than the posterior itself implies, making answers to each item more unanimous, sets more sharply split between narrow and broad generalisation, and near-but-off-rule neighbours of an outlier example rejected more firmly.

### decision_threshold_exception_tempered — pruned (experiment1 end of experiment)

**Outcome:** 17.7 nats behind parity_restricted_range_conjunctions (2.6× dse)

**Hypothesis:** Refinement of `exception_tolerant_tempered_mixture`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they do not answer yes with probability equal to their belief that the target is in the concept; instead they compare that belief against an internal criterion and respond near-deterministically, with a fitted decision sharpness and criterion. The one change is this thresholded (sharpened) decision rule on the posterior: beliefs above the criterion are pushed toward a confident yes and beliefs below it toward a confident no, so responses to each item are more unanimous, sets differ more in breadth, and endorsement falls off more steeply with numerical distance than the graded posterior itself implies.

### upward_counting_progressions — pruned (experiment1 end of experiment)

**Outcome:** 13.0 nats behind parity_restricted_range_conjunctions (2.7× dse)

**Hypothesis:** People do exception-tolerant, tempered Bayesian concept learning over arithmetic rules and numerical intervals, but their concept vocabulary also contains counting-up sequences: arithmetic progressions that start at some number and climb by a fixed step to 100 ("16, 24, 32, … upward"), this whole family sharing a fitted share of the prior. Because such a sequence must start at or below the smallest example and continues indefinitely upward, people fill in the gaps between the examples at the inferred spacing and extend their generalisation above the examples more readily than below them.

### set_adaptive_attention_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 132.2 nats behind parity_restricted_range_conjunctions (4.5× dse)

**Hypothesis:** People judge membership by summed similarity to the stored examples (an exemplar model), where the similarity metric adapts to the set: attention goes to the dimensions on which the examples agree — their final digit, parity, and remainders modulo 3, 4 and 5 — and to numerical magnitude in proportion to how tightly the examples cluster, with magnitude compared on a logarithmic (Weber-like) scale. A target is endorsed in proportion to its total similarity to the examples (Luce choice against a fixed criterion), so targets between several examples collect similarity from all of them, generalisation extends further upward than downward, and a final digit shared by every example becomes a near-mandatory match for distant targets.

### graded_profile_with_example_span — pruned (experiment1 end of experiment)

**Outcome:** 33.2 nats behind parity_restricted_range_conjunctions (2.7× dse)

**Hypothesis:** Refinement of `graded_profile_with_neighborhoods`: people still match the target against a frequency-weighted, distinctiveness-weighted profile of the examples' properties plus each example's soft numerical neighbourhood, but the profile also contains the numerical span the examples cover ("numbers between the smallest and largest example") as one more property, shared by every example, counting more the narrower the span, with soft edges that fade over the same fitted neighbourhood width. The one change is this span (interval) property: targets lying between the examples but far from any single one gain support, closing the critique that people generalise over the range spanned by the examples more than the incumbent predicts.

### graded_profile_with_spanned_interval — pruned (experiment1 end of experiment)

**Outcome:** 35.3 nats behind parity_restricted_range_conjunctions (2.8× dse)

**Hypothesis:** Refinement of `graded_profile_with_neighborhoods`: people match the target against the same frequency-weighted, distinctiveness-weighted profile of the examples' arithmetic and surface properties and each example's own soft numerical neighbourhood, but when there are two or more examples the profile also contains the numerical interval they span ("numbers between 23 and 41") as one more property, shared by every example, weighted by the same rarity rule (narrow spans count more), and with soft edges of the same fitted width. The one change is this spanned-interval property: targets lying between the examples gain support even when far from every individual example, closing the critique that people generalise over the examples' range more than the incumbent predicts.

### place_value_grid_regions — pruned (experiment1 end of experiment)

**Outcome:** 23.6 nats behind parity_restricted_range_conjunctions (2.9× dse)

**Hypothesis:** People do exception-tolerant, tempered Bayesian concept learning over arithmetic rules, but they represent magnitude on a decimal place-value grid (tens digit by ones digit) rather than on a continuous number line: their non-arithmetic concepts are rectangular regions of that grid — a range of tens digits crossed with a range of final digits, such as "the 40s to 90s ending in 2" or "the 20s and 30s" — weighed by the size principle. This disagrees most sharply with the current best (number-line interval) model on targets one step from an example across a decade boundary or a final-digit change (e.g. 26 or 28 for an outlier 27 among 92, 82, 42), which it predicts people reject more, and on distant targets sharing a tens or final digit with the examples, and filling the grid block the examples span, which it predicts people endorse more.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 277.6 nats behind parity_restricted_range_conjunctions (5.9× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Outcome:** 229.9 nats behind parity_restricted_range_conjunctions (7.4× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### first_example_anchored_concept — dropped (experiment2 round 0)

**Outcome:** fold 0 too slow to fit: the fit of 'first_example_anchored_concept' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People anchor their concept on the first example they see: they take the first example as more surely produced by the program than the later ones (by a fitted factor), treating mainly the later examples as possible slips, so a later example that breaks the pattern set up by the first is discounted as an exception while the first example itself is rarely explained away (otherwise they do the same exception-tolerant, tempered Bayesian concept learning over arithmetic rules and moderate-width ranges on a compressed number line). The same examples shown in a different order therefore lead to different generalisations: people generalise around the first example and rules containing it, and reject neighbours of an off-pattern example more when it comes later in the list.
