# Tried before

64 hypotheses proposed earlier in this project are no longer in the model set. Read them before you propose:

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

### graded_profile_with_neighborhoods — pruned (experiment2 end of experiment)

**Outcome:** 292.9 nats behind rule_plus_exception_neighbourhood (10.0× dse)

**Hypothesis:** Refinement of `graded_feature_profile_matching`: people still judge membership by matching the target against a frequency-weighted profile of the examples' properties (parity, multiples, powers, squares, primes, final digit, contained digits, decade), rarer properties counting more, but the profile also contains each example's own numerical neighbourhood ("numbers close to 27") as one more graded property, with a fitted width, weighted by the same distinctiveness rule and shared in proportion to how many examples fall near it. The one change is this soft neighbourhood property: targets just one to three away from an example gain strong support that decays steeply within a few units, closing the critique that human generalisation falls off over short numerical distances far more sharply than the current models predict.

### upward_counting_progressions — pruned (experiment1 end of experiment)

**Outcome:** 13.0 nats behind parity_restricted_range_conjunctions (2.7× dse)

**Hypothesis:** People do exception-tolerant, tempered Bayesian concept learning over arithmetic rules and numerical intervals, but their concept vocabulary also contains counting-up sequences: arithmetic progressions that start at some number and climb by a fixed step to 100 ("16, 24, 32, … upward"), this whole family sharing a fitted share of the prior. Because such a sequence must start at or below the smallest example and continues indefinitely upward, people fill in the gaps between the examples at the inferred spacing and extend their generalisation above the examples more readily than below them.

### set_adaptive_attention_exemplar — pruned (experiment1 end of experiment)

**Outcome:** 132.2 nats behind parity_restricted_range_conjunctions (4.5× dse)

**Hypothesis:** People judge membership by summed similarity to the stored examples (an exemplar model), where the similarity metric adapts to the set: attention goes to the dimensions on which the examples agree — their final digit, parity, and remainders modulo 3, 4 and 5 — and to numerical magnitude in proportion to how tightly the examples cluster, with magnitude compared on a logarithmic (Weber-like) scale. A target is endorsed in proportion to its total similarity to the examples (Luce choice against a fixed criterion), so targets between several examples collect similarity from all of them, generalisation extends further upward than downward, and a final digit shared by every example becomes a near-mandatory match for distant targets.

### log_compressed_number_line_intervals — pruned (experiment2 end of experiment)

**Outcome:** 230.7 nats behind rule_plus_exception_neighbourhood (8.9× dse)

**Hypothesis:** People represent numerical-magnitude concepts ("numbers between 40 and 60") as regions on a logarithmically compressed mental number line, not on the linear 1-100 scale: when they infer which stretch of numbers the program produces, a stretch's size, and how likely the examples are to fall where they do within it, are judged in compressed (Weber-like) units, while arithmetic and digit rules are judged as before. Because the same numerical step is subjectively smaller among large numbers, people extend a magnitude region upward beyond the largest example more readily than downward below the smallest, and generalise across the span between the examples.

### graded_profile_with_example_span — pruned (experiment1 end of experiment)

**Outcome:** 33.2 nats behind parity_restricted_range_conjunctions (2.7× dse)

**Hypothesis:** Refinement of `graded_profile_with_neighborhoods`: people still match the target against a frequency-weighted, distinctiveness-weighted profile of the examples' properties plus each example's soft numerical neighbourhood, but the profile also contains the numerical span the examples cover ("numbers between the smallest and largest example") as one more property, shared by every example, counting more the narrower the span, with soft edges that fade over the same fitted neighbourhood width. The one change is this span (interval) property: targets lying between the examples but far from any single one gain support, closing the critique that people generalise over the range spanned by the examples more than the incumbent predicts.

### graded_profile_with_spanned_interval — pruned (experiment1 end of experiment)

**Outcome:** 35.3 nats behind parity_restricted_range_conjunctions (2.8× dse)

**Hypothesis:** Refinement of `graded_profile_with_neighborhoods`: people match the target against the same frequency-weighted, distinctiveness-weighted profile of the examples' arithmetic and surface properties and each example's own soft numerical neighbourhood, but when there are two or more examples the profile also contains the numerical interval they span ("numbers between 23 and 41") as one more property, shared by every example, weighted by the same rarity rule (narrow spans count more), and with soft edges of the same fitted width. The one change is this spanned-interval property: targets lying between the examples gain support even when far from every individual example, closing the critique that people generalise over the examples' range more than the incumbent predicts.

### log_compressed_interval_sampling — pruned (experiment2 end of experiment)

**Outcome:** 247.6 nats behind rule_plus_exception_neighbourhood (8.9× dse)

**Hypothesis:** Refinement of `exception_tolerant_tempered_mixture`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, but they picture the examples of an interval concept as sampled along a compressed (roughly logarithmic) mental number line, so within a range larger numbers are individually less likely to be drawn and the size principle charges less for extending a range upward than downward. The one change is this compressed-number-line sampling of interval members (degree of compression fitted; no compression gives back essentially the original model), which predicts that people generalise past the largest example more than below the smallest, closing the critique that humans extend upward more than the symmetric models predict.

### place_value_grid_regions — pruned (experiment1 end of experiment)

**Outcome:** 23.6 nats behind parity_restricted_range_conjunctions (2.9× dse)

**Hypothesis:** People do exception-tolerant, tempered Bayesian concept learning over arithmetic rules, but they represent magnitude on a decimal place-value grid (tens digit by ones digit) rather than on a continuous number line: their non-arithmetic concepts are rectangular regions of that grid — a range of tens digits crossed with a range of final digits, such as "the 40s to 90s ending in 2" or "the 20s and 30s" — weighed by the size principle. This disagrees most sharply with the current best (number-line interval) model on targets one step from an example across a decade boundary or a final-digit change (e.g. 26 or 28 for an outlier 27 among 92, 82, 42), which it predicts people reject more, and on distant targets sharing a tens or final digit with the examples, and filling the grid block the examples span, which it predicts people endorse more.

### round_boundary_interval_prior — pruned (experiment2 end of experiment)

**Outcome:** 247.5 nats behind rule_plus_exception_neighbourhood (8.9× dse)

**Hypothesis:** People infer numerical-range concepts anchored on round numbers: when they consider that the program produces "a range of numbers", they think of ranges whose edges fall on decade boundaries ("from 20 to 50", "the 80s", "up to 100") as far more plausible than ranges with arbitrary edges such as 27-92 (the strength of this round-boundary preference fitted), while otherwise doing the same exception-tolerant, tempered Bayesian learning over arithmetic rules and ranges sampled along a compressed number line. Because the preferred range is snapped outward to the nearest decade edges, people fill in the span between the examples, generalise within the examples' decades but not across a decade boundary just beside an example, and extend above an example ending in a low digit more than below it.

### parity_restricted_range_conjunctions — pruned (experiment2 end of experiment)

**Outcome:** 230.0 nats behind rule_plus_exception_neighbourhood (8.4× dse)

**Hypothesis:** People can build compound concepts that join a numerical range with the examples' parity ("even numbers between 20 and 60"), but considering such a two-part concept carries a cognitive cost, so these conjunctions get only a fitted, reduced share of prior belief alongside the simple rules and plain ranges of exception-tolerant Bayesian concept learning on a compressed number line. Once considered, the size principle strongly favours these small compound concepts, so people fill in the range around the examples but reject immediate neighbours of the wrong parity (and odd-one-out neighbours of an outlier example) far more than a learner who only has plain ranges.

### erlang_width_compressed_intervals — pruned (experiment2 end of experiment)

**Outcome:** 135.4 nats behind rule_plus_exception_neighbourhood (6.2× dse)

**Hypothesis:** Refinement of `log_compressed_interval_sampling`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all intervals of 1-100, with interval members pictured as sampled along a compressed number line, but they do not consider every numerical range equally plausible beforehand — they expect a program that produces "a range of numbers" to produce a range of moderate width, so tiny ranges hugging one or two examples are a priori unlikely (an Erlang-shaped prior on range width, its typical width fitted). The one change is this width prior on intervals, replacing the flat prior: people fill in the whole span between the examples more, do not build a tight range around a lone exceptional example, and lean on parity or digit rules rather than a tiny range for a target just one step from an example.

### parity_restricted_compressed_intervals — pruned (experiment2 end of experiment)

**Outcome:** 231.6 nats behind rule_plus_exception_neighbourhood (8.5× dse)

**Hypothesis:** Refinement of `log_compressed_interval_sampling`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all ranges of 1-100, with range members pictured as sampled along a compressed mental number line, but their vocabulary of magnitude concepts also contains parity-restricted ranges ("even numbers between 20 and 40", "odd numbers between 51 and 67"), which share a fitted portion of the prior given to ranges. The one change is these conjunctive parity-and-magnitude concepts: when the examples share a parity, people generalise to nearby numbers of the same parity but reject immediate neighbours (which flip parity) and near numbers around an off-pattern example more than a learner with only plain ranges would.

### weber_scaled_profile_neighborhoods — pruned (experiment2 end of experiment)

**Outcome:** 379.4 nats behind rule_plus_exception_neighbourhood (11.6× dse)

**Hypothesis:** Refinement of `graded_profile_with_neighborhoods`: people still judge membership by matching the target against a frequency-weighted, distinctiveness-weighted profile of the examples' arithmetic and surface properties plus each example's own soft numerical neighbourhood, but that neighbourhood lives on a logarithmically compressed (Weber-like) mental number line rather than the linear 1-100 scale. The one change is this compressed neighbourhood (grafted from the compressed-number-line models): "close to 8" covers only a unit or two while "close to 80" covers many, the neighbourhood reaches further above an example than below it, and a broader neighbourhood counts as a less distinctive property — so people extend upward beyond the examples more than downward, as the critique of the incumbent reports.

### tenenbaum_rules_similarity — pruned (experiment1 end of experiment)

**Outcome:** 277.6 nats behind parity_restricted_range_conjunctions (5.9× dse)

**Hypothesis:** Bayesian concept learning over 33 mathematical rules (odd, even, squares, cubes, primes, multiples of 3-12, powers of 2-10, ending in 1-9) and every interval of 1-100. The rules share a prior fraction lambda equally; the intervals share the rest by an Erlang density on their size (scale sigma). Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### bigelow_piantadosi_mixture — pruned (experiment1 end of experiment)

**Outcome:** 229.9 nats behind parity_restricted_range_conjunctions (7.4× dse)

**Hypothesis:** Bayesian concept learning over 45 rules (odd, even, primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100. A rule is chosen with probability lambda_0, and the rules are equally likely; each interval has prior (1 - lambda_0) / 100^2. Examples are sampled from the concept (the size principle), 10% of them uniformly from 1-100, and "yes" is the posterior probability that the concept contains the target.

### rule_vs_range_size_principle — pruned (experiment2 end of experiment)

**Outcome:** 129.4 nats behind rule_plus_exception_neighbourhood (6.0× dse)

**Hypothesis:** Refinement of `erlang_width_compressed_intervals`: people do the same exception-tolerant Bayesian concept learning over the same 45 arithmetic and digit rules and width-weighted ranges on a compressed number line, but they do not apply a single size principle to both kinds of concept. When they weigh how well a symbolic rule ("ends in 2", "even", "multiples of 4") explains the examples, their sensitivity to its size differs, by a fitted factor, from their sensitivity when they weigh a numerical range. The one change is this kind-specific strength of the size principle, which decides whether more examples mainly tighten the range or mainly sharpen the choice between narrow and broad rules (such as "ends in 2" against "even").

### luce_concept_selection — pruned (experiment2 end of experiment)

**Outcome:** 491.5 nats behind rule_plus_exception_neighbourhood (9.7× dse)

**Hypothesis:** People do not average their predictions over every concept they find plausible; they commit to the concept that best explains the examples, choosing among the candidate rules and numerical ranges (same exception-tolerant learner over the 45 rules and compressed-line ranges with a moderate-width prior) by a Luce choice whose sharpness on the posterior is fitted, from soft averaging to near-single-hypothesis selection. Because the winning concept takes nearly all the weight, when every example shares a final digit or a parity people extend that rule almost unanimously and reject parity-violating neighbours more firmly than a hedging Bayesian learner would.

### first_example_anchored_concept — dropped (experiment2 round 0)

**Outcome:** fold 0 too slow to fit: the fit of 'first_example_anchored_concept' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People anchor their concept on the first example they see: they take the first example as more surely produced by the program than the later ones (by a fitted factor), treating mainly the later examples as possible slips, so a later example that breaks the pattern set up by the first is discounted as an exception while the first example itself is rarely explained away (otherwise they do the same exception-tolerant, tempered Bayesian concept learning over arithmetic rules and moderate-width ranges on a compressed number line). The same examples shown in a different order therefore lead to different generalisations: people generalise around the first example and rules containing it, and reject neighbours of an off-pattern example more when it comes later in the list.

### final_digit_salient_rules — pruned (experiment2 end of experiment)

**Outcome:** 13.4 nats behind rule_plus_exception_neighbourhood (3.3× dse)

**Hypothesis:** Refinement of `erlang_width_compressed_intervals`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and moderate-width ranges sampled along a compressed number line, but they do not consider all rules equally plausible beforehand — rules that can be read straight off a numeral's final digit (odd, even, "ends in d") are perceptually salient and get a fitted prior boost over the other arithmetic and digit rules. The one change is this final-digit salience in the rule prior: when all examples share a final digit or a parity, people commit to that rule more firmly than to a range, so they endorse distant same-ending and same-parity numbers more and reject opposite-parity neighbours (one step from an example) more than the incumbent predicts.

### family_salience_rule_prior — pruned (experiment2 end of experiment)

**Outcome:** 17.5 nats behind rule_plus_exception_neighbourhood (2.3× dse)

**Hypothesis:** Refinement of `erlang_width_compressed_intervals`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules and all ranges of 1-100 (members sampled along a compressed number line, Erlang-shaped width prior), but they do not find every kind of rule equally plausible beforehand — whole families of rules differ in salience (parity, final-digit, contained-digit, multiples, powers, and special sets such as primes or squares), with the share of prior belief each family receives fitted and split equally among its members. The one change is this family-salience prior over rules, replacing the flat one: if parity and final-digit rules are especially salient, people extend a shared final digit to distant targets almost unanimously and reject opposite-parity neighbours of same-parity examples more firmly than the incumbent predicts.

### parity_ranges_erlang_span_prior — pruned (experiment2 end of experiment)

**Outcome:** 136.0 nats behind rule_plus_exception_neighbourhood (6.2× dse)

**Hypothesis:** Refinement of `parity_restricted_compressed_intervals`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules, plain ranges and parity-restricted ranges ("even numbers between 20 and 40"), with range members pictured as sampled along a compressed number line, but they do not consider every range equally plausible beforehand — they expect a program that produces "a range" (plain or parity-restricted) to span a moderate stretch of the number line, so tiny ranges hugging one or two examples are a priori unlikely (an Erlang-shaped prior on the range's span, its typical span fitted, grafted from `erlang_width_compressed_intervals`). The one change is this span prior on all range concepts: people fill in the stretch between the examples with numbers of the examples' parity, reject parity-flipping immediate neighbours, and do not build a tight range around a lone off-pattern example, leaning on parity or digit rules instead.

### spread_adaptive_log_similarity — pruned (experiment2 end of experiment)

**Outcome:** 876.8 nats behind rule_plus_exception_neighbourhood (17.8× dse)

**Hypothesis:** People judge whether the program would produce a target purely by how close it lies to the examples on a logarithmically compressed mental number line, with an exponential (Shepard-style) generalization gradient around each example whose breadth is set by how spread out the examples are: tightly clustered examples yield a narrow gradient, widely scattered examples a broad one. Endorsement is the probability that at least one example "captures" the target under this gradient, so targets right at or between nearby examples are endorsed, generalization reaches further above an example than below it, and arithmetic or digit regularities play no role.

### recognition_shortcut_for_examples — pruned (experiment2 end of experiment)

**Outcome:** 229.7 nats behind rule_plus_exception_neighbourhood (9.0× dse)

**Hypothesis:** People answer by a recognition shortcut before any concept inference: when the number asked about is one they were just shown, they recognise it and answer yes directly, with a fixed (fitted) reliability, instead of consulting their inferred concept. Only for numbers they were not shown do they fall back on simple Bayesian concept learning over arithmetic rules and magnitude ranges, so shown examples are endorsed near-uniformly even when they are off-pattern outliers that the inferred concept would treat as exceptions.

### rule_plus_exception_concepts — pruned (experiment2 end of experiment)

**Outcome:** 8.6 nats behind rule_plus_exception_neighbourhood (2.8× dse)

**Hypothesis:** Refinement of `final_digit_salient_rules`: people do the same tempered, exception-tolerant Bayesian concept learning over the same 45 rules (final-digit rules salient) and moderate-width ranges on a compressed number line, but their concept vocabulary also contains "rule plus one exception" concepts — a rule together with a single extra number it does not contain ("ends in 2, and also 27"), like the classic "powers of 2, or 37" — which share a fitted, reduced part of each rule's prior. The one change is these explicit rule-with-exception concepts: when one example breaks the others' pattern, people treat that example as a genuine member of the program's output rather than a random slip, so they endorse the off-pattern example itself almost unanimously while still rejecting its numerical neighbours and extending the rule to distant numbers.

### example_recognition_digit_salient — pruned (experiment2 end of experiment)

**Outcome:** 10.2 nats behind rule_plus_exception_neighbourhood (2.6× dse)

**Hypothesis:** Refinement of `final_digit_salient_rules`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules (final-digit rules made more salient) and moderate-width ranges on a compressed number line, but when the number asked about is one of the examples they were just shown, they often simply recognise it from memory and say yes outright — it was produced, so the program produces it — instead of judging it from their inferred concept. The one change is this fitted recognition probability for shown examples: people endorse the examples themselves more than an exception-tolerant learner (who suspects any example might be off-concept noise) would, and their tolerance of exceptions then governs only how they generalise to new numbers.

### example_recognition_salient_rules — rejected (experiment2 round 1 candidate 4 refine incumbent final_digit_salient_rules)

**Outcome:** predicts like existing model 'example_recognition_digit_salient' (p_yes RMSE 0.00000 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of example_recognition_digit_salient, not a new hypothesis.

**Hypothesis:** Refinement of `final_digit_salient_rules`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules (final-digit rules made salient) and Erlang-width ranges on a compressed number line, but when the target is one of the examples they were just shown, they also recognise it from memory as a number the program has already produced, and with a fitted probability answer yes on that recognition alone, whatever concept they inferred. The one change is this episodic recognition of shown examples: people endorse the examples themselves almost unanimously, even an off-pattern example their inferred concept treats as an exception, more than the incumbent's concept-membership belief predicts.

### example_recognition_family_salience — pruned (experiment2 end of experiment)

**Outcome:** 16.0 nats behind rule_plus_exception_neighbourhood (2.2× dse)

**Hypothesis:** Refinement of `family_salience_rule_prior`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules (with fitted family salience) and Erlang-width ranges on a compressed number line, but when the target is one of the shown examples they also use what they remember directly — that the program has already produced this very number — so even when their inferred concept leaves it out (as an exception) they still answer yes with a fitted recognition probability. The one change is this episodic recognition of shown examples, which separates "is it in the concept?" from "has the program produced it?": people endorse the examples themselves more than concept membership alone implies (closing the critique that the best model under-predicts yes to example targets), and the learner can treat off-pattern examples as exceptions without that also making it reject those examples.

### redundant_example_evidence — rejected (experiment2 round 1 candidate 4 refine incumbent final_digit_salient_rules repair 1)

**Outcome:** too slow to fit: the fit of 'redundant_example_evidence' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `final_digit_salient_rules`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules (final-digit rules salient) and moderate-width ranges on a compressed number line, but they treat the examples of a set as partly redundant evidence rather than independent draws, so each additional example adds less weight to the size principle than the one before (the strength of this discount fitted). The one change is this diminishing per-example evidence: with three or four examples people tighten their generalisation, and shift from broad rules and wide ranges to narrow ones, less than a learner who counts every example in full, while a single example is weighed just as the incumbent weighs it.

### pairwise_exemplar_consequential_regions — pruned (experiment2 end of experiment)

**Outcome:** 634.6 nats behind rule_plus_exception_neighbourhood (15.1× dse)

**Hypothesis:** People do not infer one concept that explains all the examples together. They keep each example in memory as a separate exemplar and compare the target with each one in turn, judging how likely the two numbers are to fall in a common consequential region — a shared arithmetic or digit rule, or a stretch of the number line of moderate width, where smaller shared regions count for more — and their yes probability is the average of these pairwise similarities across the stored examples. Because every example generalises on its own and no example is ever discounted as an exception, people endorse numbers around an off-pattern example (e.g. neighbours of the one odd number in an even set) far more than a joint concept learner predicts, and more examples do not tighten generalisation.

### two_source_example_partition — pruned (experiment2 end of experiment)

**Outcome:** 10.5 nats behind rule_plus_exception_neighbourhood (3.1× dse)

**Hypothesis:** People believe the hidden program may draw its outputs from more than one source, so they spontaneously split the examples into groups: when one example does not fit the others they can attribute it to a second, separate source (with a fitted prior readiness to posit such a split, weighed by how much better two sources explain the examples than one), and they infer a concept for each source — rules or numerical ranges, by the same exception-tolerant, tempered Bayesian concept learning with salient final-digit rules on a compressed number line. A target is endorsed if either source would produce it, so people generalise around an off-pattern example in its own right (saying yes to its numerical neighbours and its own rules) while still extending the majority's rule to distant numbers.

### rule_plus_exception_range — dropped (experiment2 round 2)

**Outcome:** fold 0 too slow to fit: the fit of 'rule_plus_exception_range' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `final_digit_salient_rules`: people do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules (final-digit rules salient) and moderate-width ranges on a compressed number line, but their vocabulary also has "rule plus an exception range" concepts. These are a rule together with one short run of two to seven consecutive numbers ("ends in 2, and also the numbers around 27"), and they take a fitted share of each rule's prior, with shorter runs more plausible. The one change is these rule-plus-run concepts: when one example breaks the others' pattern, people treat the stretch around it as part of the program's output, not just the single number, so they endorse that off-pattern example and its close neighbours of either parity while still extending the rule to distant numbers.

### arithmetic_progression_residue_rules — pruned (experiment2 end of experiment)

**Outcome:** 15.8 nats behind rule_plus_exception_neighbourhood (3.6× dse)

**Hypothesis:** People do the same exception-tolerant, tempered Bayesian concept learning over the same 45 rules (final-digit rules salient) and moderate-width ranges on a compressed number line, but they also notice regular spacing: their concept vocabulary contains "every m-th number" progressions that do not start at zero ("5, 11, 17, … — numbers that leave remainder 5 when divided by 6", for steps 3 to 12), which share a fitted part of the rules' prior. The one change is these equal-step progression concepts: when the examples are evenly spaced off the multiples (e.g. 53, 47, 41 or 34, 26 or 15, 11), people endorse distant numbers on the same step (35, 59, 65; 18, 42, 50) and reject the in-between same-parity numbers far more than the best model, which can only explain such sets by a range or a parity rule.

### moderate_width_exception_patch — dropped (experiment2 round 3)

**Outcome:** fold 0 too slow to fit: the fit of 'moderate_width_exception_patch' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People do exception-tolerant, tempered Bayesian concept learning over arithmetic and digit rules (final-digit rules salient) and moderate-width ranges on a compressed number line, and they can explain an example that breaks a rule's pattern as a "rule plus an exceptional patch" concept — but they picture that patch the way they picture any stretch of the number line: as a region of moderate width, not as a lone point. Their expectations about the patch's width are Erlang-shaped (tiny patches and isolated single numbers are a priori unlikely, with the typical width fitted), so when one example breaks the others' pattern people endorse its numerical neighbours about as readily as the off-pattern example itself, while still rejecting rule-violating neighbours of the majority examples and extending the rule to distant numbers.

### example_cued_exception_generation — dropped (experiment2 round 3)

**Outcome:** fold 0 too slow to fit: the fit of 'example_cued_exception_generation' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People generate "rule plus exception" concepts bottom-up from the examples rather than holding them in their vocabulary in advance: while checking a candidate rule against the examples, an example that the rule fails to cover captures attention, and the learner then entertains the rule together with a small fuzzy region around that particular noticed example ("ends in 2, and also around 27"), giving such example-cued exception concepts a fitted, substantial share of the rule's belief instead of the tiny share an exception at an arbitrary number would get. Otherwise they do the same tempered, exception-tolerant Bayesian concept learning over the same rules (final-digit rules salient) and moderate-width ranges on a compressed number line, so when one example breaks the others' pattern people endorse its close neighbours of either parity much more than they endorse equally close rule-violating neighbours of the conforming examples, while for sets that all fit a rule no exception concepts arise at all.

### unpenalised_exception_neighbourhood — pruned (experiment2 end of experiment)

**Outcome:** 1.9 nats behind rule_plus_exception_neighbourhood (2.6× dse)

**Hypothesis:** People infer the hidden program's concept the way a tempered, exception-tolerant Bayesian learner does (the same 45 rules with salient final-digit rules, moderate-width ranges on a compressed number line, and "rule plus a fuzzy neighbourhood exception" concepts), with one distortion: they apply the size principle only to the regular part of a concept and not to its exception. When one example breaks the others' pattern, they explain it with a fuzzy region around it but do not count the region's other numbers against how well the concept explains the examples. As a result they generalise freely to the off-pattern example's close neighbours, nearly as much as to the example itself, while still rejecting rule-violating neighbours of the majority examples and extending the rule to distant numbers.

### two_part_rule_exception_sampling — pruned (experiment2 end of experiment)

**Outcome:** 14.4 nats behind rule_plus_exception_neighbourhood (2.8× dse)

**Hypothesis:** Refinement of `rule_plus_exception_neighbourhood`: people consider the same "rule plus a small fuzzy exception region" concepts ("ends in 2, and also numbers around 27") alongside the same rules and ranges, learned by the same tempered, exception-tolerant Bayesian inference. What changes is how they picture such a program producing numbers. It is not drawing evenly from the rule and the region pooled together. It is a two-part program: it mostly outputs rule members and, a fitted share of the time, outputs a number from the exception region. The one change is this two-part sampling. A wider exception region therefore makes only the off-pattern example less expected, not the rule-following ones, so people endorse the off-pattern example's close neighbours (of either parity) much more than numbers that are merely close to a majority example, as the critique reports, while still extending the rule to distant numbers.

### adjacent_number_confusion — pruned (experiment2 end of experiment)

**Outcome:** 15.8 nats behind rule_plus_exception_neighbourhood (3.9× dse)

**Hypothesis:** People infer the program's concept the way an exception-tolerant, tempered Bayesian learner does (arithmetic and digit rules with salient final-digit rules, moderate-width ranges on a compressed number line), but they hold the number asked about only approximately: with a fitted probability they judge an adjacent number (one above or one below) instead of the target itself. Under this approximate-number confusion, a target next to any example "borrows" that example's endorsement, whatever its parity or rule membership. So people endorse the immediate neighbours of examples, including the off-pattern example's neighbours, more than concept membership implies, and a number sitting just beside a rule the examples share is endorsed more than distant numbers that break it.

### first_example_anchored_neighbourhood — rejected (experiment2 round 4 candidate 1 lens 6)

**Outcome:** predicts like existing model 'rule_plus_exception_neighbourhood' (p_yes RMSE 0.00120 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of rule_plus_exception_neighbourhood, not a new hypothesis.

**Hypothesis:** People anchor their concept on the first example they are shown: they trust the first example as a genuine output of the program differently (by a fitted amount, expected to be more) than the later examples, which they more readily explain away as slips, while otherwise doing the same exception-tolerant Bayesian learning over rules, ranges and rule-plus-neighbourhood-exception concepts as the best model. The order of examples therefore matters: an off-pattern example shown later is discounted and its neighbours rejected, whereas an off-pattern example shown first is taken as genuine and generalised around.

### place_value_grid_similarity — pruned (experiment2 end of experiment)

**Outcome:** 786.6 nats behind rule_plus_exception_neighbourhood (17.1× dse)

**Hypothesis:** People judge whether the program would produce a target by how close it looks to the examples on a decimal place-value grid rather than on the number line: each numeral is a position given by its tens digit and its ones digit, and similarity to an example falls off exponentially with the city-block distance between them on that grid, with separately fitted sensitivities to tens-digit and ones-digit differences. The yes probability grows with the target's average grid similarity to the examples, so a distant target sharing an example's final digit (same column) or decade (same row) is endorsed, while a numerically adjacent target across a decade boundary (29 versus 30) is judged dissimilar and rejected.

### rule_family_salience_neighbourhood — dropped (experiment2 round 4)

**Outcome:** fold 0 too slow to fit: the fit of 'rule_family_salience_neighbourhood' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `rule_plus_exception_neighbourhood`: people do the same tempered, exception-tolerant Bayesian concept learning over the same 45 rules, moderate-width ranges on a compressed number line, and "rule plus a small fuzzy exception region" concepts. The one change, grafted from `family_salience_rule_prior`, is that their prior over rules is not just "final-digit rules are salient, all others equal": each family of rules (parity, final digit, contained digit, multiples, powers, special sets such as primes or squares) has its own fitted salience. Rules readable straight off the numeral's digits (parity, final digit, and also a digit the number contains) come to mind far more readily than arithmetic ones, so people extend a shared contained digit to distant numbers more, and extend a less familiar arithmetic regularity (such as "multiples of 7" for 14, 98, 91) less, than the incumbent predicts.

### iter4_candidate4 — dropped (experiment2 round 4)

**Outcome:** fold 0 too slow to fit: the fit of 'iter4_candidate4' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** Refinement of `rule_plus_exception_neighbourhood`: people do the same tempered, exception-tolerant Bayesian concept learning over the same 45 rules (final-digit rules salient), moderate-width ranges on a compressed number line, and "rule plus a small fuzzy neighbourhood exception" concepts ("ends in 2, and also numbers around 27", radius zero to three, smaller radii likelier), but when they code the off-pattern example as an exception they can alternatively code it at the place-value level, as the whole decade it falls in ("ends in 2, and also the twenties"), with a fitted share of the exception prior going to these rule-plus-decade concepts (grafted from `rule_plus_decade_exception`). The one change is this second, decade-level format for the exception: when one example breaks the others' pattern, people endorse its immediate neighbours on both sides and also its farther same-decade neighbours, while endorsing numbers a few steps away across a decade boundary less, and still extending the rule to distant numbers.

### sequential_win_stay_lose_shift — rejected (experiment2 round 4 candidate 1 lens 6 repair 1)

**Outcome:** too slow to fit: the fit of 'sequential_win_stay_lose_shift' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People do not weigh all the examples at once; they hold one working guess about the concept and revise it example by example in the order shown (win-stay, lose-shift). After the first example they pick a concept (a rule or a moderate-width range) in proportion to how plausible it is; when the next example fits their current guess they keep it unexamined, and only when an example contradicts it do they (usually, at a fitted rate) drop it and re-pick a concept in light of all the examples seen so far. So early examples lock in broad concepts that later consistent examples never tighten, and the same examples in a different order lead to different generalisations: an off-pattern example forces a revision only if it arrives after a guess it contradicts.

### attentional_lapse_guessing — dropped (experiment3 round 0)

**Outcome:** fold 2 too slow to fit: the fit of 'attentional_lapse_guessing' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People infer the program's concept exactly as the current best learner does (tempered, exception-tolerant Bayesian concept learning over rules, moderate-width ranges and example-anchored rule-plus-neighbourhood concepts), but their answer does not always come from that inference: on some trials attention lapses and they respond by guessing, so even numbers their concept clearly excludes get a residual rate of "yes" (a floor) and numbers it clearly includes get a residual rate of "no" (a ceiling below one), the two rates fitted separately. The decision rule, not the inference, is what changes: endorsement is compressed toward the middle by a constant, belief-independent guessing component, most visibly for the far-off and the clearly-in targets.

### rule_or_range_union — dropped (experiment3 round 0)

**Outcome:** fold 2 too slow to fit: the fit of 'rule_or_range_union' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People infer the program's concept by tempered, exception-tolerant Bayesian learning over arithmetic and digit rules (final-digit rules salient) and moderate-width ranges on a compressed number line, and their concept vocabulary also contains unions of a rule with a stretch of the number line — "ends in 2, or anything from 24 to 31" — built around the examples the rule fails to cover. They expect such an add-on stretch to be a short range of moderate width placed anywhere that covers the off-rule example (not necessarily centred on it, with point-sized and very wide add-ons a priori less likely), so when one example breaks the others' shared pattern people endorse a whole run of numbers around that outlier, of either parity, while still extending the rule to distant numbers and rejecting rule-breaking neighbours of the majority examples.

### discounted_anchored_exception_region — rejected (experiment3 round 0 candidate 3 refine incumbent example_anchored_exception_neighbourhood)

**Outcome:** predicts like existing model 'example_anchored_exception_neighbourhood' (p_yes RMSE 0.00132 < 0.002 on the 512-stimulus novelty pool) — a near-duplicate of example_anchored_exception_neighbourhood, not a new hypothesis.

**Hypothesis:** Refinement of `example_anchored_exception_neighbourhood`: people consider the same "rule plus a small fuzzy neighbourhood around an off-pattern example" concepts (anchored only on shown examples the rule misses) alongside the same rules and ranges, learned by the same tempered, exception-tolerant Bayesian inference, but when they weigh how well such a concept explains the examples they charge it in full for the rule and for the off-pattern examples they actually saw, and only partly (by a fitted fraction) for the never-shown numbers in the fuzzy region. The one change is this discounted size-principle charge on the exception region: a wider region around the outlier costs little evidence, so people endorse the outlier's close neighbours, falling off with distance, even more than off-rule neighbours of the majority examples (closing the critique's outlier-versus-majority neighbour gap), while still extending the rule to distant numbers.

### pedagogical_teacher_sampling — rejected (experiment3 round 1 candidate 0 lens 0)

**Outcome:** too slow to fit: the fit of 'pedagogical_teacher_sampling' (target_accept 0.99) was still sampling after the 30-minute limit and was stopped. Every sampling run of a candidate's admission fit has a 30-minute limit. Make the model cheaper to evaluate and easier to sample: vectorise the likelihood over trials (no Python loops, pytensor scan or per-trial subgraphs), compute features once per unique sequence (in compute_features or prepare_observed, not in the graph), and drop or merge parameters the data barely constrain — a weakly identified posterior makes NUTS take maximal-length trajectories.

**Hypothesis:** People assume the examples were picked by a helpful teacher who wanted them to guess the program, not drawn at random from it (pedagogical sampling): when they ask how well a candidate concept explains the examples, they expect a teacher of that concept to have chosen numbers that point unambiguously to it and to avoid numbers that also belong to many other concepts (such as 16 or 64, which are powers, squares and even). So an ambiguous "special" number among the examples counts as strong evidence for the rule it is special for and as weak evidence for a mere range around it, while unremarkable numbers favour ranges — which changes how broadly people generalise from each set, in a way no random-sampling learner can express.
